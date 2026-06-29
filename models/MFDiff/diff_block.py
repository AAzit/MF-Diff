import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from math import sqrt
import scipy.sparse as sp

import torch.nn.init as init

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # (B, L, heads, D)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # (B, heads, L, L)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)  # (B, L, heads, D)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        # x:(B, L, channels)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # (B, L, heads, D)
        keys = self.key_projection(keys).view(B, S, H, -1)  # (B, L, heads, D)
        values = self.value_projection(values).view(B, S, H, -1)    # (B, L, heads, D)

        # out:(B, L, heads, D)
        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, L, -1)    # (B, L, heads*D)

        # out :(B, L, channels)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, support=None, itp_x=None):
        # x:(B, L, channels)

        if itp_x is not None:
            # itp cross-attn
            new_x = self.attention(
                itp_x, x, x,
                attn_mask=attn_mask,
            )   # new_x:(B, L, channels)

        else:
            # self-attn
            new_x = self.attention(
                x, x, x,
                attn_mask=attn_mask,
            )   # new_x:(B, L, channels)

        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # (B, L, dff)
        y = self.dropout(self.conv2(y).transpose(-1, 1))    # (B, L, channels)

        return self.norm2(x + y)


# in Graph-wavenet
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support


class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, support):
        x_in = x    # x:(B*L, N, channels)

        # if K == 1:
        if x.shape[1] == 1:
            return x

        x = x.permute(0, 2, 1)  #(B*L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)  #(B*L, channel, K)->(B*L, channel, K, 1)
        else:
            squeeze = False
        out = [x] if self.include_self else []  # out[x(B*L, channel, K, 1)]

        for a in support:
            if a.dim() == 2:    # (N,N)
                x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous() # (B*L,channel,N,1)*(N,N)->(B*L,channel,N,1)
            else: # a.dim() == 3    (B*L,N,N)
                x1 = torch.einsum('ncvl,nwv->ncwl', (x, a)).contiguous() # (B*L,channel,N,1)*(N,N)->(B*L,channel,N,1)
            out.append(x1)
            for k in range(2, self.order + 1):
                if a.dim() == 2:    # (N,N)
                    x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()    # (B*L,channel,N,1)
                else: # a.dim() == 3    (B*L,N,N)
                    x2 = torch.einsum('ncvl,nwv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1) # (B*L,channel*m,N,1)
        out = self.mlp(out) # (B*L,channel*m,N,1)->(B*L,channel,N,1)
        if squeeze:
            out = out.squeeze(-1)   # (B*L,channel,N)
        out = out.permute(0, 2, 1)
        return out


class EncoderGraphLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", adj_mx=None):
        super(EncoderGraphLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.GCN = AdaptiveGCN(channels=d_model)
        self.norm1_local = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, support=None, itp_x=None):
        # x:(B*L, N, channels)

        y_in1 = x
        y_local = self.GCN(x, support) 
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)


        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask,
        )   # new_x:(B*L, N, channels)
        y_attn = x + self.dropout(new_x)
        y_attn = self.norm1(y_attn)

        y = y_local + y_attn
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # (B*L, N, channels)
        y = self.dropout(self.conv2(y).transpose(-1, 1))    # (B*L, N, channels)

        return self.norm2(x + y)


class TransformerEncoder(nn.Module):
    """
    based on iTransformer
    """
    def __init__(self, attn_layers, norm_layer=None):
        super(TransformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, support=None, itp_x=None):   
        # x:(B*N, L, channels),以下将B*N用B代替
        # x:(B, L, channels)
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask, support=support, itp_x=itp_x)    # x:(B, L, channels)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DiffusionEmbedding(nn.Module):

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlock(nn.Module):

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, adj_mx=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = TransformerEncoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(attention=FullAttention(), d_model=channels, n_heads=nheads),
                    d_model=channels,
                    activation='gelu'
                ) for l in range(1) # e_layers = 1
            ],
            norm_layer=torch.nn.LayerNorm(channels)
        )

        self.feature_layer = TransformerEncoder(
            attn_layers=[
                EncoderGraphLayer(
                    attention=AttentionLayer(attention=FullAttention(), d_model=channels, n_heads=nheads),
                    d_model=channels,
                    activation='gelu',
                    adj_mx=adj_mx
                ) for l in range(1) # e_layers = 1
            ],
            norm_layer=torch.nn.LayerNorm(channels)
        )


    def forward_time(self, y, base_shape, itp_x=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if itp_x is not None:   # itp cross-attn
            itp_x = itp_x.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            y = self.time_layer(y.permute(0, 2, 1), itp_x=itp_x.permute(0, 2, 1)).permute(0, 2, 1)  # transformer input (B*N, L, channels)
        else:   # self-attn
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # transformer input (B*N, L, channels)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, support):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)  # (B*L, channels,N)
        y = self.feature_layer(y.permute(0, 2, 1), support=support).permute(0, 2, 1)  # transformer_input (K, B*L, channels)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, support, itp_x):

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb  # y(B, channels, N*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # cond_info(B, cond_info_dim, N*L)-->cond_info(B,channel,K*L)
        y = y + cond_info.reshape(B, channel, K * L)

        itp_x = itp_x.reshape(B, channel, K * L)
        itp_x = itp_x + diffusion_emb
        itp_x = itp_x + cond_info.reshape(B, channel, K * L)

        y = self.forward_time(y, base_shape, itp_x)    # (B,channel,K*L)
        y = self.forward_feature(y, base_shape, support)  # (B,channel,K*L)
        y = self.mid_projection(y)  # y(B,channel,K*L)-->(B,2*channel,N*L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)   # (B,channel,K*L)-->(B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)  # (B,channel,K,L)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)  # (B,channel,K,L)
        return (x + residual) / math.sqrt(2.0), skip


class NoiseEstimator(nn.Module):
    def __init__(self, layers):
        super(NoiseEstimator, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond_info, diffusion_emb, support, itp_x):
        skips = []
        for layer in self.layers:
            x, skip = layer(x, cond_info, diffusion_emb, support, itp_x)
            skips.append(skip)
        skip_concat = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.layers))
        B, _, K, L = skip_concat.shape
        return skip_concat.reshape(B, -1, K * L)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)
        self.output_projection2 = Conv1d_with_init(channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

    def forward(self, x_hidden, B, K, L):  # (B, channel, K*L) => (B, K, L)
        x = self.output_projection1(x_hidden)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class denoising_network(nn.Module):

    def __init__(self, config, inputdim=2, adj_mx=None):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.seqlen = config["seqlen"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        # compute supports for gwn-------------------
        device = adj_mx.device
        self.support = compute_support_gwn(np.array(adj_mx.cpu()), device=device)   # [adj1(N,N), adj2(N,N)]
        self.support.append([self.nodevec]) # [adj1(N,N), adj2(N,N)]

        self.encoder = NoiseEstimator(
            nn.ModuleList(
                [
                    ResidualBlock(
                        side_dim=config["side_dim"],
                        channels=self.channels,
                        diffusion_embedding_dim=config["diffusion_embedding_dim"],
                        nheads=config["nheads"],
                    )
                    for _ in range(config["layers"])
                ]
            )
        )
        self.decoder = Decoder(self.channels)

        # --------------------------itp module----------------------------
        import pickle as pk
        from itp_module import MSFE
        self.itp_construction = MSFE(
            channels=config["channels"], 
            heads=config["nheads"], 
            seqlen=config["seqlen"],
            DWT_level=config["DWT_level"],
            )
        
        dataset = config["dataset"]
        with open(f"../datasets/{dataset}/{dataset}_meanstd.pk", "rb") as fb:
            mean, std = pk.load(fb)
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)

        self.itp_projection = Conv1d_with_init(inputdim, self.channels, 1)

    def forward(
        self,
        x, 
        cond_info,
        diffusion_step,
        itp_x,
        DWT_x,
    ):
        B, inputdim, K, L = x.shape

        itp_x = self.itp_construction(itp_x, DWT_x)
        # --------------------------------------------------

        x_hidden = self.__embedding(x, B, K, L, inputdim)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        forward_noise_hidden = self.encoder(x_hidden, cond_info, diffusion_emb, self.support, itp_x)

        forward_noise = self.decoder(forward_noise_hidden, B, K, L) # (B, channels, N*L)-->(B, 1, N*L)

        return (
            forward_noise
        )

    def impute(self, x, cond_info, diffusion_step, itp_x, DWT_x):
        B, inputdim, K, L = x.shape

        itp_x = self.itp_construction(itp_x, DWT_x)

        x_enc = self.__embedding(x, B, K, L, inputdim)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        x_hidden = self.encoder(x_enc, cond_info, diffusion_emb, self.support, itp_x)

        x_noise = self.decoder(x_hidden, B, K, L)
        return x_noise

    # Private helper functions
    def __embedding(self, x, B, K, L, input_dim):
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        return x
