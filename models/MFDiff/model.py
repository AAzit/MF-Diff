import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diff_block import denoising_network


class MFDiff_base(nn.Module):

    def __init__(self, target_dim, config, device, adj_mx):
        super().__init__()
        self.device = device
        self.target_dim = target_dim  # target_dim = number of features

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["nodeemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        ) 

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1
        self.diffmodel = denoising_network(config_diff, input_dim, adj_mx)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

        self.is_fast_sampling = config["model"]["is_fast_sampling"]
        self.eta = config["model"]["eta"]
        self.K_fast = config["model"]["K_fast"]
        

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(
        self, observed_tp: torch.Tensor, cond_mask: torch.Tensor
    ) -> torch.Tensor:
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(
            observed_tp, self.emb_time_dim
        )  # (B, L, emb_time)
        time_embed = time_embed.unsqueeze(2).expand(
            -1, -1, K, -1
        )  # (B, L, K, emb_time)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb_feature)
        feature_embed = (
            feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        )  # (B, L, K, emb_feature)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B, L, K, emb_total)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb_total, K, L)

        side_info = torch.cat(
            [side_info, cond_mask.unsqueeze(1)], dim=1
        )  # (B, emb_total + 1, K, L)

        return side_info

    def calc_loss_valid(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        side_info,
        is_train,
        X_itp,
        X_DWT,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                X_Tilde,
                cond_mask,
                X_Tilde_mask,
                indicating_mask,
                side_info,
                is_train=is_train,
                set_t=t,
                X_itp=X_itp,
                X_DWT=X_DWT,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        side_info,
        is_train=1,
        set_t=-1,
        X_itp=None,
        X_DWT=None,
    ):
        B, K, L = X_Tilde.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        # add noise to observed data
        current_alpha = self.alpha_torch[t]  # (B, 1, 1)
        noise = torch.randn_like(X_Tilde)
        noisy_data = (current_alpha**0.5) * X_Tilde + (
            1.0 - current_alpha
        ) ** 0.5 * noise  # (B, K, L)

        # get the input to the diffusion model
        (
            total_input, 
            all_observed_input,
        ) = self.set_input_to_diffmodel(
            X_original=X_Tilde,
            noisy_data=noisy_data,
            cond_mask=cond_mask,
        )

        # get the output of diffusion model
        (
            forward_pred_noise,
        ) = self.diffmodel(
            total_input,
            side_info,
            t,
            X_itp,
            X_DWT,
        )  # if in validation mode, pred_total_input and pred_side_info are None

        target_mask = X_Tilde_mask - cond_mask
        residual = (noise - forward_pred_noise) * target_mask
        num_eval = target_mask.sum()
        loss_noise = (residual**2).sum() / (num_eval if num_eval > 0 else 1)

        torch.cuda.empty_cache()

        return loss_noise

    def set_input_to_diffmodel(
        self,
        X_original,
        noisy_data,
        cond_mask,
    ):
        def get_noisy_total_input(X_original, X_noisy, mask) -> torch.Tensor:
            return (mask * X_original).unsqueeze(1) + ((1 - mask) * X_noisy).unsqueeze(
                1
            )

        total_input = get_noisy_total_input(X_original, noisy_data, cond_mask)
        return (
            total_input,
            X_original.unsqueeze(1),
        )

    def impute(self, X_Tilde, cond_mask, side_info, n_samples, X_itp, X_DWT):
        B, K, L = X_Tilde.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            current_sample = torch.randn_like(X_Tilde)
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * X_Tilde).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = cond_obs + noisy_target

                predicted = self.diffmodel.impute(
                    diff_input, side_info, torch.tensor([t]).to(self.device), X_itp, X_DWT
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    
    def impute_DDIM(self, X_Tilde, cond_mask, side_info, n_samples, X_itp, X_DWT):
        B, K, L = X_Tilde.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        total_steps = self.num_steps
        ddim_steps = torch.linspace(0, total_steps - 1, self.K_fast).long().to(self.device)
        ddim_steps = ddim_steps.flip(0)  # T -> 0

        for i in range(n_samples):

            current_sample = torch.randn_like(X_Tilde)

            for idx in range(len(ddim_steps) - 1):
                t = ddim_steps[idx]
                t_prev = ddim_steps[idx + 1]

                cond_obs = (cond_mask * X_Tilde).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = cond_obs + noisy_target

                predicted = self.diffmodel.impute(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(self.device),
                    X_itp,
                    X_DWT
                )

                sigma_ddim = ((1 - alpha_hat_prev) / (1 - alpha_hat_t)).sqrt()
                sigma = self.eta * sigma_ddim

                alpha_hat_t = self.alpha_hat[t].view(1, 1, 1)
                x0_pred = (current_sample - (1 - alpha_hat_t).sqrt() * predicted) / alpha_hat_t.sqrt()
                alpha_hat_prev = self.alpha_hat[t_prev].view(1, 1, 1)
                pred_dir = (1 - alpha_hat_prev - sigma ** 2).sqrt() * predicted
                x_prev_mean = alpha_hat_prev.sqrt() * x0_pred + pred_dir

                if t_prev > 0:
                    noise = torch.randn_like(current_sample)
                    current_sample = x_prev_mean + sigma * noise
                else:
                    current_sample = x_prev_mean

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        if is_train:
            (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                X_itp,
                X_DWT,
            ) = self.process_data(batch, is_train)
        else:
            (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask, X_itp, X_DWT) = (
                self.process_data(batch, is_train)
            )

        cond_mask = X_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            X_Tilde=X_Tilde,
            cond_mask=cond_mask,
            X_Tilde_mask=X_Tilde_mask,
            indicating_mask=indicating_mask,
            side_info=side_info,
            is_train=is_train,
            X_itp=X_itp,
            X_DWT=X_DWT,
        )

    def evaluate(self, batch, n_samples):
        (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask, X_itp, X_DWT) = (
            self.process_data(batch, istrain=0)
        )

        with torch.no_grad():
            cond_mask = X_mask
            eval_mask = X_Tilde_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            if self.is_fast_sampling:
                samples = self.impute_DDIM(X_Tilde, cond_mask, side_info, n_samples, X_itp, X_DWT)
            else:
                samples = self.impute(X_Tilde, cond_mask, side_info, n_samples, X_itp, X_DWT)

        return samples, X_Tilde, eval_mask, X_Tilde_mask, observed_tp


class MFDiff(MFDiff_base):

    def __init__(self, config, device, target_dim=36, seq_len=24, adj_mx=None):
        super(MFDiff, self).__init__(target_dim, config, device, adj_mx)
        self.seq_len = seq_len

    def process_data(self, batch, istrain=1):
        if istrain == 1:
            (
                X_tensor,
                mask_tensor,
                indicating_mask_tensor,
                X_Tilde_tensor,
                pred_tensor, 
                pred_mask_tensor,
                X_itp_tensor,
                DWT_tensor,
            ) = batch
        else:
            X_tensor, mask_tensor, X_Tilde_tensor, indicating_mask_tensor, X_itp_tensor, DWT_tensor = batch

        X_Tilde = X_Tilde_tensor.to(self.device).float()  # B,L,F
        X_mask = mask_tensor.to(
            self.device
        ).float() 
        indicating_mask = indicating_mask_tensor.to(
            self.device
        ).float()  # indicating mask
        X_Tilde_mask = X_mask + indicating_mask

        batch_size = X_Tilde.shape[0]
        
        observed_tp = (
            torch.from_numpy(
                np.tile(
                    np.arange(self.seq_len), batch_size
                ).reshape(  # [0, 1, 2, 3, seq_len - 1] * batch_size
                    batch_size, self.seq_len
                )
            )
            .to(self.device)
            .float()
        )

        X_Tilde = X_Tilde.permute(0, 2, 1)  # B,F,L
        X_Tilde_mask = X_Tilde_mask.permute(0, 2, 1)
        X_mask = X_mask.permute(0, 2, 1)
        indicating_mask = indicating_mask.permute(0, 2, 1)
        
        X_itp = X_itp_tensor.to(self.device).float()
        X_itp = X_itp.permute(0, 2, 1)

        X_DWT = DWT_tensor.to(self.device).float()  # B,L,N,level+1
        X_DWT = X_DWT.permute(0, 2, 1, 3)  # B,N,L,level+1

        if istrain == 0:
            return (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask, X_itp, X_DWT)

        else:
            return (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                X_itp,
                X_DWT,
            )
