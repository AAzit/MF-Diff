import numpy as np
import pickle as pk

import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import sample_mask

import pandas as pd
import torchcde
import os

import pywt
def disentangle(x, w, j=2):
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0], np.zeros_like(coef[1]), np.zeros_like(coef[2])]
    coefh1 = [np.zeros_like(coef[0]), coef[1], np.zeros_like(coef[2])]
    coefh2 = [np.zeros_like(coef[0]), np.zeros_like(coef[1]), coef[2]]

    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh1 = pywt.waverec(coefh1, w).transpose(0,3,2,1)
    xh2 = pywt.waverec(coefh2, w).transpose(0,3,2,1)
    # x = x.transpose(0,3,2,1)

    return xl, xh1, xh2


def generate_val_test_dataloader(
    dataset,
    seq_len,
    missing_ratio,
    missing_pattern="point",
    batch_size=4,
    mode="val",
    num_workers=0,
    DWT_level=2
):


    with open(f"../datasets/{dataset}/{dataset}_meanstd.pk", "rb") as f:
        train_mean, train_std = pk.load(f)  # (N),(N)

    if dataset in ['PEMS04']:
        df = np.load(f"../datasets/{dataset}/{dataset}.npz")['data'][:, :, 0]
        ob_mask = (df != 0.).astype('uint8')
        c_data = ((df - train_mean) / train_std) * ob_mask
        c_data = np.where(ob_mask, c_data, np.nan)
    elif dataset in ['CELP-WA', 'CELP-WI']:
        df = pd.read_csv(f"../datasets/{dataset}/{dataset}.csv")
        df = np.array(df.iloc[:, 1:].values)
        ob_mask = (df != 0.).astype('uint8')
        c_data = ((df - train_mean) / train_std) * ob_mask
        c_data = np.where(ob_mask, c_data, np.nan)

    # 0.7:0.1:0.2
    if mode == "val":
        val_start = int(len(c_data) * 0.7)
        val_end = int(len(c_data) * 0.8)
        data = c_data[val_start:val_end, :]
        print("val data shape: ", data.shape)
        val_SEED = 9101111
        rng = np.random.default_rng(val_SEED)
    elif mode == "test":
        test_start = int(len(c_data) * 0.8)
        test_end = int(len(c_data))
        data = c_data[test_start:test_end, :]
        print("test data shape: ", data.shape)
        test_SEED = 9101110
        rng = np.random.default_rng(test_SEED)
    else:
        assert False, "mode must be val or test"

    X_Tilde = data
    gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)
    if missing_pattern == "block":
        # block missing
        indicating_mask = sample_mask(
            shape=data.shape,
            p=missing_ratio,
            p_noise=0.0,
            max_seq=24,
            min_seq=12,
            rng=rng,
        )
    else:
        # point missing
        indicating_mask = sample_mask(
            shape=data.shape,
            p=0.0,
            p_noise=missing_ratio,
            max_seq=24,
            min_seq=12,
            rng=rng,
        )
    X = X_Tilde * (1 - indicating_mask)

    mask = gt_mask * (1 - indicating_mask)

    print(
        mode
        + ": original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            1 - np.sum(gt_mask) / gt_mask.size,
            np.sum(indicating_mask) / indicating_mask.size,
            missing_pattern,
            1 - np.sum(mask) / mask.size,
        )
    )
    X = np.nan_to_num(X)
    X_Tilde = np.nan_to_num(X_Tilde)

    print("generating itp data...")
    tmp_data = torch.tensor(X_Tilde).to(torch.float64)  # (L,N)
    tmp_mask = torch.tensor(mask).to(torch.float64)  # (L,N)
    itp_data = torch.where(tmp_mask == 0, float('nan'), tmp_data).to(torch.float32)       # cde_input (N,L,1)
    itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
    itp_data = np.array(itp_data)

    sample_nums = data.shape[0] // seq_len
    print(mode + " samples: {}".format(sample_nums))
    input_X_list, input_mask_list, eval_mask, output_gt_list, X_itp_list = [], [], [], [], []
    for i in range(sample_nums):
        input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
        input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
        eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
        output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
        X_itp_list.append(itp_data[i * seq_len : (i + 1) * seq_len])

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    X_itp_tensor = torch.from_numpy(np.array(X_itp_list)).float()

    save_path = f"../datasets/{dataset}/2LevelDWT/{missing_pattern}{missing_ratio}/{mode}_tensor_list.pt"
    if os.path.exists(save_path):
        X_L_tensor, X_H1_tensor, X_H2_tensor = torch.load(save_path)
    else:
        X_L_list, X_H1_list, X_H2_list  = [], [], []
        for i in range(sample_nums):
            X_itp = itp_data[i * seq_len : (i + 1) * seq_len] * train_std + train_mean  # L, N
            denorm_X = np.expand_dims(np.expand_dims(X_itp,axis=0), axis=-1)   # B, L, N, 1  (B=1)
            denorm_XL, denorm_XH1, denorm_XH2  = disentangle(denorm_X, "db1", j=2)  # (B, L, N, 1)
            XL = ((denorm_XL.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
            XH1 = ((denorm_XH1.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
            XH2 = ((denorm_XH2.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
            X_L_list.append(XL) # L,N
            X_H1_list.append(XH1)
            X_H2_list.append(XH2)
        X_L_tensor = torch.from_numpy(np.array(X_L_list)).float()
        X_H1_tensor = torch.from_numpy(np.array(X_H1_list)).float()
        X_H2_tensor = torch.from_numpy(np.array(X_H2_list)).float()
        tensor_list = [X_L_tensor, X_H1_tensor, X_H2_tensor]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tensor_list, save_path)
    DWT_tensor = torch.stack([X_L_tensor, X_H1_tensor, X_H2_tensor], dim=-1)
    
    tensor_dataset = TensorDataset(
        X_tensor, mask_tensor, X_Tilde_tensor, eval_mask_tensor, X_itp_tensor, DWT_tensor
    )
    dataloader = DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


def generate_train_dataloader(
    dataset, seq_len, missing_ratio, missing_pattern, batch_size=4, mode="train"
):
    
    # df = pd.read_csv(f"../datasets/{dataset}/{dataset}.csv")
    # df = np.array(df.iloc[:, 1:].values)

    # [mean, std] = np.mean(df[:int(df.shape[0]*0.7), :], axis=0), np.std(df[:int(df.shape[0]*0.7), :], axis=0)
    # with open(f"../datasets/{dataset}/{dataset}_meanstd.pk", 'wb') as file:
    #     pk.dump([mean, std], file)

    with open(f"../datasets/{dataset}/{dataset}_meanstd.pk", "rb") as f:
        train_mean, train_std = pk.load(f)  # (N),(N)

    S = None 
    if dataset in ['PEMS04']:
        df = np.load(f"../datasets/{dataset}/{dataset}.npz")['data'][:, :, 0]
        ob_mask = (df != 0.).astype('uint8')
        c_data = ((df - train_mean) / train_std) * ob_mask
        c_data = np.where(ob_mask, c_data, np.nan)
    elif dataset in ['CELP-WA', 'CELP-WI']:
        df = pd.read_csv(f"../datasets/{dataset}/{dataset}.csv")
        df = np.array(df.iloc[:, 1:].values)
        ob_mask = (df != 0.).astype('uint8')
        c_data = ((df - train_mean) / train_std) * ob_mask
        c_data = np.where(ob_mask, c_data, np.nan)
        S = 12


    # 0.7:0.1:0.2
    train_start = 0
    train_end = int(len(c_data) * 0.7)
    train_data = c_data[train_start:train_end, :]
    print("train data shape: ", train_data.shape)

    train_SEED = 9101112
    train_rng = np.random.default_rng(train_SEED)

    X_Tilde = train_data  # raw_data (L,N)
    gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)  

    if missing_pattern == "block":
        # block missing
        indicating_mask = sample_mask(
            shape=train_data.shape,
            p=missing_ratio,
            p_noise=0.0,
            max_seq=24,
            min_seq=12,
            rng=train_rng,
        )
    else:
        # point missing
        indicating_mask = sample_mask(
            shape=train_data.shape,
            p=0.0,
            p_noise=missing_ratio,
            max_seq=24,
            min_seq=12,
            rng=train_rng,
        )

    X = X_Tilde * (1 - indicating_mask)

    mask = gt_mask * (1 - indicating_mask)
    print(
        "Train: original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            1 - np.sum(gt_mask) / gt_mask.size,
            np.sum(indicating_mask) / indicating_mask.size,
            missing_pattern,
            1 - np.sum(mask) / mask.size,
        )
    )

    X = np.nan_to_num(X)
    X_Tilde = np.nan_to_num(X_Tilde)

    print("generating itp data...")

    tmp_data = torch.tensor(X_Tilde).to(torch.float64)  # (L,N)
    tmp_mask = torch.tensor(mask).to(torch.float64)  # (L,N)
    itp_data = torch.where(tmp_mask == 0, float('nan'), tmp_data).to(torch.float32)     # cde_input (N,L,1)
    itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
    itp_data = np.array(itp_data)

    if S is None:
        train_nums = train_data.shape[0] // seq_len - 1
        print("train samples: {}".format(train_nums))
        (
            input_X_list,
            input_mask_list,
            eval_mask,
            output_gt_list,
            pred_gt_list,
            pred_gt_mask,
            X_itp_list,
        ) = ([], [], [], [], [], [], [])
        for i in range(train_nums):
            input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
            input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
            eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
            output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
            pred_gt_list.append(X_Tilde[(i + 1) * seq_len : (i + 2) * seq_len])
            pred_gt_mask.append(gt_mask[(i + 1) * seq_len : (i + 2) * seq_len])
            X_itp_list.append(itp_data[i * seq_len : (i + 1) * seq_len])

    else:
        train_nums = ((train_data.shape[0] - seq_len) // S + 1) - 1
        print("train samples: {}".format(train_nums))
        (
            input_X_list,
            input_mask_list,
            eval_mask,
            output_gt_list,
            pred_gt_list,
            pred_gt_mask,
            X_itp_list,
        ) = ([], [], [], [], [], [], [])
        for i in range(train_nums):
            start_idx = i * S
            end_idx = start_idx + seq_len
            pred_start_idx = (i+1) * S
            pred_end_idx = pred_start_idx + seq_len

            input_X_list.append(X[start_idx : end_idx])
            input_mask_list.append(mask[start_idx : end_idx])
            eval_mask.append(indicating_mask[start_idx : end_idx])
            output_gt_list.append(X_Tilde[start_idx : end_idx])
            pred_gt_list.append(X_Tilde[pred_start_idx : pred_end_idx])
            pred_gt_mask.append(gt_mask[pred_start_idx : pred_end_idx])
            X_itp_list.append(itp_data[start_idx : end_idx])

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    pred_gt_tensor = torch.from_numpy(np.array(pred_gt_list)).float()
    pred_gt_mask_tensor = torch.from_numpy(np.array(pred_gt_mask)).float()
    X_itp_tensor = torch.from_numpy(np.array(X_itp_list)).float()

    save_path = f"../datasets/{dataset}/2LevelDWT/{missing_pattern}{missing_ratio}/{mode}_tensor_list.pt"
    if os.path.exists(save_path):
        X_L_tensor, X_H1_tensor, X_H2_tensor = torch.load(save_path)
    else:
        X_L_list, X_H1_list, X_H2_list  = [], [], []
        if S is None:
            train_nums = train_data.shape[0] // seq_len - 1
            for i in range(train_nums):
                X_itp = itp_data[i * seq_len : (i + 1) * seq_len] * train_std + train_mean  # L, N
                denorm_X = np.expand_dims(np.expand_dims(X_itp,axis=0), axis=-1)   # B, L, N, 1  (B=1)
                denorm_XL, denorm_XH1, denorm_XH2  = disentangle(denorm_X, "db1", j=2)  # (B, L, N, 1)
                XL = ((denorm_XL.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                XH1 = ((denorm_XH1.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                XH2 = ((denorm_XH2.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                X_L_list.append(XL)
                X_H1_list.append(XH1)
                X_H2_list.append(XH2)
        else:
            train_nums = ((train_data.shape[0] - seq_len) // S + 1) - 1
            for i in range(train_nums):
                start_idx = i * S
                end_idx = start_idx + seq_len
                X_itp = itp_data[start_idx : end_idx] * train_std + train_mean  # L, N
                denorm_X = np.expand_dims(np.expand_dims(X_itp,axis=0), axis=-1)   # B, L, N, 1  (B=1)
                denorm_XL, denorm_XH1, denorm_XH2  = disentangle(denorm_X, "db1", j=2)  # (B, L, N, 1)
                XL = ((denorm_XL.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                XH1 = ((denorm_XH1.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                XH2 = ((denorm_XH2.squeeze(-1).squeeze(0) - train_mean) / (train_std + 1e-8))
                X_L_list.append(XL)
                X_H1_list.append(XH1)
                X_H2_list.append(XH2)
        X_L_tensor = torch.from_numpy(np.array(X_L_list)).float()
        X_H1_tensor = torch.from_numpy(np.array(X_H1_list)).float()
        X_H2_tensor = torch.from_numpy(np.array(X_H2_list)).float()
        tensor_list = [X_L_tensor, X_H1_tensor, X_H2_tensor]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tensor_list, save_path)
    DWT_tensor = torch.stack([X_L_tensor, X_H1_tensor, X_H2_tensor], dim=-1)

    train_dataset = TensorDataset(
        X_tensor,
        mask_tensor,
        eval_mask_tensor,
        X_Tilde_tensor,
        pred_gt_tensor,
        pred_gt_mask_tensor,
        X_itp_tensor,
        DWT_tensor
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


if __name__ == "__main__":
    dataset_path = "../datasets/PEMS04/"
    seq_len = 24
    miss_rate = 0.2
    batch_size = 4
    missing_pattern = "block"
    train_loader = generate_train_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
    )
    val_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
