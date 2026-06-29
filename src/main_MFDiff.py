import argparse
import json
import yaml
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb


sys.path.append("../dataloader")
sys.path.append("../models/MFDiff")
sys.path.append("../utils")
from model import MFDiff
from utils import *
import signal
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import math
import pickle

def get_similarity_CELP(dataset):
    adj_file = f"../datasets/{dataset}/{dataset}_adj.npy"
    adj = np.load(adj_file)
    return adj

def get_similarity_PEMS(args, thr=0.1, force_symmetric=False, sparse=False):

    # build 2-direction graph based on ASTGNN
    distance_df_filename = f"../datasets/{args.dataset}/{args.dataset}.csv"
    num_of_vertices = args.feature

    id_file_path = f"../datasets/{args.dataset}/{args.dataset}.txt"
    id_filename = id_file_path if os.path.exists(id_file_path) else None

    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    
    else:
        import csv
        A = np.eye(int(num_of_vertices), dtype=np.float32)
        distaneA = np.full((int(num_of_vertices), int(num_of_vertices)), -np.inf, dtype=np.float32)
        np.fill_diagonal(distaneA, 0.0)

        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline() 
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance

        else: 
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance

        finite_dist = distaneA.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        sigma = finite_dist.std()
        adj = np.exp(-np.square(distaneA / sigma))
        # adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)

        return A, distaneA, adj
def train(
    model,
    config,
    args,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=1,
    foldername="",
    current_time=None,
):

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/MFDiff_{}.pth".format(current_time)

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10

    torch.save(model.state_dict(), output_path)

    for epoch_no in tqdm(range(config["epochs"])):
        avg_loss = 0.0
        model.train()

        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss_noise = model(train_batch)

            loss = loss_noise  # compute total loss
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        lr_scheduler.step()
        train_loss = avg_loss / batch_no

        # if train_loss == nan, restart training
        if math.isnan(train_loss):
            print(f"Warning: NaN detected in train_loss. Restarting training...")
            restart_flag = True
            return restart_flag

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and epoch_no >=150:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                valid_loss = avg_loss_valid / batch_no
                print(
                    "Epoch {}: train loss = {}  valid loss = {}".format(
                        epoch_no + 1,
                        train_loss,
                        valid_loss,
                    )
                )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                torch.save(model.state_dict(), output_path)

        else:
            print(
                "Epoch {}: train loss = {}".format(
                    epoch_no + 1, train_loss
                )
            )
        
        # epoch_clip
        if epoch_no == args.epoch_clip:
            restart_flag = False
            return restart_flag

    # end training
    restart_flag = False
    return restart_flag


def evaluate(
    model,
    test_loader,
    nsample,
    scaler,
    mean_scaler,
    save_result_path,
    current_time=None,
):

    with torch.no_grad():
        model.eval()

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        imputed_data = []
        groundtruth = []
        eval_mask = []
        results = {}
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = (
                    output  # imputed results
                )
                samples = samples.permute(0, 1, 3, 2)  # samples(B,nsample,N,L)-->samples(B,nsample,L,N)
                c_target = c_target.permute(0, 2, 1)  # (B,L,N)
                eval_points = eval_points.permute(0, 2, 1)  # (B,L,N)
                observed_points = observed_points.permute(0, 2, 1)  # (B,L,N)

                samples_median = samples.median(
                    dim=1
                )  # use median as prediction to calculate the RMSE and MAE, include the median values and the indices

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                output = samples_median.values * scaler + mean_scaler
                X_Tilde = c_target * scaler + mean_scaler
                eval_M = eval_points
                imputed_data.append(output.cpu().numpy())
                groundtruth.append(X_Tilde.cpu().numpy())
                eval_mask.append(eval_M.cpu().numpy())

                if args.evaluate_clip:
                    break

            results["imputed_data"] = np.concatenate(imputed_data, axis=0)
            results["groundtruth"] = np.concatenate(groundtruth, axis=0)
            results["eval_mask"] = np.concatenate(eval_mask, axis=0)

            mae, rmse, mape, mse, r2 = missed_eval_np(
                results["imputed_data"],
                results["groundtruth"],
                1 - results["eval_mask"],
            )

            all_target = torch.cat(all_target, dim=0)  # (B,L,K)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)  # (B,L,K)
            all_observed_point = torch.cat(all_observed_point, dim=0)  # (B,L,K)
            all_observed_time = torch.cat(all_observed_time, dim=0)  # (B,L)
            all_generated_samples = torch.cat(
                all_generated_samples, dim=0
            )  # (B,nsample,L,K)

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            print(
                "mae = {:.3f}, rmse = {:.3f}, mape = {:.3f}%, mse = {:.3f}, r2 = {:.3f}, CRPS = {:.4f}".format(
                    mae, rmse, mape * 100, mse, r2, CRPS
                )
            )
            np.save(save_result_path + "/MFDiff_{}.npy".format(current_time), results)


def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)

    seed_torch(args.seed)
    path = "../config/{}_{}.yaml".format(args.dataset, args.missing_pattern)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print(json.dumps(config, indent=4))

    # load args
    dataset = args.dataset
    dataset_path = args.dataset_path
    seq_len = args.seq_len
    miss_rate = args.missing_ratio
    missing_pattern = args.missing_pattern
    batch_size = config["train"]["batch_size"]

    saving_path = args.saving_path + "/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    save_result_path = args.save_result_path + "/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)

    # load data
    train_loader = generate_train_dataloader(
        dataset,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
    )
    val_loader = generate_val_test_dataloader(
        dataset,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )

    # get explicit_adj_mx
    if args.dataset in ['PEMS04']:
        _, _, adj_mx = get_similarity_PEMS(args)
    elif args.dataset in ['CELP-WA', 'CELP-WI']:
        adj_mx = get_similarity_CELP(dataset=args.dataset)
    else:
        raise ValueError("cannot get Adj")
    adj_mx = torch.tensor(adj_mx).to(args.device)

    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
    with open(f"../datasets/{dataset}/{dataset}_meanstd.pk", "rb") as fb:
        mean, std = pk.load(fb)
    mean = torch.from_numpy(mean).to(args.device)
    std = torch.from_numpy(std).to(args.device)

    config["model"]["timeemb"] = args.time_emb_dim
    config["model"]["nodeemb"] = args.node_emb_dim
    config["diffusion"]["dataset"] = args.dataset
    config["diffusion"]["DWT_level"] = args.DWT_level
    config["model"]["is_fast_sampling"] = args.is_fast_sampling
    config["model"]["eta"] = args.eta
    config["model"]["K_fast"] = args.K_fast

    # training
    if args.scratch:
        restart_training = True
        while restart_training == True:
            model = MFDiff(
                config, args.device, target_dim=args.feature, seq_len=args.seq_len, adj_mx=adj_mx
            ).to(args.device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f'Total parameters: {total_params}')

            Flag = train(
                model,
                config["train"],
                args,
                train_loader,
                valid_loader=val_loader,
                foldername=saving_path,
                current_time=current_time,
            )
            restart_training = Flag

        print("load model from", saving_path)
        model.load_state_dict(
            torch.load(saving_path + "/MFDiff_{}.pth".format(current_time))
        )
    else:   # inference using pretrained model
        model = MFDiff(
            config, args.device, target_dim=args.feature, seq_len=args.seq_len, adj_mx=adj_mx
        ).to(args.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params}')
        print("load model from", args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))

    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=std,
        mean_scaler=mean,
        save_result_path=save_result_path,
        current_time=current_time
    )


if __name__ == "__main__":

    pid = os.getpid()

    parser = argparse.ArgumentParser(description="MFDiff")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="PEMS04", type=str, help="dataset name")
    parser.add_argument("--dataset_path", type=str, default="../datasets/PEMS04/",)
    parser.add_argument("--save_result_path", type=str, default="../results/", help="the save path of imputed data",)
    parser.add_argument("--saving_path", type=str, default="../saved_models", help="saving model pth")
    parser.add_argument("--checkpoint_path",type=str,default="",)
    parser.add_argument("--seq_len", type=int, default=24, help="sequence length")
    parser.add_argument("--feature", help="feature nums", type=int, default=None)
    parser.add_argument("--missing_pattern",type=str,default="block",help="missing pattern on training set",)
    parser.add_argument("--missing_ratio", type=float, default=0.02, help="missing ratio on training set")
    parser.add_argument("--scratch", default=True, help="test or scratch")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)

    #
    parser.add_argument("--time_emb_dim", type=int, default=96)
    parser.add_argument("--node_emb_dim", type=int, default=16)
    parser.add_argument("--epoch_clip", type=int, default=1000)
    parser.add_argument("--evaluate_clip", default=False)

    parser.add_argument("--DWT_level", type=int, default=3)
    parser.add_argument("--is_fast_sampling", type=bool, default=True)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--K_fast", type=int, default=5)

    args = parser.parse_args()

    # 动态选择 DataLoader 模块
    module_name = f"dataloaderITP_{args.DWT_level}LevelDWT"
    exec(f"from {module_name} import *")

    # add feature
    dataset_to_feature = {
        "PEMS04": 307,
        "CELP-WA": 377,
        "CELP-WI": 247
    }
    args.feature = dataset_to_feature[args.dataset]

    print(args)
    import setproctitle
    setproctitle.setproctitle(f'{parser.description}_{args.dataset}_{args.missing_pattern}')

    start_time = time.time()
    main(args)
    print("Spend Time: ", time.time() - start_time)

    os.kill(pid, signal.SIGKILL)


