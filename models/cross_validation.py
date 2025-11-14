import os
import math
import time
import copy
import numpy as np
from typing import Dict, Callable, List
from torch.utils.data import Subset
from preprocessing.scaler_imputer import Standardize_ZeroImpute, Scaled_Imputed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from types import SimpleNamespace

def to_namespace(obj):
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

def move_batch_to_device(batch, device):
    
    batch['target']   = batch['target'].to(device)
    batch['T (K)']    = batch['T (K)'].to(device)            

    # Move solutes
    solutes = []
    for solute in batch['solute']:
        solute_ns = to_namespace(solute)
        for k, v in vars(solute_ns).items():
            if isinstance(v, torch.Tensor):
                setattr(solute_ns, k, v.to(device))
        solutes.append(solute_ns)
    batch['solute'] = solutes

    # Move solvents
    solvents_out = []
    for solvent in batch['solvent']:
        solvent_ns = to_namespace(solvent)
        for k, v in vars(solvent_ns).items():
            if isinstance(v, torch.Tensor):
                setattr(solvent_ns, k, v.to(device))
        solvents_out.append(solvent_ns)
    batch['solvent'] = solvents_out
    return batch
# ---------------- Metrics ----------------
def mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(y_pred, y_true)

@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    if y_true.dim() == 2 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(1)
    if y_pred.dim() == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)
    diff = y_pred - y_true
    mse = (diff ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = diff.abs().mean().item()
    return {"loss": mse, "rmse": rmse, "mae": mae}

# ---------------- Early Stopping ----------------
class EarlyStopping:
    def __init__(self, patience=20, delta=1e-7):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.count = 0
        self.best_state = None

    def step(self, value, model):
        if value < self.best - self.delta:
            self.best = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.count = 0
            return True  
        else:
            self.count += 1
            return False 

    @property
    def should_stop(self):
        return self.count >= self.patience

# ---------------- Epoch runners ----------------
def run_one_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    total_n = 0
    all_y, all_yhat = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        target = batch["target"] 

        if is_train:
            optimizer.zero_grad()

        output = model(batch)    
        loss = mse_loss(target, output)

        if is_train:
            loss.backward()
            optimizer.step()

        B = target.size(0)
        total_loss_sum += loss.item() * B
        total_n += B
        all_y.append(target.detach())
        all_yhat.append(output.detach())

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_yhat, dim=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss_sum / max(total_n, 1)
    return metrics

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss_sum = 0.0
    total_n = 0
    all_y, all_yhat = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        target = batch["target"]
        output = model(batch)
        loss = mse_loss(target, output)

        B = target.size(0)
        total_loss_sum += loss.item() * B
        total_n += B
        all_y.append(target.detach())
        all_yhat.append(output.detach())

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_yhat, dim=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss_sum / max(total_n, 1)
    return metrics, y_true, y_pred

def collate_fn(batch):
    batched_data = {
        'solv_name': [item['solv_name'] for item in batch],
        'solu_name': [item['solu_name'] for item in batch],
        'solute': [item['solute'] for item in batch],                 
        'solvent': [item['solvent'] for item in batch],     
        
        'T (K)': torch.stack([item['T (K)'] for item in batch]),      
        'target': torch.stack([item['target'] for item in batch])    
    }
    return batched_data

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)                # (B,1)
        preds.append(out.detach().cpu())
        trues.append(batch["target"].detach().cpu())
    y_pred = torch.cat(preds, dim=0).numpy().reshape(-1)
    y_true = torch.cat(trues, dim=0).numpy().reshape(-1)
    return y_true, y_pred

class _scaled_imputed_partial(torch.utils.data.Dataset):
    def __init__(self, base_subset, scaler):
        self.base = base_subset     
        self.scaler = scaler

        # cache train-mean for T (as float)
        m = getattr(self.scaler, "temp_stats", None)
        self.t_mean = float(m.mean.squeeze()) if m is not None else 0.0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # solute/solvent: use scaler transforms
        sol = item["solute"]["fp"]
        sov = item["solvent"]["fp"]
        sol_np = sol.detach().cpu().numpy()
        sov_np = sov.detach().cpu().numpy()

        sol_z = self.scaler.transform_solute(sol_np)  
        sov_z = self.scaler.transform_solvent(sov_np)  

        # temperature: keep raw K (NO scaling), only mean-impute if NaN/Inf
        T_val = float(item["T (K)"]) if isinstance(item["T (K)"], torch.Tensor) else float(item["T (K)"])
        if not np.isfinite(T_val):
            T_val = self.t_mean

        # write back as tensors
        item["solute"]["fp"]  = torch.tensor(sol_z, dtype=torch.float32)
        item["solvent"]["fp"] = torch.tensor(sov_z, dtype=torch.float32)
        item["T (K)"]         = torch.tensor(T_val, dtype=torch.float32)
        return item

# ---------------- Main CV trainer ----------------
def train_kfold_cv(
    train_dataset,
    test_dataset,
    model_builders: Dict[str, Callable[[torch.device], nn.Module]],
    output_root: str = "./runs_cv",
    batch_size: int = 64,
    lr: float = 3e-4,
    max_epochs: int = 150,
    patience: int = 20,
    num_workers: int = 0,
    seed: int = 42,
    n: int=5
):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    time_tag = time.strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(output_root, f"CV_{time_tag}")
    os.makedirs(root_dir, exist_ok=True)
    n_splits = n

    # KFold over train_dataset
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(train_dataset))

    # summary
    summary = {arch: [] for arch in model_builders.keys()}

    # test_loader 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    # arch -> list[n_models] of state_dict paths
    arch_fold_ckpts: Dict[str, List[str]] = {arch: [] for arch in model_builders.keys()}

    for arch_name, build_model in model_builders.items():
        print(f"\n=== Architecture: {arch_name} ===")
        arch_dir = os.path.join(root_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        for fold, (tr_idx, val_idx) in enumerate(kfold.split(indices), start=1):
            print(f"\n[Fold {fold}/{n_splits}]")

            ds_train = Subset(train_dataset, tr_idx.tolist())
            ds_val   = Subset(train_dataset, val_idx.tolist())

            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
            val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,collate_fn=collate_fn,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)

            model = build_model(device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            stopper = EarlyStopping(patience=patience, delta=1e-7)

            tb_dir = os.path.join(arch_dir, f"fold{fold}")
            writer = SummaryWriter(log_dir=tb_dir)

            best_epoch = 0
            for epoch in range(1, max_epochs + 1):
                train_m = run_one_epoch(model, train_loader, device, optimizer)
                val_m, _, _ = evaluate(model, val_loader, device)

                # TensorBoard (train/val)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_loss", train_m["loss"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_rmse", train_m["rmse"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_mae",  train_m["mae"],  epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_loss", val_m["loss"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_rmse", val_m["rmse"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_mae",  val_m["mae"],  epoch)

                print(f"Epoch {epoch:03d} | "
                      f"Train L {train_m['loss']:.6f} RMSE {train_m['rmse']:.6f} MAE {train_m['mae']:.6f} || "
                      f"Val L {val_m['loss']:.6f} RMSE {val_m['rmse']:.6f} MAE {val_m['mae']:.6f}")

                improved = stopper.step(val_m["loss"], model)
                if improved:
                    best_epoch = epoch
                if stopper.should_stop:
                    print(f"Early stopping at epoch {epoch} (no improve {patience})")
                    break

            # fold
            if stopper.best_state is not None:
                model.load_state_dict(stopper.best_state)

            final_val_m, _, _ = evaluate(model, val_loader, device)
            ckpt_path = os.path.join(arch_dir, f"fold{fold}_best.pth")
            torch.save({
                "model_state": model.state_dict(),
                "arch": arch_name,
                "fold": fold,
                "best_epoch": best_epoch,
                "val_metrics": final_val_m,
                "seed": seed,
            }, ckpt_path)
            print(f"Saved best model to: {ckpt_path}")
            arch_fold_ckpts[arch_name].append(ckpt_path)

            # fold-level hparams
            writer.add_hparams(
                hparam_dict={
                    "arch": arch_name,
                    "fold": fold,
                    "lr": lr,
                    "batch_size": batch_size,
                },
                metric_dict={
                    "val/loss": final_val_m["loss"],
                    "val/rmse": final_val_m["rmse"],
                    "val/mae":  final_val_m["mae"],
                }
            )

            summary[arch_name].append({
                "fold": fold,
                "val": final_val_m
            })

            writer.close()

        # --- ensemble test  ---
        print(f"\n[TEST] Ensemble predictions for arch '{arch_name}'")
        # state load , test, then avg. 
        all_fold_preds = []  
        with torch.no_grad():
            for ckpt in arch_fold_ckpts[arch_name]:
                # fresh model & load state
                model = build_model(device).to(device)
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state["model_state"])
                model.eval()

                fold_preds = []
                for batch in test_loader:
                    batch = move_batch_to_device(batch, device)
                    out = model(batch)  # (B,1)
                    fold_preds.append(out.cpu())
                fold_preds = torch.cat(fold_preds, dim=0)  
                all_fold_preds.append(fold_preds)

        ensemble_pred = torch.stack(all_fold_preds, dim=0).mean(dim=0) 
        ensemble_pred_std = torch.stack(all_fold_preds, dim=0).std(dim=0)  

        # test save
        _, y_test_true, _ = evaluate(model, test_loader, device)  
        y_true_np = y_test_true.cpu().numpy().reshape(-1)
        y_pred_np = ensemble_pred.cpu().numpy().reshape(-1)
        y_std     = ensemble_pred_std.cpu().numpy().reshape(-1)

        # CSV save
        csv_path = os.path.join(arch_dir, f"test_predictions_ensemble.csv")
        import pandas as pd
        pd.DataFrame({"y_true": y_true_np, "y_pred": y_pred_np, 'y_pred_std': y_std}).to_csv(csv_path, index=False)
        print(f"Saved test ensemble predictions CSV: {csv_path}")

    # -------- print summary --------
    print("\n========== SUMMARY ==========")
    for arch_name, rows in summary.items():
        vals = np.array([[r["val"]["loss"], r["val"]["rmse"], r["val"]["mae"]] for r in rows])
        mv = vals.mean(axis=0); sv = vals.std(axis=0)
        print(f"\n[{arch_name}] Val  : "
              f"Loss {mv[0]:.6f}±{sv[0]:.6f} | RMSE {mv[1]:.6f}±{sv[1]:.6f} | MAE {mv[2]:.6f}±{sv[2]:.6f}")

    print(f"\nTensorBoard log dirs under: {root_dir}")
    print("Launch with:")
    print(f"  tensorboard --logdir {root_dir}")

#%===========================================================
def _fit_scaler_on_subset(dataset, indices):
    sol_list, sov_list, T_list = [], [], []

    if isinstance(dataset, torch.utils.data.Subset):
        base_data = dataset.dataset
        base_indices = dataset.indices
    else:
        base_data = dataset
        base_indices = np.arange(len(dataset))

    if indices is None:
        indices = base_indices
    else:
        indices = [base_indices[i] for i in indices if i < len(base_indices)]

    for i in indices:
        it = base_data[i]
        sol_list.append(it["solute"]["fp"])
        sov_list.append(it["solvent"]["fp"])
        T_list.append(it["T (K)"])
    # to numpy stacks
    sol = torch.stack(sol_list).detach().cpu().numpy()
    sov = torch.stack(sov_list).detach().cpu().numpy()
    T   = torch.tensor(T_list).view(-1, 1).detach().cpu().numpy()

    scaler = Standardize_ZeroImpute()
    scaler.fit(solute_mat=sol, solvent_mat=sov, temp_vec=T)
    return scaler

def _wrap_scaled(dataset, scaler, indices):
    if indices is None:
        subset = dataset
    else:
        subset = Subset(dataset, indices)
    return Scaled_Imputed(subset, scaler)

def train_kfold_cv_fastprop(
    train_dataset,
    model_builders: Dict[str, Callable[[torch.device], nn.Module]],
    output_root: str = "./runs_cv_fastprop",
    batch_size: int = 64,
    lr: float = 3e-4,
    max_epochs: int = 150,
    patience: int = 20,
    num_workers: int = 0,
    seed: int = 42,
    n: int=5
):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    time_tag = time.strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(output_root, f"CV_{time_tag}")
    os.makedirs(root_dir, exist_ok=True)
    n_splits = n

    # KFold over train_dataset
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(train_dataset))

    # summary
    summary = {arch: [] for arch in model_builders.keys()}

    # arch -> list[n_models] of state_dict paths
    arch_fold_ckpts: Dict[str, List[str]] = {arch: [] for arch in model_builders.keys()}

    for arch_name, build_model in model_builders.items():
        print(f"\n=== Architecture: {arch_name} ===")
        arch_dir = os.path.join(root_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        for fold, (tr_idx, val_idx) in enumerate(kfold.split(indices), start=1):
            print(f"\n[Fold {fold}/{n_splits}]")
            scaler = _fit_scaler_on_subset(train_dataset, tr_idx)

            scaler_path = os.path.join(arch_dir, f"scaler{fold}.npz")
            scaler.save(scaler_path)

            is_film = ("film" in arch_name.lower())  # e.g., "fastprop+FiLM"
            if is_film:
                # scale solute/solvent only; temperature kept in K (NaN->train mean)
                ds_train = _scaled_imputed_partial(Subset(train_dataset, tr_idx), scaler)
                ds_val   = _scaled_imputed_partial(Subset(train_dataset, val_idx),   scaler)
            else:
                # baseline: standardize+zero-impute on all (incl. temperature)
                ds_train = _wrap_scaled(train_dataset, scaler, tr_idx)
                ds_val   = _wrap_scaled(train_dataset, scaler, val_idx)

            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
            val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,collate_fn=collate_fn,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)

            model = build_model(device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            stopper = EarlyStopping(patience=patience, delta=1e-7)

            tb_dir = os.path.join(arch_dir, f"fold{fold}")
            writer = SummaryWriter(log_dir=tb_dir)

            best_epoch = 0
            for epoch in range(1, max_epochs + 1):
                train_m = run_one_epoch(model, train_loader, device, optimizer)
                val_m, _, _ = evaluate(model, val_loader, device)

                # TensorBoard (train/val)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_loss", train_m["loss"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_rmse", train_m["rmse"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/train_mae",  train_m["mae"],  epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_loss", val_m["loss"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_rmse", val_m["rmse"], epoch)
                writer.add_scalar(f"{arch_name}/fold{fold}/val_mae",  val_m["mae"],  epoch)

                print(f"Epoch {epoch:03d} | "
                      f"Train L {train_m['loss']:.6f} RMSE {train_m['rmse']:.6f} MAE {train_m['mae']:.6f} || "
                      f"Val L {val_m['loss']:.6f} RMSE {val_m['rmse']:.6f} MAE {val_m['mae']:.6f}")

                improved = stopper.step(val_m["loss"], model)
                if improved:
                    best_epoch = epoch
                if stopper.should_stop:
                    print(f"Early stopping at epoch {epoch} (no improve {patience})")
                    break

            # fold
            if stopper.best_state is not None:
                model.load_state_dict(stopper.best_state)
            

            final_val_m, _, _ = evaluate(model, val_loader, device)
            ckpt_path = os.path.join(arch_dir, f"fold{fold}_best.pth")
            torch.save({
                "model_state": model.state_dict(),
                "arch": arch_name,
                "fold": fold,
                "best_epoch": best_epoch,
                "val_metrics": final_val_m,
                "seed": seed,
            }, ckpt_path)
            print(f"Saved best model to: {ckpt_path}")
            arch_fold_ckpts[arch_name].append(ckpt_path)

            # fold-level hparams
            writer.add_hparams(
                hparam_dict={
                    "arch": arch_name,
                    "fold": fold,
                    "lr": lr,
                    "batch_size": batch_size,
                },
                metric_dict={
                    "val/loss": final_val_m["loss"],
                    "val/rmse": final_val_m["rmse"],
                    "val/mae":  final_val_m["mae"],
                }
            )

            summary[arch_name].append({
                "fold": fold,
                "val": final_val_m
            })

            writer.close()

    # -------- print summary --------
    print("\n========== SUMMARY ==========")
    for arch_name, rows in summary.items():
        vals = np.array([[r["val"]["loss"], r["val"]["rmse"], r["val"]["mae"]] for r in rows])
        mv = vals.mean(axis=0); sv = vals.std(axis=0)
        print(f"\n[{arch_name}] Val  : "
              f"Loss {mv[0]:.6f}±{sv[0]:.6f} | RMSE {mv[1]:.6f}±{sv[1]:.6f} | MAE {mv[2]:.6f}±{sv[2]:.6f}")

    print(f"\nTensorBoard log dirs under: {root_dir}")
    print("Launch with:")
    print(f"  tensorboard --logdir {root_dir}")
