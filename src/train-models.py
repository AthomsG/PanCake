#!/usr/bin/env python3
# Train PyTorch LogReg (LBFGS) + MLP with cross-validation, unique run dir, and params JSON.
# - Saves per-fold coefficients and per-fold AUCs into run_dir/fold_csv/
# - Keeps prior defaults/behavior; adds --cv_folds and run metadata.
import os, argparse, math, warnings, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --------------------------- CLI ---------------------------
def get_args():
    ap = argparse.ArgumentParser(description="Train PyTorch LogReg (LBFGS) + MLP; evaluate & save")
    ap.add_argument("--input_dir", default="outputs/process-data",
                    help="Dir with genes_filtered.parquet & species_filtered.parquet")
    ap.add_argument("--out_dir", default="outputs/train-models",
                    help="Base directory to write outputs")
    ap.add_argument("--rank", default="s__", choices=["s__","g__","f__","o__","c__","p__"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay_mlp", type=float, default=1e-4)
    ap.add_argument("--l1_logreg", type=float, default=1e-5, help="L1 on LogReg weights")
    ap.add_argument("--logreg_weight_decay", type=float, default=0.0, help="L2 on LogReg weights (≈ 1/C)")
    ap.add_argument("--logreg_tol", type=float, default=1e-4, help="LBFGS tolerance (≈ sklearn tol)")
    ap.add_argument("--logreg_max_iter", type=int, default=100, help="LBFGS max iterations (≈ sklearn max_iter)")
    ap.add_argument("--n_null", type=int, default=10, help="Number of null-label permutations (train split only)")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds (>=2)")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction inside each fold's train split")
    return ap.parse_args()

# --------------------------- Utils ---------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def per_gene_auc(y_true, y_prob):
    aucs = np.full(y_true.shape[1], np.nan, dtype=float)
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if np.unique(yj).size < 2:
            continue
        aucs[j] = roc_auc_score(yj, y_prob[:, j])
    return aucs

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        loss = criterion(model(xb), yb)
        bs = xb.size(0); tot += loss.item() * bs; n += bs
    return tot / max(1, n)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval(); probs_list, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs_list.append(torch.sigmoid(model(xb)).cpu().numpy())
        ys.append(yb.cpu().numpy())
    return np.vstack(probs_list), np.vstack(ys)

# --------------------------- Models ---------------------------
class BaselineLogisticRegression(nn.Module):
    """Multi-label logistic regression (one linear layer)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # bias kept
    def forward(self, x):
        return self.linear(x)

class BetterMLP(nn.Module):
    """2-layer MLP with BatchNorm & Dropout"""
    # Keep the exact signature you currently have
    def __init__(self, in_dim, out_dim, h1=128, h2=256, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# --------------------------- LogReg: full-batch LBFGS ---------------------------
def train_logreg_lbfgs_fullbatch(
    in_dim, out_dim, Xtr_np, Ytr_np, Xval_np, Yval_np, device, *,
    pos_weight_vec, l1_lambda, l2_weight_decay, max_iter, tol
):
    class MultiLabelLogReg(nn.Module):
        def __init__(self, d, t):
            super().__init__()
            self.W = nn.Parameter(torch.zeros(d, t, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(t,   dtype=torch.float32))
        def forward(self, X):
            return X @ self.W + self.b

    model = MultiLabelLogReg(in_dim, out_dim).to(device)
    Xtr = torch.as_tensor(Xtr_np, device=device, dtype=torch.float32)
    Ytr = torch.as_tensor(Ytr_np, device=device, dtype=torch.float32)
    Xval = torch.as_tensor(Xval_np, device=device, dtype=torch.float32)
    Yval = torch.as_tensor(Yval_np, device=device, dtype=torch.float32)

    T = out_dim
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec, reduction="mean")

    loss_hist_tr, loss_hist_val = [], []

    opt = optim.LBFGS(
        [model.W, model.b],
        max_iter=max_iter,
        tolerance_grad=tol,
        line_search_fn="strong_wolfe",
        lr=1.0
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)
        data_mean = criterion(logits, Ytr)  # (1/(N*T)) sum BCE
        loss = T * data_mean                # scale to per-label mean (match reference)
        if l2_weight_decay > 0:
            loss = loss + 0.5 * l2_weight_decay * (model.W * model.W).sum()  # no bias L2
        if l1_lambda > 0:
            loss = loss + l1_lambda * model.W.abs().mean()                   # L1 on weights only
        loss.backward()

        with torch.no_grad():
            loss_hist_tr.append(float(loss.detach().cpu()))
            val_logits = model(Xval)
            val_data_mean = criterion(val_logits, Yval)
            val_loss = T * val_data_mean
            if l2_weight_decay > 0:
                val_loss = val_loss + 0.5 * l2_weight_decay * (model.W * model.W).sum()
            if l1_lambda > 0:
                val_loss = val_loss + l1_lambda * model.W.abs().mean()
            loss_hist_val.append(float(val_loss.detach().cpu()))
        return loss

    tqdm.write("Fitting LogReg with LBFGS (full-batch)...")
    model.train()
    opt.step(closure)
    return model, np.array(loss_hist_tr, dtype=float), np.array(loss_hist_val, dtype=float)

# --------------------------- MLP trainer (Adam + early stop) ---------------------------
def train_mlp(
    in_dim, out_dim, train_loader, val_loader, device, *,
    lr=1e-3, weight_decay=1e-4, max_epochs=400, patience=40, min_delta=1e-4,
    pos_weight=None, seed=0
):
    # Keep the call as before (your train_mlp used 256/128 explicitly)
    model = BetterMLP(in_dim, out_dim, h1=256, h2=128, p_drop=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    tr_curve, val_curve = [], []
    best_val, best_state, no_imp = math.inf, None, 0

    for _ in tqdm(range(1, max_epochs + 1), desc="MLP (real)", leave=True):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        tr_loss = eval_epoch(model, train_loader, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        tr_curve.append(tr_loss); val_curve.append(val_loss)

        sched.step(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, np.array(tr_curve, dtype=float), np.array(val_curve, dtype=float)

# --------------------------- Main ---------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Unique run directory
    run_name = time.strftime("run_%Y%m%d_%H%M%S") + f"_seed{args.seed}_k{args.cv_folds}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    fold_csv_dir = os.path.join(run_dir, "fold_csv")
    os.makedirs(fold_csv_dir, exist_ok=True)

    # Save params JSON
    params = {
        "rank": args.rank,
        "batch": args.batch,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay_mlp": args.weight_decay_mlp,
        "l1_logreg": args.l1_logreg,
        "logreg_weight_decay": args.logreg_weight_decay,
        "logreg_tol": args.logreg_tol,
        "logreg_max_iter": args.logreg_max_iter,
        "n_null": args.n_null,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "val_frac": args.val_frac,
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Load processed data
    genes_pq   = os.path.join(args.input_dir, "genes_filtered.parquet")
    species_pq = os.path.join(args.input_dir, "species_filtered.parquet")
    if not (os.path.isfile(genes_pq) and os.path.isfile(species_pq)):
        raise FileNotFoundError("Expected filtered parquet files under outputs/process-data. Run process-data first.")

    df_genes   = pd.read_parquet(genes_pq)
    df_species = pd.read_parquet(species_pq)

    # Align
    shared = df_genes.index.intersection(df_species.index)
    df_genes   = df_genes.loc[shared]
    df_species = df_species.loc[shared]

    # Binary matrices
    X = (df_species > 0).astype(np.float32).values
    Y = (df_genes   > 0).astype(np.float32).values
    species_names = df_species.columns.to_numpy()
    gene_names    = df_genes.columns.to_numpy()

    device = pick_device()
    print("Using device:", device)

    # --- Cross-validation setup ---
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be >= 2")
    rng = np.random.default_rng(args.seed)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    strat_labels = (Y.sum(axis=1) > 0).astype(int)

    # Storage to aggregate across folds (for final plots)
    all_auc_logreg, all_auc_mlp, all_auc_null_mean = [], [], []
    fold_summaries = []

    # ---- CV loop ----
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, strat_labels), start=1):
        print(f"\n=== Fold {fold_idx}/{args.cv_folds} ===")

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        Y_trainval, Y_test = Y[trainval_idx], Y[test_idx]

        # inner val split
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=args.val_frac, random_state=args.seed,
            stratify=(Y_trainval.sum(axis=1) > 0)
        )

        # tensors/loaders
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        X_tr_t, Y_tr_t = to_t(X_tr), to_t(Y_tr)
        X_val_t, Y_val_t = to_t(X_val), to_t(Y_val)
        X_te_t,  Y_te_t  = to_t(X_test), to_t(Y_test)

        train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=args.batch, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_te_t,  Y_te_t ), batch_size=args.batch, shuffle=False)

        # pos_weight per gene computed on training portion of this fold
        pos = Y_tr.sum(axis=0, dtype=np.float64)
        neg = Y_tr.shape[0] - pos
        with np.errstate(divide='ignore', invalid='ignore'):
            pw = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        pw = np.clip(pw, 1.0, 100.0)
        pos_weight_vec = torch.tensor(pw, dtype=torch.float32, device=device)

        in_dim, out_dim = X_tr.shape[1], Y_tr.shape[1]

        # ---- LogReg (LBFGS) ----
        logreg_lbfgs, logreg_tr_curve, logreg_val_curve = train_logreg_lbfgs_fullbatch(
            in_dim, out_dim, X_tr, Y_tr, X_val, Y_val, device,
            pos_weight_vec=pos_weight_vec,
            l1_lambda=args.l1_logreg,
            l2_weight_decay=args.logreg_weight_decay,
            max_iter=args.logreg_max_iter,
            tol=args.logreg_tol
        )

        # ---- MLP (Adam + early stop) ----
        mlp_real, mlp_tr_curve, mlp_val_curve = train_mlp(
            in_dim, out_dim, train_loader, val_loader, device,
            lr=args.lr, weight_decay=args.weight_decay_mlp,
            max_epochs=args.epochs, patience=args.patience, min_delta=1e-4,
            pos_weight=pos_weight_vec, seed=args.seed + fold_idx
        )

        # ---- Null permutations (MLP, train labels only) ----
        auc_null_list = []
        for i in range(args.n_null):
            y_tr_perm = Y_tr.copy()
            rng_perm = np.random.default_rng(args.seed + 1000*fold_idx + i + 1)
            idx = rng_perm.permutation(y_tr_perm.shape[0])
            train_loader_null = DataLoader(
                TensorDataset(X_tr_t, torch.as_tensor(y_tr_perm[idx], dtype=torch.float32)),
                batch_size=args.batch, shuffle=True
            )
            mlp_null_i, _, _ = train_mlp(
                in_dim, out_dim, train_loader_null, val_loader, device,
                lr=args.lr, weight_decay=args.weight_decay_mlp,
                max_epochs=args.epochs, patience=args.patience, min_delta=1e-4,
                pos_weight=pos_weight_vec, seed=args.seed + 1000*fold_idx + i + 1
            )
            proba_null_i, y_test_check = predict_probs(mlp_null_i, test_loader, device)
            auc_null_list.append(per_gene_auc(y_test_check, proba_null_i))
        auc_null_arr = np.vstack(auc_null_list) if len(auc_null_list) else np.empty((0, out_dim))
        auc_null_mean = np.nanmean(auc_null_arr, axis=0) if auc_null_arr.size else np.full(out_dim, np.nan)

        # ---- Evaluate on test split ----
        proba_logreg, y_test = predict_probs(logreg_lbfgs, test_loader, device)
        proba_mlp_real, _    = predict_probs(mlp_real,   test_loader, device)

        auc_logreg   = per_gene_auc(y_test, proba_logreg)
        auc_mlp_real = per_gene_auc(y_test, proba_mlp_real)

        all_auc_logreg.append(auc_logreg)
        all_auc_mlp.append(auc_mlp_real)
        all_auc_null_mean.append(auc_null_mean)

        # ---- Save per-fold coefficients (genes as rows, species as columns) ----
        if isinstance(logreg_lbfgs, BaselineLogisticRegression):
            W = logreg_lbfgs.linear.weight.detach().cpu().numpy().T
        else:
            W = logreg_lbfgs.W.detach().cpu().numpy()  # (in_dim, out_dim)
        coef_df = pd.DataFrame(W, index=species_names, columns=gene_names).T
        coef_path = os.path.join(fold_csv_dir, f"logreg_coefficients_fold{fold_idx}.csv")
        coef_df.to_csv(coef_path)

        # ---- Save per-fold AUCs ----
        auc_df = pd.DataFrame({
            "gene": gene_names,
            "auc_logreg": auc_logreg,
            "auc_mlp": auc_mlp_real,
            "auc_null_mean": auc_null_mean
        }).set_index("gene")
        auc_path = os.path.join(fold_csv_dir, f"per_gene_auc_fold{fold_idx}.csv")
        auc_df.to_csv(auc_path)

        # ---- Save training/validation curves per fold ----
        if logreg_tr_curve.size and logreg_val_curve.size:
            plt.figure(figsize=(5, 3.2), dpi=300)
            plt.plot(logreg_tr_curve, label="train", linewidth=1.5)
            plt.plot(logreg_val_curve, label="val", linewidth=1.5)
            plt.xlabel("LBFGS closures (line-search steps)")
            plt.ylabel("Loss")
            plt.title(f"LogReg (LBFGS) train/val loss — fold {fold_idx}")
            plt.legend(frameon=False); plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"logreg_train_val_curve_fold{fold_idx}.png"), bbox_inches="tight")
            plt.close()

        if mlp_tr_curve.size and mlp_val_curve.size:
            plt.figure(figsize=(5, 3.2), dpi=300)
            plt.plot(mlp_tr_curve, label="train", linewidth=1.5)
            plt.plot(mlp_val_curve, label="val", linewidth=1.5)
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.title(f"MLP train/val loss — fold {fold_idx}")
            plt.legend(frameon=False); plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"mlp_train_val_curve_fold{fold_idx}.png"), bbox_inches="tight")
            plt.close()

        # Fold summary
        fold_summaries.append({
            "fold": fold_idx,
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "mean_auc_logreg": float(np.nanmean(auc_logreg)),
            "mean_auc_mlp": float(np.nanmean(auc_mlp_real)),
            "mean_auc_null": float(np.nanmean(auc_null_mean)),
        })

    # ---- Aggregate over folds for a single AUC distribution plot ----
    auc_logreg_all   = np.concatenate(all_auc_logreg, axis=0)
    auc_mlp_all      = np.concatenate(all_auc_mlp, axis=0)
    auc_null_mean_all= np.concatenate(all_auc_null_mean, axis=0)

    plt.figure(figsize=(4, 3), dpi=300)
    bins = np.linspace(0.0, 1.0, 41)
    plt.hist(auc_mlp_all[~np.isnan(auc_mlp_all)], bins=bins, alpha=0.7, density=True,
             edgecolor="white", label="MLP")
    plt.hist(auc_null_mean_all[~np.isnan(auc_null_mean_all)], bins=bins, alpha=0.7, density=True,
             edgecolor="white", label="Null")
    plt.hist(auc_logreg_all[~np.isnan(auc_logreg_all)], bins=bins, alpha=0.55, density=True,
             edgecolor="white", label="LogReg")
    ax = plt.gca(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xlabel("ROC AUC"); plt.ylabel("Density"); plt.title("ROC AUC distributions (CV aggregated)")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "auc_distributions.png"), bbox_inches="tight")
    plt.close()

    # Save fold summary JSON/CSV
    with open(os.path.join(run_dir, "fold_summary.json"), "w") as f:
        json.dump(fold_summaries, f, indent=2)
    pd.DataFrame(fold_summaries).to_csv(os.path.join(run_dir, "fold_summary.csv"), index=False)

    print("\n✅ Done.")
    print(f"Run directory: {run_dir}")
    print(f"- Per-fold CSVs: {fold_csv_dir}")
    print(f"- Params JSON:   {os.path.join(run_dir, 'params.json')}")
    print(f"- AUC plot:      {os.path.join(run_dir, 'auc_distributions.png')}")

if __name__ == "__main__":
    main()
