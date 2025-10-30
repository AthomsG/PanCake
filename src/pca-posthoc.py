#!/usr/bin/env python3
import os
import argparse
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --------------------------- CLI ---------------------------
def get_args():
    ap = argparse.ArgumentParser(description="Run PCA on filtered model outputs with good ROC AUC performance")
    ap.add_argument("--model_dir", required=True,
                help="Directory containing trained models and fold results")
    ap.add_argument("--input_dir", required=True,
                help="Directory with genes_filtered.parquet & species_filtered.parquet")
    ap.add_argument("--output_dir", required=True,
                help="Directory to save posthoc analysis results")
    ap.add_argument("--auc_threshold", type=float, default=0.85,
                help="ROC AUC threshold to filter high-performing outputs")
    ap.add_argument("--batch_size", type=int, default=128,
                help="Batch size for model inference")
    ap.add_argument("--model_type", choices=["LogReg", "MLP", "both"], default="MLP",
                help="Which model type to analyze")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# --------------------------- Models ---------------------------
class BaselineLogisticRegression(nn.Module):
    """Multi-label logistic regression (one linear layer)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # bias kept
    def forward(self, x):
        return self.linear(x)

class MultiLabelLogReg(nn.Module):
    """Multi-label logistic regression with direct parameters"""
    def __init__(self, d, t):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, t, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(t, dtype=torch.float32))
    def forward(self, X):
        return X @ self.W + self.b

class BetterMLP(nn.Module):
    """2-layer MLP with BatchNorm & Dropout"""
    def __init__(self, in_dim, out_dim, h1=256, h2=128, p_drop=0.2):
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

# --------------------------- Utils ---------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_run_with_models(model_dir):
    """Find the latest run directory containing model weights"""
    for run_dir in sorted(glob.glob(os.path.join(model_dir, "run_*")), reverse=True):
        weights_dir = os.path.join(run_dir, "model-weights")
        fold_csv_dir = os.path.join(run_dir, "fold_csv")
        if os.path.isdir(weights_dir) and os.path.isdir(fold_csv_dir):
            return run_dir, weights_dir, fold_csv_dir
    return None, None, None

def load_model_info(run_dir):
    """Load model parameters from params.json"""
    params_file = os.path.join(run_dir, "params.json")
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    return {}

def load_auc_data(fold_csv_dir, auc_threshold):
    """Load and aggregate AUC data from all folds, filter by threshold"""
    auc_dfs = []
    
    # Load AUC data from each fold
    for auc_file in sorted(glob.glob(os.path.join(fold_csv_dir, "per_gene_auc_fold*.csv"))):
        fold_df = pd.read_csv(auc_file)
        auc_dfs.append(fold_df)
    
    if not auc_dfs:
        raise ValueError(f"No AUC data found in {fold_csv_dir}")
    
    # Combine all fold data
    all_auc_df = pd.concat(auc_dfs, ignore_index=True)
    
    # Group by gene and calculate mean AUC for each model type
    gene_aucs = all_auc_df.groupby("gene").mean()
    
    # Filter genes with AUC above threshold for each model type
    logreg_good_genes = gene_aucs[gene_aucs["auc_logreg"] > auc_threshold].index.tolist()
    mlp_good_genes = gene_aucs[gene_aucs["auc_mlp"] > auc_threshold].index.tolist()
    
    print(f"Found {len(logreg_good_genes)} genes with LogReg AUC > {auc_threshold}")
    print(f"Found {len(mlp_good_genes)} genes with MLP AUC > {auc_threshold}")
    
    return logreg_good_genes, mlp_good_genes, gene_aucs

def load_model(model_file, device):
    """Load a saved model from a .pt file"""
    model_data = torch.load(model_file, map_location=device)
    
    # Extract model metadata
    in_dim = model_data.get("in_dim")
    out_dim = model_data.get("out_dim")
    model_state = model_data.get("model_state")
    species_names = model_data.get("species_names", [])
    gene_names = model_data.get("gene_names", [])
    
    # Detect model type from filename
    if "LogReg" in model_file:
        # Use the MultiLabelLogReg class that matches the saved model structure
        model = MultiLabelLogReg(in_dim, out_dim)
    else:  # MLP
        h1 = model_data.get("h1", 256)
        h2 = model_data.get("h2", 128)
        p_drop = model_data.get("p_drop", 0.2)
        model = BetterMLP(in_dim, out_dim, h1=h1, h2=h2, p_drop=p_drop)
    
    # Load weights
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, gene_names, species_names

@torch.no_grad()
def get_model_outputs(model, data_loader, device, apply_sigmoid=True):
    """Get outputs from model for all samples"""
    outputs = []
    for X_batch, _ in tqdm(data_loader, desc="Getting model outputs"):
        X_batch = X_batch.to(device)
        output = model(X_batch)
        if apply_sigmoid:
            output = torch.sigmoid(output)
        outputs.append(output.cpu().numpy())
    return np.vstack(outputs)

def run_pca_analysis(outputs, n_components=2):
    """Run PCA on model outputs"""
    # Standardize the outputs
    scaler = StandardScaler()
    outputs_scaled = scaler.fit_transform(outputs)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(outputs_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    
    return components, explained_variance, pca

def plot_pca(components, explained_variance, sample_ids=None, save_path=None, title="PCA of Model Outputs"):
    """Plot PCA results"""
    plt.figure(figsize=(4, 3), dpi=300)
    
    # Create scatter plot
    scatter = plt.scatter(components[:, 0], components[:, 1], alpha=0.6, s=30)
    
    # Add labels
    plt.xlabel(f"PC1 ({explained_variance[0]:.2f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
    plt.title(title)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tighten layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.close()

def plot_pca_with_density(components, explained_variance, save_path=None, title="PCA with Density"):
    """Plot PCA results with density information"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
    
    # Left plot: scatter plot
    scatter = ax1.scatter(components[:, 0], components[:, 1], alpha=0.6, s=30)
    ax1.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)")
    ax1.set_ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
    ax1.set_title("PCA Scatter Plot")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right plot: kernel density estimation
    sns.kdeplot(x=components[:, 0], y=components[:, 1], cmap="viridis", fill=True, ax=ax2)
    ax2.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)")
    ax2.set_ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
    ax2.set_title("Density Estimation")
    
    # Overall title
    fig.suptitle(title, fontsize=16)
    
    # Tighten layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.close()

def find_cluster_centers_and_reconstruct(components, filtered_outputs, pca, scaler, 
                                          good_gene_names, n_clusters=3):
    """
    Find cluster centers in PCA space and reconstruct gene probabilities at those centers.
    Returns the 'inverse PCA' - gene probability patterns at cluster centers.
    """
    # Perform K-means clustering on PCA components
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(components)
    cluster_centers_pca = kmeans.cluster_centers_  # Shape: (n_clusters, 2)
    
    # Count samples per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sample_counts = dict(zip(unique, counts))
    
    # Inverse transform: PCA space -> scaled space
    # We need to pad the 2D centers to full dimensionality for inverse transform
    n_features = filtered_outputs.shape[1]
    
    # Get the PCA components (loadings)
    pca_components = pca.components_  # Shape: (2, n_features)
    
    # Reconstruct in scaled space
    reconstructed_scaled = cluster_centers_pca @ pca_components  # (n_clusters, n_features)
    
    # Inverse transform: scaled space -> original space
    reconstructed_probs = scaler.inverse_transform(reconstructed_scaled)  # (n_clusters, n_features)
    
    # Clip to valid probability range [0, 1]
    reconstructed_probs = np.clip(reconstructed_probs, 0.0, 1.0)
    
    # Create DataFrame for easy interpretation
    cluster_gene_probs = pd.DataFrame(
        reconstructed_probs,
        columns=good_gene_names,
        index=[f"Cluster_{i+1}" for i in range(n_clusters)]
    )
    
    return cluster_gene_probs, cluster_labels, cluster_centers_pca, cluster_sample_counts

def plot_pca_with_clusters(components, explained_variance, cluster_labels, cluster_centers, 
                           save_path=None, title="PCA with Clusters"):
    """Plot PCA results with cluster assignments"""
    plt.figure(figsize=(5, 4), dpi=300)
    
    # Plot points colored by cluster
    scatter = plt.scatter(components[:, 0], components[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
    
    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
               c='red', marker='X', s=200, edgecolors='black', linewidths=2,
               label='Cluster Centers')
    
    plt.xlabel(f"PC1 ({explained_variance[0]:.2f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def analyze_fold_model(model_file, gene_names, good_genes, X, device, output_dir, batch_size, model_name, fold_idx):
    """Analyze a single fold model"""
    # Load model
    model, model_gene_names, species_names = load_model(model_file, device)
    
    # Convert gene names to indices if necessary
    if isinstance(model_gene_names, list) and isinstance(gene_names, pd.Index):
        model_gene_names = pd.Index(model_gene_names)
    
    # Find indices of good genes in the model's output
    good_gene_indices = []
    for gene in good_genes:
        if gene in model_gene_names:
            idx = model_gene_names.get_loc(gene)
            good_gene_indices.append(idx)
    
    if not good_gene_indices:
        print(f"No good genes found for {model_file}, skipping...")
        return None
    
    # Create data loader for all samples
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    X_t = to_t(X)
    # Using dummy Y since we don't need it for inference
    Y_dummy = torch.zeros((X.shape[0], 1), dtype=torch.float32)
    all_samples_loader = DataLoader(TensorDataset(X_t, Y_dummy), batch_size=batch_size, shuffle=False)
    
    # Get model outputs for all samples
    outputs = get_model_outputs(model, all_samples_loader, device)
    
    # Filter outputs to only include good genes
    filtered_outputs = outputs[:, good_gene_indices]
    good_gene_names_list = [model_gene_names[i] for i in good_gene_indices]
    
    # Standardize and run PCA
    scaler = StandardScaler()
    outputs_scaled = scaler.fit_transform(filtered_outputs)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(outputs_scaled)
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_dir, f"{model_name}_fold{fold_idx}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Find cluster centers and reconstruct gene probabilities
    cluster_gene_probs, cluster_labels, cluster_centers, cluster_sample_counts = find_cluster_centers_and_reconstruct(
        components, filtered_outputs, pca, scaler, good_gene_names_list, n_clusters=3
    )
    
    # Save reconstructed gene probabilities at cluster centers
    cluster_probs_path = os.path.join(model_output_dir, "cluster_gene_probabilities.csv")
    cluster_gene_probs.to_csv(cluster_probs_path)
    print(f"Saved cluster gene probabilities to {cluster_probs_path}")
    
    # Save cluster statistics including sample counts
    cluster_stats_path = os.path.join(model_output_dir, "cluster_statistics.txt")
    with open(cluster_stats_path, 'w') as f:
        f.write(f"Cluster Statistics for {model_name} Fold {fold_idx}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {len(cluster_labels)}\n")
        f.write(f"Number of clusters: {len(cluster_sample_counts)}\n\n")
        for cluster_id in sorted(cluster_sample_counts.keys()):
            count = cluster_sample_counts[cluster_id]
            percentage = 100.0 * count / len(cluster_labels)
            f.write(f"Cluster {cluster_id + 1}:\n")
            f.write(f"  Samples: {count} ({percentage:.2f}%)\n")
            f.write(f"  Center (PC1, PC2): ({cluster_centers[cluster_id, 0]:.4f}, {cluster_centers[cluster_id, 1]:.4f})\n\n")
    print(f"Saved cluster statistics to {cluster_stats_path}")
    
    # Plot PCA with cluster assignments
    plot_pca_with_clusters(
        components, explained_variance, cluster_labels, cluster_centers,
        save_path=os.path.join(model_output_dir, "pca_clusters.png"),
        title=f"PCA Clusters - {model_name} Fold {fold_idx}"
    )
    
    # Plot PCA results (original)
    plot_pca(
        components, explained_variance, 
        save_path=os.path.join(model_output_dir, "pca_scatter.png"),
        title=f"PCA of {model_name} Fold {fold_idx} (AUC > Threshold)"
    )
    
    # Plot PCA with density
    plot_pca_with_density(
        components, explained_variance,
        save_path=os.path.join(model_output_dir, "pca_density.png"),
        title=f"PCA of {model_name} Fold {fold_idx} (AUC > Threshold)"
    )
    
    # Save PCA components and variance explained
    np.savez(
        os.path.join(model_output_dir, "pca_results.npz"),
        components=components,
        explained_variance=explained_variance,
        good_gene_indices=good_gene_indices,
        cluster_centers_pca=cluster_centers,
        cluster_labels=cluster_labels,
        cluster_sample_counts=np.array([cluster_sample_counts[i] for i in sorted(cluster_sample_counts.keys())])
    )
    
    # Save mapping of PC indices to gene names
    with open(os.path.join(model_output_dir, "good_genes.json"), 'w') as f:
        json.dump({
            "good_genes": good_gene_names_list,
            "explained_variance": explained_variance.tolist(),
            "n_clusters": 3,
            "cluster_centers_pc1_pc2": cluster_centers.tolist(),
            "cluster_sample_counts": {f"Cluster_{k+1}": int(v) for k, v in cluster_sample_counts.items()}
        }, f, indent=2)
    
    return {
        "components": components,
        "explained_variance": explained_variance,
        "good_gene_indices": good_gene_indices,
        "fold_idx": fold_idx,
        "model_name": model_name,
        "cluster_gene_probs": cluster_gene_probs,
        "cluster_sample_counts": cluster_sample_counts
    }

def create_summary_plots(all_results, output_dir):
    """Create summary plots combining results from all folds and models"""
    if not all_results:
        print("No results to summarize")
        return
    
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Plot explained variance by model and fold
    plt.figure(figsize=(4, 3), dpi=300)
    model_types = sorted(set(r["model_name"] for r in all_results))
    
    for model_name in model_types:
        model_results = [r for r in all_results if r["model_name"] == model_name]
        fold_indices = [r["fold_idx"] for r in model_results]
        variances = [r["explained_variance"][0] + r["explained_variance"][1] for r in model_results]
        
        plt.plot(fold_indices, variances, 'o-', label=f"{model_name}")
    
    plt.xlabel("Fold Index")
    plt.ylabel("Total Explained Variance (PC1 + PC2) %")
    plt.title("PCA Explained Variance by Model and Fold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(summary_dir, "explained_variance_by_fold.png"), bbox_inches="tight")
    plt.close()
    
    # Create a summary JSON file
    summary = {
        "model_results": [
            {
                "model_name": r["model_name"],
                "fold_idx": r["fold_idx"],
                "num_good_genes": len(r["good_gene_indices"]),
                "explained_variance": r["explained_variance"].tolist()
            }
            for r in all_results
        ]
    }
    
    with open(os.path.join(summary_dir, "pca_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

# --------------------------- Main ---------------------------
def main():
    args = get_args()
    
    # Setup directories
    run_dir, weights_dir, fold_csv_dir = get_run_with_models(args.model_dir)
    if not run_dir:
        raise ValueError(f"Could not find run directory with model weights in {args.model_dir}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "posthoc-analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing models from: {run_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load model info
    model_info = load_model_info(run_dir)
    print(f"Original model parameters: {json.dumps(model_info, indent=2)}")
    
    # Load data
    genes_pq = os.path.join(args.input_dir, "genes_filtered.parquet")
    species_pq = os.path.join(args.input_dir, "species_filtered.parquet")
    
    if not (os.path.isfile(genes_pq) and os.path.isfile(species_pq)):
        raise FileNotFoundError(f"Could not find filtered parquet files in {args.input_dir}")
    
    df_genes = pd.read_parquet(genes_pq)
    df_species = pd.read_parquet(species_pq)
    
    # Align data
    shared = df_genes.index.intersection(df_species.index)
    df_genes = df_genes.loc[shared]
    df_species = df_species.loc[shared]
    
    # Binary matrices
    X = (df_species > 0).astype(np.float32).values
    species_names = df_species.columns.to_numpy()
    gene_names = df_genes.columns
    
    # Get device
    device = pick_device()
    print(f"Using device: {device}")
    
    # Load AUC data and filter good genes
    logreg_good_genes, mlp_good_genes, gene_aucs = load_auc_data(fold_csv_dir, args.auc_threshold)
    
    # Find model files
    logreg_files = sorted(glob.glob(os.path.join(weights_dir, "LogReg_fold*.pt")))
    mlp_files = sorted(glob.glob(os.path.join(weights_dir, "MLP_fold*.pt")))
    
    # Process models based on user selection
    all_results = []
    
    if args.model_type in ["LogReg", "both"] and logreg_good_genes:
        print(f"\nProcessing {len(logreg_files)} LogReg models...")
        for logreg_file in logreg_files:
            fold_idx = int(os.path.basename(logreg_file).split("_fold")[1].split(".")[0])
            print(f"Analyzing LogReg fold {fold_idx}...")
            result = analyze_fold_model(
                logreg_file, gene_names, logreg_good_genes, X, device, 
                output_dir, args.batch_size, "LogReg", fold_idx
            )
            if result:
                all_results.append(result)
    
    if args.model_type in ["MLP", "both"] and mlp_good_genes:
        print(f"\nProcessing {len(mlp_files)} MLP models...")
        for mlp_file in mlp_files:
            fold_idx = int(os.path.basename(mlp_file).split("_fold")[1].split(".")[0])
            print(f"Analyzing MLP fold {fold_idx}...")
            result = analyze_fold_model(
                mlp_file, gene_names, mlp_good_genes, X, device, 
                output_dir, args.batch_size, "MLP", fold_idx
            )
            if result:
                all_results.append(result)
    
    # Create summary plots
    create_summary_plots(all_results, output_dir)
    
    print(f"\n✅ Done. PCA analysis results saved to: {output_dir}")
    print(f"- Individual model plots: {output_dir}/[ModelName]_fold[N]/")
    print(f"- Summary plots: {output_dir}/summary/")

if __name__ == "__main__":
    main()
