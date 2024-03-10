import pandas as pd
import numpy as np
import os

def set_path(new_path: str):
    os.environ["DATA_PATH"] = new_path

def get_data_path():
    return os.environ.get("DATA_PATH", "data/")

def get_cluster_weights():
    path = get_data_path()
    return pd.read_excel(path + "cluster_weights.xlsx").set_index("cluster")

def get_train_data():
    path = get_data_path()
    return pd.read_parquet(path + "train_data.pqt")

def get_test_data():
    path = get_data_path()
    return pd.read_parquet(path + "test_data.pqt")

def get_sample_submission():
    path = get_data_path()
    return pd.read_csv(path + "sample_submission.csv")

def get_final_proba(test_start_cluster_proba: pd.DataFrame, transition_proba: np.array):
    # test_start_cluster_proba: (n_samples, n_clusters)
    # transition_proba: (n_samples, n_clusters, n_clusters)
    # return (n_samples, n_clusters)
    
    return np.einsum("ij,ijk->ik", test_start_cluster_proba, transition_proba)

clusters = [
    '{other}',
    '{}',
    '{α, β}',
    '{α, γ}',
    '{α, δ}',
    '{α, ε, η}',
    '{α, ε, θ}',
    '{α, ε, ψ}',
    '{α, ε}',
    '{α, η}',
    '{α, θ}',
    '{α, λ}',
    '{α, μ}',
    '{α, π}',
    '{α, ψ}',
    '{α}',
    '{λ}'
]