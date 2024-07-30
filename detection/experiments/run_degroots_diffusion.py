import os
import sys
# os.chdir('/sise/home/tommarz/hate_speech_detection/')
import pickle
import numpy as np
import igraph as ig
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
# from detection.diffusion_method.degroots_diffusion import degroots_diffusion

import igraph as ig

def degroots_diffusion(g, seed_hate_users=None, frac=None, size=None, initial_belief = 0, iterations=10, random_state=None):
    # hate_nodes_indices = seed_hate_users.indices
    # g = h.copy()
    # g.reverse_edges()
    labeled_nodes = g.vs.select(lambda v: v['label']!=-1)
    if seed_hate_users is None:
        if size is None and frac is None:
            raise ValueError("Please pass a list of seed hate users or size/frac to sample from")
        if size is None:
            size = int(frac * len(labeled_nodes))
        np.random.seed(random_state)
        seed_hate_users = np.random.choice(labeled_nodes.indices, size, replace=False)
    initial_beliefs = np.full(g.vcount(), initial_belief)   
    initial_beliefs[seed_hate_users] = 1

    # Get the adjacency matrix as a numpy array
    A = np.array(g.get_adjacency(attribute='weight').data)
    
    # Normalize the adjacency matrix
    row_sums = A.sum(axis=1, where=(A > 0))  # Sum only where there are non-zero entries
    A_normalized = np.divide(A, row_sums[:, np.newaxis], out=np.zeros_like(A), where=row_sums[:, np.newaxis] != 0)
    # history = [initial_beliefs.copy()]
    beliefs = initial_beliefs
    # Simulation of opinion dynamics
    for _ in range(iterations):
        beliefs = A_normalized.dot(beliefs)
        # beliefs = A_normalized.dot(history[-1])
        # history.append(beliefs.copy())
    return [beliefs]

dataset = sys.argv[1]
seed = 0
print(dataset)

network_output_dir = "/sise/home/tommarz/hate_speech_detection/data/networks_data"
raw_graphs_dict_path = os.path.join(network_output_dir, "raw_graphs_dict.p")
network_dataset_output_dir = os.path.join(network_output_dir, dataset)
raw_network_path  = os.path.join(network_dataset_output_dir, "raw_network.p")
largest_cc_path  = os.path.join(network_dataset_output_dir, "largest_cc.p")

with open(largest_cc_path, 'rb') as f:
    g = pickle.load(f)
g.summary()

# g = largest_cc.copy()
g.reverse_edges()

labeled_nodes = g.vs.select(lambda v: v['label'] != -1)

# %%
y_true = labeled_nodes['label']

np.random.seed(seed)
seeds = np.random.randint(0, 2**32-1, 5)

histories = []
metrics = []
for seed in seeds:
    history = degroots_diffusion(g, frac=0.05, iterations=1, initial_belief=0.5, random_state=seed)
    # histories.append(history)
    
    labeled_nodes_opinions = history[-1][labeled_nodes.indices]

    scaler = MinMaxScaler()
    scaled_opinions = scaler.fit_transform(labeled_nodes_opinions.reshape(-1, 1)).flatten()

    preds = scaled_opinions >= 0.5
    metrics.append([accuracy_score(y_true, preds), precision_score(y_true, preds),  recall_score(y_true, preds),  f1_score(y_true, preds), roc_auc_score(y_true, labeled_nodes_opinions)])

results_df = pd.DataFrame(metrics, columns=['acccuracy', 'precision', 'recall', 'f1', 'roc_auc'], index=seeds)

df = pd.concat([results_df.mean(axis=0), results_df.std(axis=0)], axis=1, names=['mean', 'std'])

s = "& DeGroot's Diffusion"
for mean, std in df.values:
    s += (f' & ${mean:.3f} \pm ${std:.3f}')
s+= '\\\\'
print(s)