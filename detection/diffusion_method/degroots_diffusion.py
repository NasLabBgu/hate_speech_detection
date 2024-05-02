import igraph as ig
import numpy as np

def degroots_diffusion(h, seed_hate_users=None, frac=None, size=None, initial_belief = 0.5, iterations=10, random_state=None):
    # hate_nodes_indices = seed_hate_users.indices
    g = h.copy()
    g.reverse_edges()
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
    history = [initial_beliefs.copy()]
    # Simulation of opinion dynamics
    for _ in range(iterations):
        beliefs = A_normalized.dot(history[-1])
        history.append(beliefs.copy())
    return history