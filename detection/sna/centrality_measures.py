import os
import igraph as ig
import pandas as pd
import pickle
import sys


main_dir = '/sise/home/tommarz/hate_speech_detection/'
detection_dir = os.path.join(main_dir, 'detection')
sna_dir = os.path.join(detection_dir, 'sna')

def load_graph(path):
    try:
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found. Please ensure the file exists in the directory.")

def calculate_centrality_measures(graph):
    centrality_measures = {
        'vertex_id': list(range(graph.vcount())),
        'degree': graph.degree(),
        'betweenness': graph.betweenness(),
        'closeness': graph.closeness(),
        'eigenvector': graph.eigenvector_centrality(),
        'pagerank': graph.pagerank(),
        'hub_score': graph.hub_score(),
        'authority_score': graph.authority_score()
    }
    return pd.DataFrame(centrality_measures)

def save_to_csv(df, dataset_name):
    filename = f"{dataset_name}_centrality_measures.csv"
    output_path = os.path.join(sna_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Centrality measures saved to {filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python centrality_measures.py <dataset_name>")
        sys.exit(1)
    
    dataset = sys.argv[1].strip()
    largest_cc_path  = os.path.join(f"/sise/home/tommarz/hate_speech_detection/data/networks_data/{dataset}/largest_cc.p")

    g = load_graph(largest_cc_path)
    centrality_df = calculate_centrality_measures(g)
    save_to_csv(centrality_df, dataset)
    
if __name__ == "__main__":
    main()