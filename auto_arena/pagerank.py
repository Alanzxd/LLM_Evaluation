import json
import networkx as nx
import numpy as np
from utils import existing_model_paths, gt_scores
import fire
import scipy.stats
import pandas as pd

model_names = list(existing_model_paths.keys())


def load_jsonl(filename):
    """Load data from a JSONL file."""
    with open(filename, 'r') as file:
        return np.array([json.loads(line.strip()) for line in file][0])

def original_pagerank(link_matrix, d=0.85, tol=1e-8): 
    transition_matrix = link_matrix / link_matrix.sum(axis=1, keepdims=True)

    def pagerank(T, d, tol):
        n = T.shape[0]
        PR = np.ones(n) / n  # Start with equal probability for each node
        delta = 1           # Change in PR from one step to the next
        while delta > tol:
            new_PR = (1 - d) / n + d * np.dot(T.T, PR)
            delta = np.linalg.norm(new_PR - PR, 1)  # Sum of absolute differences
            PR = new_PR
        return PR

    scores = pagerank(transition_matrix, d, tol)
    sorted_indices = np.argsort(-scores) 

    return scores, sorted_indices

def digraph_bt_pagerank(bt_scores, alpha=0.85, tol=1e-6):
    # Create a directed graph
    G = nx.DiGraph()

    for i in range(len(bt_scores)):
        G.add_node(i)

    for i in range(len(bt_scores)):
        for j in range(len(bt_scores)):
            if i != j:  
                weight = np.exp(bt_scores[i] - bt_scores[j])
                G.add_edge(i, j, weight=weight)
    

    pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=None, weight='weight', tol=tol)
    scores = []
    for i in range(len(pagerank_scores.keys())):
        scores.append(pagerank_scores[i])
    # sorted_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)
    # print(sorted_nodes)
    scores = np.array(scores)
    sorted_indices = np.argsort(-scores)

    return scores, sorted_indices

def bt_pagerank(link_matrix, bt_scores, d=0.85, tol=1e-8):
    adjusted_link_matrix = np.zeros_like(link_matrix)
    for i in range(link_matrix.shape[0]):
        for j in range(link_matrix.shape[1]):
            if link_matrix[i, j] > 0:  # Only adjust existing links
                # Scaling factor: exponential of BT score
                adjusted_link_matrix[i, j] = link_matrix[i, j] * np.exp(-bt_scores[j])

    # Normalize adjusted link matrix to get transition probabilities
    transition_matrix = adjusted_link_matrix / adjusted_link_matrix.sum(axis=1, keepdims=True)

    def pagerank(T, d, tol):
        n = T.shape[0]
        PR = np.ones(n) / n  # Start with equal probability for each node
        delta = 1
        while delta > tol:
            new_PR = (1 - d) / n + d * np.dot(T.T, PR)
            delta = np.linalg.norm(new_PR - PR, 1)
            PR = new_PR
        return PR

    scores = pagerank(transition_matrix, d, tol)
    sorted_indices = np.argsort(-scores) 

    return scores, sorted_indices


def rank_scores(scores):
    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(sorted_scores):
        ranks[index] = rank
    return ranks


def main(file_path, algo, d=0.85):
    # Sanity Check
    # model_list = ["A", "B", "C", "D", "E"]
    # matrix = np.array(
    #     [[0, 60, 20, 15, 5],
    #     [50, 0, 10, 5, 1],
    #     [40, 20, 0, 15, 2],
    #     [25, 15, 20, 0, 0],
    #     [5, 15, 20, 60, 0]]
    #     )
    # print(calc_score(matrix))

    matrix = load_jsonl(file_path)
    matrix = np.array(-matrix) 
    matrix = np.exp(matrix)

    df = pd.DataFrame(matrix, index=model_names, columns=model_names)
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)  
    print(df)
    print("="*50)

   

    # sum_score = list(df.sum())
    # sum_sorted_indices = np.argsort(sum_score)[::-1]


    df_normalized = df.div(df.sum(axis=1), axis='rows')
    normalized_sum_score = list(df_normalized.sum())
    norm_sum_sorted_indices = np.argsort(normalized_sum_score)[::-1]

    if algo == "original_pagerank": 
        pg_scores, pg_sorted_indices = original_pagerank(matrix,d=d)
        pg_ranks = rank_scores(pg_scores)
    if algo == "bt_pagerank":
        bt = np.array([-0.86629933, 0.1032422,  -0.41369063, -0.09334841, -0.14655842, -0.07036105, 0.5359005,   0.0600469,   0.64905757,  0.5880494 ])
        pg_scores, pg_sorted_indices = bt_pagerank(matrix,bt, d=d)
        pg_ranks = rank_scores(pg_scores)
    if algo == "digraph_bt_pagerank":
        bt = np.array([-0.86629933, 0.1032422,  -0.41369063, -0.09334841, -0.14655842, -0.07036105, 0.5359005,   0.0600469,   0.64905757,  0.5880494 ])
        pg_scores, pg_sorted_indices = digraph_bt_pagerank(bt)
        pg_ranks = rank_scores(pg_scores)

    
    gt_scores_list = [gt_scores[model] for model in model_names]
    gt_ranks = rank_scores(gt_scores_list)
    gt_sorted_indices = np.argsort(gt_scores_list)[::-1]

    # pearson_sum_pg_correlation, _ = scipy.stats.pearsonr(sum_score, pg_scores)
    # pearson_norm_sum_pg_correlation, _ = scipy.stats.pearsonr(normalized_sum_score, pg_scores)
    # pearson_sum_correlation, _ = scipy.stats.pearsonr(sum_score, gt_scores_list)
    # pearson_norm_sum_correlation, _ = scipy.stats.pearsonr(normalized_sum_score, gt_scores_list)

    pearson_score_correlation, _ = scipy.stats.pearsonr(pg_scores, gt_scores_list)
    pearson_rank_correlation, _ = scipy.stats.pearsonr(pg_ranks, gt_ranks)

    kt_rank_correlation, _ = scipy.stats.kendalltau(pg_ranks, gt_ranks)
    print(f"Algo: {algo}")
    print("Ranked Scores:")
    for index in pg_sorted_indices:
        model_name = model_names[index]
        pg_score = pg_scores[index]
        gt_score = gt_scores[model_name]
        print(f"{model_name}: PageRank Score = {pg_score:.4f}, GT Score = {gt_score}")

    print("="*50)
    # print("Pearson Corr between Total Votes Received and PageRank Score:", pearson_sum_pg_correlation)
    # print("Pearson Corr between Normalized Votes Received and PageRank Score:", pearson_norm_sum_pg_correlation)

    # print("Pearson Corr between Total Votes Received and ChatArena Elo:", pearson_sum_correlation)
    # print("Pearson Corr between Normalized Votes Received and ChatArena Elo:", pearson_norm_sum_correlation) 

    print("-"*50)
    print(f"Pearson Corr between {algo} Scores and ChatArena Elo:", pearson_score_correlation)
    print(f"Pearson Corr between {algo} and ChatArena Rankings:", pearson_rank_correlation)
    print(f"Kendall Tau Corr between {algo} and ChatArena Rankings: ", kt_rank_correlation)
    print("="*50)
    header = f"Rank| Normalized Votes   | {algo} Score      | Latest ChatArena     "
    print(header)
    print("-"* len(header))
    for rank in range(len(model_names)):
        print(f"{rank + 1} | {model_names[norm_sum_sorted_indices[rank]]:20} | {model_names[pg_sorted_indices[rank]]:20} | {model_names[gt_sorted_indices[rank]]}")
    

if __name__ == "__main__":
    fire.Fire(main)