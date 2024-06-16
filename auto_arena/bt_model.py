import os
import json
import torch
import numpy as np
from torch import optim
from utils import existing_model_paths, gt_scores
from tqdm import tqdm
import fire

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = list(existing_model_paths.keys())

# Format: (model1, model2, preference_for_model1)
# preference_for_model1 is 1 if model1 is preferred over model2, 0 otherwise

def prepare_comparisons(base_dir, model_names=model_names):
    model_index_map = {name: idx for idx, name in enumerate(model_names)} 
    save_dic = {}

    for subdir in os.listdir(base_dir):
        print(f"Working on {subdir}")
        comparisons = []

        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    for each in record: 
                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = model_index_map.get(model1)
                        idx2 = model_index_map.get(model2)

                        if idx1 is not None and idx2 is not None:
                            if winner == model1: 
                                comparisons.append((idx1, idx2, 0))
                            elif winner == model2: 
                                comparisons.append((idx1, idx2, 1))

        bt_scores = calc_bt_score(comparisons)
        save_dic[subdir] = bt_scores
    
    file_path = os.path.join('transition_matrices', 'matrix.jsonl')

    with open(file_path, 'w') as file:
        for name in sorted(save_dic, key=lambda x: model_index_map[x]):
            json_line = json.dumps(save_dic[name].tolist()) 
            file.write(json_line + '\n')

    print(f"Data saved to {file_path}")




def calc_bt_score(comparisons, model_names=model_names): 
    comparisons_tensor = torch.tensor(comparisons, dtype=torch.float32).to(device)
    num_models = len(model_names)

    ξ = torch.zeros(num_models, requires_grad=True, device=device)
    optimizer = optim.AdamW([ξ], lr=0.0001)

    def bt_log_likelihood_pytorch(ξ, comparisons):
        likelihood = 0
        model1_indices = comparisons[:, 0].long()
        model2_indices = comparisons[:, 1].long()
        prefs = comparisons[:, 2]
        
        ps = torch.sigmoid(ξ[model1_indices] - ξ[model2_indices])
        likelihood = torch.sum(prefs * torch.log(ps + 1e-15) + (1 - prefs) * torch.log(1 - ps + 1e-15))
        
        return -likelihood

    batch_size = 64
    num_comparisons = comparisons_tensor.shape[0]
    num_batches = (num_comparisons + batch_size - 1) // batch_size

    for iteration in range(1000):
        total_loss = 0
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, num_comparisons)
            batch = comparisons_tensor[batch_start:batch_end]

            optimizer.zero_grad()
            loss = bt_log_likelihood_pytorch(ξ, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if iteration % 100 == 0: 
            print(f"Iteration {iteration}, Average Batch Loss: {total_loss / num_batches}")

    ξ_estimates = ξ.detach().cpu().numpy() 
    print(ξ_estimates)

    return ξ_estimates

def main(base_dir = "judgements"):
    print(model_names)
    prepare_comparisons(base_dir)

if __name__ == '__main__':
    fire.Fire(main)