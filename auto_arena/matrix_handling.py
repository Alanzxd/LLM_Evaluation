import sys
import os
import json
import numpy as np
import pandas as pd
from utils import existing_model_paths
from datetime import datetime


# JSON UTILS
def save_to_jsonl(data, filename):
    """Saves a Python data structure to a .jsonl file."""
    with open(filename, 'w') as f:
        f.write(json.dumps(data) + '\n')
    
def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

model_names = list(existing_model_paths.keys())

# CORE
def search_voting_records(judge_name, model_a, model_b, question_id):
    """
    Searches the voting records to check if a specific judgment has been made.
    It checks both model orders since the order in which models are listed might vary.
    """
    records_path = f"judgements/{judge_name}/voting_records.jsonl"
    records = load_jsonl(records_path)[0]
    # Search for any record where the models and question ID match the search criteria
    for record in records:
        models_match = ((record.get("response_A") == model_a and record.get("response_B") == model_b) or
                        (record.get("response_A") == model_b and record.get("response_B") == model_a))
        question_match = record.get("question_id") == question_id
        if models_match and question_match:
            return True
    return False

def update_voting_records(judge_name, response_A_name, response_B_name, question_id, won, data_id):
    """Updates the voting records with a new voting result."""
    records_path = f"judgements/{judge_name}/voting_records.jsonl"
    records = load_jsonl(records_path)[0]
    
    # Append a new record to the list of records
    new_record = {
        "response_A": response_A_name,
        "response_B": response_B_name,
        "Won": won,
        "question_id": question_id,
        "data_id": data_id
    }
    records.append(new_record)  # Ensure this is a flat append operation

    # Save updated records back to the JSONL file
    save_to_jsonl(records, records_path)

def update_transition_vector(judge_name, winner_name):
    """Updates the transition vector for a given judge and winner."""
    vector_path = f"judgements/{judge_name}/transition_vector.jsonl"
    vector = load_jsonl(vector_path)[0][0]
    model_names = list(existing_model_paths.keys())
    try:
        winner_index = model_names.index(winner_name)
        vector[winner_index] += 1
        save_to_jsonl([vector], vector_path)
    except ValueError as e:
        print(f"Error updating vector: {str(e)}")

def concatenate_transition_vectors(model_names=model_names):
    """Concatenate individual transition vectors into a full transition matrix."""
    matrix = np.zeros((len(model_names), len(model_names)))
    for index, judge_name in enumerate(model_names):
        vector_path = f"judgements/{judge_name}/transition_vector.jsonl"
        vector = load_jsonl(vector_path)[0]
        matrix[index, :] = np.array(vector)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    print(matrix)
    directory = "/data/shared/leo/auto_arena/transition_matrices"
    filename = f"{directory}/transition_matrix_{timestamp}.jsonl"
    matrix_list = matrix.tolist()
    save_to_jsonl(matrix_list, filename)
    print(f"Saved matrix to {filename}")
    return matrix

# MISC.
def add_new_model(new_model_name):
    """Adds a new model to the matrix and records, expanding them as necessary."""
    model_names = list(existing_model_paths.keys())
    matrix = load_jsonl("transition_matrix.jsonl")[0]
    records = load_jsonl("voting_records.jsonl")[0] 
    model_names.append(new_model_name)
    new_index = len(matrix)
    matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    records[new_model_name] = []

def print_matrix():
    """Prints the current transition matrix in a formatted way."""
    print("    " + "  ".join(model_names))
    for i, row in enumerate(matrix):
        print(f"{model_names[i]}: {row}")


# FIRST-TIME INITIALIZATION
def initialize_structures(model_names=model_names):
    n = len(model_names)
    transition_matrix = np.zeros((n, n), dtype=int)
    voting_records = {model: [] for model in model_names}
    return transition_matrix, voting_records

def initialize_judge_directories(model_names=model_names):
    """Initialize directories and JSONL files for a new judge."""
    for judge_name in model_names: 
        base_path = f"judgements/{judge_name}"
        os.makedirs(base_path, exist_ok=True)
        voting_records_path = f"{base_path}/voting_records.jsonl"
        transition_vector_path = f"{base_path}/transition_vector.jsonl"
        judgements_vector_path = f"{base_path}/judgements.jsonl"

        # Initialize files if they do not exist
        if not os.path.exists(voting_records_path):
            save_to_jsonl([], voting_records_path)
        if not os.path.exists(transition_vector_path):
            save_to_jsonl([np.zeros(len(model_names)).tolist()], transition_vector_path)
        if not os.path.exists(judgements_vector_path):
            save_to_jsonl([], judgements_vector_path)

def main():
    # pass
    # update_voting_records("llama2-13b-chat", "mistral-7b-instruct-2", "qwen1.5-14b-chat", 1, "mistral-7b-instruct-2", 'some-unique-id-123')
    # update_transition_vector("llama2-13b-chat", "mistral-7b-instruct-2")
    # print(search_voting_records("llama2-13b-chat", "mistral-7b-instruct-2", "qwen1.5-14b-chat",1))

    # print("This operation will initialize a new matrix and delete any old data. If you want to add models, use add_new_model(), update_voting_records(), update_transition_matrix() instead. ")
    # confirm = input("Are you sure you want to proceed? (yes/no): ")
    # if confirm.lower() != 'yes':
    #     print("Operation cancelled.")
    #     sys.exit()
    # initialize_judge_directories()

    

    matrix_now = concatenate_transition_vectors()
    df = pd.DataFrame(matrix_now, index=model_names, columns=model_names)
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)  

    print(df)
    

if __name__ == "__main__":
    main()
