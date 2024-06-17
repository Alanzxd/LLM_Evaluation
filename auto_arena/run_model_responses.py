import os
import json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
import uuid
import torch.distributed as dist
import torch.multiprocessing as mp

def load_model(model_name, device):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError(f"Unsupported model: {model_name}")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        if not isinstance(tokenizer, AutoTokenizer):
            raise TypeError(f"Loaded tokenizer is not an instance of AutoTokenizer: {type(tokenizer)}")
        print(f"Tokenizer Loaded: {type(tokenizer)}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, torch_dtype="auto")
        model.to(device)  # Move the model to the correct GPU device
        print(f"Model Loaded: {type(model)}")
        return tokenizer, model

    raise FileNotFoundError(f"Model path does not exist: {model_info}")

def run_model(prompts, tokenizer, model, device):
    if not isinstance(tokenizer, AutoTokenizer):
        raise TypeError("Tokenizer is not an instance of AutoTokenizer. Ensure it is correctly initialized.")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=200)  # Use max_new_tokens to set the length of the generation

    responses = []
    for i in range(outputs.shape[0]):
        full_response = tokenizer.decode(outputs[i], skip_special_tokens=True)
        prompt_end_idx = full_response.find(prompts[i]) + len(prompts[i])
        if prompt_end_idx > -1 and prompt_end_idx < len(full_response):
            response = full_response[prompt_end_idx:].strip()
        else:
            response = full_response
        responses.append(response)

    return responses

def save_responses(responses, model_name, output_dir, prompt_ids):
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = prompt_ids[i]
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        timestamp = uuid.uuid1().int >> 64  # Generate a timestamp for the file name
        output_file = os.path.join(directory, f"{prompt_id}|{model_name}|{timestamp}.jsonl")
        with open(output_file, 'w') as f:
            json.dump({"response": response}, f, indent=4)
        if response.strip() == "":
            empty_responses.append((model_name, prompt_id))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")

def get_responses(rank, world_size, prompts, model_name, output_dir):
    device = torch.device(f"cuda:{rank}")
    print(f"Loading model {model_name} on device {device}")
    tokenizer, model = load_model(model_name, device)

    responses = run_model(prompts, tokenizer, model, device)

    save_responses(responses, model_name, output_dir, list(range(len(prompts))))

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(rank, world_size, model_names, output_dir="model_responses"):
    prompts = get_questions()
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    for model_name in model_names:
        print(f"Running model: {model_name} on rank: {rank}")
        get_responses(rank, world_size, prompts, model_name, output_dir)
    
    dist.destroy_process_group()

def run_parallel(world_size, model_names, output_dir="model_responses"):
    model_names = model_names.split(",")
    mp.spawn(run_all_models, args=(world_size, model_names, output_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    fire.Fire(run_parallel)
