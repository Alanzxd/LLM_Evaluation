import os
import json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
import uuid
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_model(model_name, device):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError(f"Unsupported model: {model_name}")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer Loaded: {type(tokenizer)}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, device_map='auto', torch_dtype="auto").to(device)
        print(f"Model Loaded: {type(model)}")

        return tokenizer, model

    raise FileNotFoundError("Model path does not exist")

def run_model(prompts, tokenizer, model, device, batch_size=1):
    responses = []
    sampling_params = SamplingParams()  # Use default parameters

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=200)

        for output in outputs:
            responses.append(tokenizer.decode(output, skip_special_tokens=True))

    return responses

def save_responses(responses, model_name, output_dir, prompt_ids):
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = prompt_ids[i]
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")
        output_file = os.path.join(directory, f"{prompt_id}|{model_name}|{timestamp}.jsonl")
        with open(output_file, 'w') as f:
            json.dump({"response": response}, f, indent=4)
        if response.strip() == "":
            empty_responses.append((model_name, prompt_id))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")

def get_responses(rank, world_size, prompts, model_name, output_dir="model_responses", batch_size=1):
    device = torch.device(f"cuda:{rank}")
    tokenizer, model = load_model(model_name, device)

    responses = run_model(prompts, tokenizer, model, device, batch_size)

    save_responses(responses, model_name, output_dir, list(range(len(prompts))))
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file):
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(rank, world_size, model_names, output_dir="model_responses", batch_size=1):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    prompts = get_questions()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in tqdm(model_names):
        print(f"Running model: {model_name} on rank: {rank}")
        get_responses(rank, world_size, prompts, model_name, output_dir, batch_size)

def run_parallel(world_size, model_names, output_dir="model_responses", batch_size=1):
    model_names = model_names.split(",")
    mp.spawn(run_all_models, args=(world_size, model_names, output_dir, batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    fire.Fire(run_parallel)

