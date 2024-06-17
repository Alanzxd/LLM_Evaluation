import os
import json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import existing_model_paths
from tqdm import tqdm
import uuid
from datetime import datetime

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        print(f"Tokenizer Loaded: {type(tokenizer)}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info)
        model.to("cuda")  # Ensure the model is moved to the GPU
        print(f"Model Loaded: {type(model)}")
        return tokenizer, model

    raise FileNotFoundError("Model path does not exist")

def run_model(prompts, tokenizer, model, num_beams=1):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

    try:
        with torch.cuda.amp.autocast():
            outputs = model.generate(**inputs, max_new_tokens=200, num_beams=num_beams)  # Use max_new_tokens and num_beams
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OutOfMemoryError during model.generate: {e}")
        torch.cuda.empty_cache()
        return [""] * len(prompts)  # Return empty responses in case of an error

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
        timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')[:-3]  # Generate timestamp in the desired format
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        output_file = os.path.join(directory, f"{prompt_id}|{model_name}|{timestamp}.jsonl")
        with open(output_file, 'w') as f:
            json.dump({"response": response}, f, indent=4)
        if response.strip() == "":
            empty_responses.append((model_name, prompt_id))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")

def get_responses(prompts, model_name, output_dir="model_responses", num_beams=1):
    if model_name in ["gpt4-1106", "gpt3.5-turbo-0125"]:
        print(f"Skipping model: {model_name}")
        return []
    else:
        tokenizer, model = load_model(model_name)
        responses = run_model(prompts, tokenizer, model, num_beams)

    save_responses(responses, model_name, output_dir, list(range(len(prompts))))
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(model_names=None, output_dir="model_responses", num_beams=1):
    prompts = get_questions()
    if model_names is None:
        model_names = list(existing_model_paths.keys())
    else:
        model_names = model_names.split(",")  # Split the string into a list of model names
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in tqdm(model_names):
        get_responses(prompts, model_name, output_dir, num_beams)

if __name__ == "__main__":
    fire.Fire(run_all_models)
