import os
import json
import fire
import torch
import gc
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
from datetime import datetime

# 禁用 Tokenizers 并行处理的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EOS_TOKEN = "<|endoftext|>"  # Define the EOS token if needed

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        vllm_model = LLM(model=model_info, gpu_memory_utilization=0.9)
        return vllm_model

    raise FileNotFoundError("Model path does not exist")

def get_sampling_params(model_name, max_new_tokens, temperature, top_p):
    # Default hyperparameters for different models
    if "qwen1.5" in model_name or "vicuna" in model_name:
        return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        return SamplingParams(max_tokens=max_new_tokens)

def format_prompt(prompt):
    return f"{prompt} {EOS_TOKEN}"

def run_vllm_model(prompts, model, sampling_params):
    formatted_prompts = [format_prompt(prompt) for prompt in prompts]
    outputs = model.generate(formatted_prompts, sampling_params=sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses

def save_responses(responses, model_name, output_dir, prompt_ids):
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = prompt_ids[i]
        timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
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

def get_responses(prompts, model_name, output_dir="model_responses", max_new_tokens=200, temperature=0.7, top_p=0.9):
    model = load_model(model_name)
    sampling_params = get_sampling_params(model_name, max_new_tokens, temperature, top_p)
    responses = run_vllm_model(prompts, model, sampling_params)
    save_responses(responses, model_name, output_dir, list(range(len(prompts))))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(output_dir="model_responses", model_names="mistral-7b-instruct-2", max_new_tokens=200, batch_size=1, temperature=0.7, top_p=0.9):
    prompts = get_questions()
    model_names = model_names.split(',')
    
    os.makedirs(output_dir, exist_ok=True)

    for model_name in tqdm(model_names):
        print(f"Processing model: {model_name}")
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
            get_responses(batch_prompts, model_name, output_dir, max_new_tokens, temperature, top_p)

if __name__ == "__main__":
    fire.Fire(run_all_models)
