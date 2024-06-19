import os
import json
import fire
import torch
import gc
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
from datetime import datetime

# Disable Tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        vllm_model = LLM(model=model_info)
        return vllm_model

    raise FileNotFoundError("Model path does not exist")

def run_vllm_model(prompts, model, max_new_tokens, temperature, top_p, top_k):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k
    )
    outputs = model.generate(prompts, sampling_params=sampling_params)
    
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    
    return responses

def save_responses(responses, model_name, output_dir, prompt_ids, prompts):
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
            empty_responses.append((model_name, prompt_id, prompts[i], response))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid, prompt, response in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")

def get_responses(prompts, model_name, output_dir="model_responses", max_new_tokens=200, temperature=0.7, top_p=0.95, top_k=40):
    model = load_model(model_name)
    responses = run_vllm_model(prompts, model, max_new_tokens, temperature, top_p, top_k)
    save_responses(responses, model_name, output_dir, list(range(len(prompts))), prompts)
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

def run_all_models(output_dir="model_responses", model_names="vicuna-33b", max_new_tokens=200, batch_size=1, temperature=0.7, top_p=0.95, top_k=40):
    prompts = get_questions()
    model_names = model_names.split(',')
    
    os.makedirs(output_dir, exist_ok=True)

    for model_name in tqdm(model_names):
        print(f"Processing model: {model_name}")
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
            get_responses(batch_prompts, model_name, output_dir, max_new_tokens, temperature, top_p, top_k)

if __name__ == "__main__":
    fire.Fire(run_all_models)


