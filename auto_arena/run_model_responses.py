import os
import json
import fire
import torch
import gc
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
import uuid
from datetime import datetime

# 禁用 Tokenizers 并行处理的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        vllm_model = LLM(model=model_info, enforce_eager=True)  # 强制使用急切模式
        vllm_model = torch.nn.DataParallel(vllm_model)  # 使用 DataParallel
        return vllm_model

    raise FileNotFoundError("Model path does not exist")

def run_vllm_model(prompts, model, max_new_tokens=100):
    sampling_params = SamplingParams(max_tokens=max_new_tokens)
    outputs = model.module.generate(prompts, sampling_params=sampling_params)  # 使用 model.module 进行推理
    
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    
    return responses

def save_responses(responses, model_name, output_dir, prompt_ids):
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = prompt_ids[i]
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
        output_file = os.path.join(directory, f"{prompt_id}|{model_name}|{timestamp}.jsonl")
        with open(output_file, 'w') as f:
            json.dump({"response": response}, f, indent=4)
        if response.strip() == "":
            empty_responses.append((model_name, prompt_id))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")

def get_responses(prompts, model_name, output_dir="model_responses", max_new_tokens=100):
    model = load_model(model_name)

    responses = run_vllm_model(prompts, model, max_new_tokens)

    save_responses(responses, model_name, output_dir, list(range(len(prompts))))
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(output_dir="model_responses", model_names=None, max_new_tokens=100, batch_size=1):
    prompts = get_questions()
    if model_names:
        model_names = model_names.split(",")
    else:
        model_names = list(existing_model_paths.keys())

    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in tqdm(model_names):
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            get_responses(batch_prompts, model_name, output_dir, max_new_tokens)
            torch.cuda.empty_cache()  # 手动释放未使用的显存
            gc.collect()  # 手动调用垃圾回收

if __name__ == "__main__":
    fire.Fire(run_all_models)


