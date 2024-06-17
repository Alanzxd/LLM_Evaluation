import os
import json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
import uuid
import openai

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        print(f"Tokenizer Loaded: {type(tokenizer)}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, device_map='auto', torch_dtype="auto")
        print(f"Model Loaded: {type(model)}")

        # Convert HF model to VLLM model
        vllm_model = LLM(model_name=model_info)
        return tokenizer, vllm_model

    raise FileNotFoundError("Model path does not exist")

def run_vllm_model(prompts, model):
    sampling_params = SamplingParams()
    outputs = model.generate(prompts, sampling_params=sampling_params)
    
    responses = []
    for output in outputs:
        responses.append(output.text)
    
    return responses

def run_openai_model(prompts, model_name, temperature=0.7, max_tokens=1024):
    openai.api_key = "sk-proj-tJPuS2rvAEYubMXSCxfCT3BlbkFJHXnkL3PMGmNhTiMJk02V"  # 设置你的 OpenAI API 密钥

    if "3.5-turbo-0125" in model_name:
        model_name = "gpt-3.5-turbo-0125"
    elif "4-1106" in model_name:
        model_name = "gpt-4-1106-preview"
    
    responses = []
    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response.choices[0].message["content"].strip()
        responses.append(text)
    return responses

def save_responses(responses, model_name, output_dir, prompt_ids):
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = prompt_ids[i]
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        output_file = os.path.join(directory, f"{prompt_id}|{model_name}|{uuid.uuid4()}.jsonl")
        with open(output_file, 'w') as f:
            json.dump({"response": response}, f, indent=4)
        if response.strip() == "":
            empty_responses.append((model_name, prompt_id))

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")

def get_responses(prompts, model_name, output_dir="model_responses"):
    if model_name in ["gpt4-1106", "gpt3.5-turbo-0125"]:
        responses = run_openai_model(prompts, model_name)
    else:
        tokenizer, model = load_model(model_name)
        responses = run_vllm_model(prompts, model)

    save_responses(responses, model_name, output_dir, list(range(len(prompts))))
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions():
    questions = load_jsonl("mt_bench_questions.jsonl")
    return [question['turns'][0] for question in questions]

def run_all_models(output_dir="model_responses"):
    prompts = get_questions()
    model_names = list(existing_model_paths.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in tqdm(model_names):
        get_responses(prompts, model_name, output_dir)

if __name__ == "__main__":
    fire.Fire(run_all_models)



