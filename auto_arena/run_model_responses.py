import os
import json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from vllm import LLM, SamplingParams
from utils import existing_model_paths
from tqdm import tqdm
import time

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if model_info == "OPENAI":
        return None, None

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        print("Tokenizer Loaded; Loading Model")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, device_map='auto', torch_dtype="auto")

        # Convert HF model to VLLM model
        vllm_model = LLM(model_name=model_info)
        return tokenizer, vllm_model

    raise FileNotFoundError("Model path does not exist")

def run_hf_model(prompts, tokenizer, model):
    if not callable(tokenizer):
        raise TypeError("Tokenizer is not callable. Ensure it is correctly initialized.")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:7")

    outputs = model.generate(**inputs)  # 使用默认生成参数

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

def run_vllm_model(prompts, model):
    sampling_params = SamplingParams()
    outputs = model.generate(prompts, sampling_params=sampling_params)
    
    responses = []
    for output in outputs:
        responses.append(output.text)
    
    return responses

def run_openai_model(prompts, model_name, client):
    if "3.5-turbo-0125" in model_name:
        model_name = "gpt-3.5-turbo-0125"
    elif "4-1106" in model_name:
        model_name = "gpt-4-1106-preview"
    responses = []
    for prompt in prompts:
        completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
        )
        text = completion.choices[0].message.content
        responses.append(str(text))
    return responses

def save_responses(prompts, responses, model_name, output_dir):
    timestamp = int(time.time())
    empty_responses = []
    for i, response in enumerate(responses):
        prompt_id = i + 1
        directory = os.path.join(output_dir, f"mt_bench_question_{prompt_id}")
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{prompt_id}|{model_name}|{timestamp}.jsonl")
        with open(filename, 'w') as f:
            json.dump({'response': [response], 'prompt': prompts[i]}, f, indent=4)
        if not response.strip():
            empty_responses.append({
                'prompt_id': prompt_id,
                'model_name': model_name,
                'timestamp': timestamp,
                'response': response
            })
    if empty_responses:
        with open(os.path.join(output_dir, 'empty_responses.jsonl'), 'a') as f:
            for empty_response in empty_responses:
                f.write(json.dumps(empty_response) + '\n')
    print(f"Responses for {model_name} saved to {output_dir}")

def get_responses(prompts, model_name, output_dir="model_responses"):
    tokenizer, model = load_model(model_name)

    if isinstance(model, LLM):
        responses = run_vllm_model(prompts, model)
    elif model_name == "OPENAI":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai.api_key)
        responses = run_openai_model(prompts, model_name, client)
    else:
        responses = run_hf_model(prompts, tokenizer, model)

    save_responses(prompts, responses, model_name, output_dir)
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
