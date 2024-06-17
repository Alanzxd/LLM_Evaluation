import os
import fire
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from vllm import LLM, SamplingParams

from utils import existing_model_paths

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

def save_responses(responses, output_file):
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"Responses saved to {output_file}")

def get_response(prompts, model_name, output_file="responses.json"):
    tokenizer, model = load_model(model_name)

    if isinstance(model, LLM):
        responses = run_vllm_model(prompts, model)
    elif model_name == "OPENAI":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai.api_key)
        responses = run_openai_model(prompts, model_name, client)
    else:
        responses = run_hf_model(prompts, tokenizer, model)

    save_responses(responses, output_file)
    return responses

if __name__ == "__main__":
    fire.Fire(get_response)
