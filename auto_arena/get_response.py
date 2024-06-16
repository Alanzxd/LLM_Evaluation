import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from vllm import LLM

from utils import existing_model_paths

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if model_info == "OPENAI":
        return None, None, model_info

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        print("Tokenizer Loaded; Loading Model")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, device_map = 'auto', torch_dtype="auto")
        return tokenizer, model, model_info
    
    if model_info.startswith("VLLM"):
        print(f"VLLM model detected, loading from: {model_info}")
        model = LLM(model_info)
        return None, model, model_info

    raise FileNotFoundError("Model path does not exist")

def run_hf_model(prompts, tokenizer, model):
    if not callable(tokenizer):
        raise TypeError("Tokenizer is not callable. Ensure it is correctly initialized.")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:7")

    # 调用model.generate，不传递任何额外参数
    outputs = model.generate(**inputs)

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
    outputs = model.generate(prompts)
    
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

if __name__ == "__main__":
    def get_response(prompts, model_name):
        tokenizer, model, model_info = load_model(model_name)

        if model_name.startswith("VLLM"):
            responses = run_vllm_model(prompts, model)
        elif model_name == "OPENAI":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=openai.api_key)
            responses = run_openai_model(prompts, model_name, client)
        else:
            responses = run_hf_model(prompts, tokenizer, model)
        
        return responses

    fire.Fire(get_response)
