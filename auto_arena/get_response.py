import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn.parallel import DistributedDataParallel 
import openai

from utils import existing_model_paths

def load_model(model_name):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    if model_info == "OPENAI":
        # print("OpenAI model detected, calling API.")
        return None, None  

    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, trust_remote_code=True)
        print("Tokenizer Loaded; Loading Model")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, device_map = 'auto', torch_dtype="auto")
        return tokenizer, model

    raise FileNotFoundError("Model path does not exist")

def run_hf_model(prompts, tokenizer, model, temperature=0.7, max_tokens=1024):
    if not callable(tokenizer):
        raise TypeError("Tokenizer is not callable. Ensure it is correctly initialized.")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:7")

    outputs = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_tokens,
        do_sample= (temperature >= 1e-4)
    )

    # VLLM 

    # Decoding all responses
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
    
    # full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # prompt_end_idx = full_response.find(prompt) + len(prompt)
    # if prompt_end_idx > -1 and prompt_end_idx < len(full_response):
    #     response = full_response[prompt_end_idx:].strip()
    # else:
    #     response = full_response

    # return response

def run_openai_model(prompts, model_name, client, temperature=0.7, max_tokens=1024):
    if "3.5-turbo-0125" in model_name: 
        model_name = "gpt-3.5-turbo-0125"
    elif "4-1106" in model_name: 
        model_name = "gpt-4-1106-preview"
    responses = []
    for prompt in prompts: 
        completion = client.chat.completions.create(
        model=model_name,
        messages=[
            # {"role": "system", "content": "You are an impartial evaluator."},
            {"role": "user", "content": prompt}
        ]
        )
        text = completion.choices[0].message.content
        responses.append(str(text))
    return responses


# def get_response(prompt, tokenizer, model, temperature=0.7, max_tokens=1024):
    
#     if tokenizer and model:
#         response = run_hf_model(prompt, model, tokenizer, model, temperature, max_tokens)

#     return str(response)

if __name__ == "__main__":
    fire.Fire(get_response)