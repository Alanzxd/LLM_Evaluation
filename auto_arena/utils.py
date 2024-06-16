import torch
import json
import sys

if torch.cuda.is_available():
    print(f"Total CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")

existing_model_paths = {
    'gpt4-1106' : "OPENAI", 
    'gpt3.5-turbo-0125' : "OPENAI", 

    "mistral-7b-instruct-2" : "/data/shared/huggingface/hub/mistral-inst-7B-v0.2", 

    "llama2-13b-chat" : "/data/shared/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8", 
    "llama3-8b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/a8977699a3d0820e80129fb3c93c20fbd9972c41", 

    "qwen1.5-14b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec", 
    "qwen1.5-32b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",

    "openchat-3.5" : "/data/shared/huggingface/hub/models--openchat--openchat_3.5/snapshots/c8ac81548666d3f8742b00048cbd42f48513ba62",

    "vicuna-13b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2", 
    "vicuna-33b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4"
}

gt_scores = {
    'gpt4-1106' : 1251,
    'gpt3.5-turbo-0125' : 1103, 

    "mistral-7b-instruct-2" : 1072, 

    "llama2-13b-chat" : 1063, 
    "llama3-8b-instruct" : 1153, 

    "qwen1.5-14b-chat" : 1108, 
    "qwen1.5-32b-chat" : 1126,

    "openchat-3.5" : 1076,

    "vicuna-13b" : 1041, 
    "vicuna-33b" : 1090
}