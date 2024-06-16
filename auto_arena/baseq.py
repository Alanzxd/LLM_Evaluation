import torch
import os
import json
from openai import OpenAI

from get_response import run_hf_model, load_model, run_openai_model
from utils import existing_model_paths

model_names = list(existing_model_paths.keys())

def base_answers(model_name): 
    tokenizer, model = load_model(model_name) # TRY HF
    if not judge_model: # OPENAI
        client = OpenAI(api_key="sk-proj-tJPuS2rvAEYubMXSCxfCT3BlbkFJHXnkL3PMGmNhTiMJk02V")

    # TODO

    if judge_model != None: #HF
        judge_responses = run_hf_model(prompts, tokenizer, judge_model)
        swapped_judge_responses = run_hf_model(swapped_prompts, tokenizer, judge_model)
    else: #OPENAI
        judge_responses = run_openai_model(prompts, judge_name, client)
    