import os
import json
import random
from tqdm import tqdm
import fire
import uuid
import torch
from openai import OpenAI

from matrix_handling import search_voting_records, update_voting_records, update_transition_vector
from get_response import run_hf_model, load_model, run_openai_model
from utils import existing_model_paths


def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_question(prompt_id):
    questions = load_jsonl("mt_bench_questions.jsonl")
    for question in questions:
        if question['question_id'] == prompt_id:
            return question['turns'][0]
    return None

def fetch_responses(model_a, model_b, prompt_id):
    directory = f"model_responses/mt_bench_question_{prompt_id}"
    response_a = None
    response_b = None
    for filename in os.listdir(directory):
        if model_a in filename and response_a == None:
            with open(os.path.join(directory, filename), 'r') as f:
                response_a = json.load(f)['response'][0]
        elif model_b in filename and response_b == None:
            with open(os.path.join(directory, filename), 'r') as f:
                response_b = json.load(f)['response'][0]

    return response_a, response_b

def find_legacy_judgements(judge_name, model_a, model_b, question_id, directory="/data/shared/llm-bench/model_judgements"):
    """Search for legacy judgements ran by zhen that match the given criteria."""
    judge_directory = os.path.join(directory, judge_name)

    if not os.path.exists(judge_directory):
        print(f"No directory found for judge: {judge_name}")
        return model_a, model_b, None

    for filename in os.listdir(judge_directory):
        filepath = os.path.join(judge_directory, filename)
        judgements = load_jsonl(filepath)

        for judgement in judgements:
            if judgement.get('question_id') == question_id:
                if judgement.get('model_a') == model_a and judgement.get('model_b') == model_b:
                    return model_a, model_b, judgement
                elif judgement.get('model_a') == model_b and judgement.get('model_b') == model_a:
                    return model_b, model_a, judgement
    return model_a, model_b, None

def create_prompt(question, response_a, response_b):
    return ("[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\u2019s instructions and answers the user\u2019s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie. \n\n[User Question]" + question + "\n\n[The Start of Assistant A's Answer]\n" + response_a + "\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n" + response_b + "\n[The End of Assistant B's Answer]\n\n[The Evaluation and the Verdict]\n")

def determine_winner(judge_response, model_a, model_b):
    if "[[A]]" in judge_response:
        winner = model_a
    elif "[[B]]" in judge_response:
        winner = model_b
    else:
        if "[A]" in judge_response and "[B]" not in judge_response:
            winner = model_a
        elif "[B]" in judge_response and "[A]" not in judge_response:
            winner = model_b
        else:
            winner = None

def save_judgement(judge_name, data):
    """Save judgement data to a JSONL file."""
    path = f"judgements/{judge_name}/judgements.jsonl"
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def run_judging_trials(judge_name, n_batches = 100, batch_size = 64):

    # Load Judge Model
    model_names = list(existing_model_paths.keys())
    tokenizer, judge_model = load_model(judge_name) # TRY HF
    if not judge_model: # OPENAI
        client = OpenAI(api_key="sk-proj-tJPuS2rvAEYubMXSCxfCT3BlbkFJHXnkL3PMGmNhTiMJk02V")

    model_pairs = []
    prompt_ids = []
    questions = []
    legacy_judgements = []

    # Preparing batches
    print("Preparing Question Batches")
    for _ in tqdm(range(batch_size * n_batches)):
        model_a, model_b = random.sample([m for m in model_names if m != judge_name], 2)
        prompt_id = random.randint(81, 160) # TODO: need dynamic update for the question base
        if not search_voting_records(judge_name, model_a, model_b, prompt_id):
            model_a, model_b, legacy = find_legacy_judgements(judge_name, model_a, model_b, prompt_id)
            model_pairs.append((model_a, model_b))
            prompt_ids.append(prompt_id)
            questions.append(get_question(prompt_id))
            legacy_judgements.append(legacy)

    # Processing batches
    print("Processing Batches")
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        current_pairs = model_pairs[start_idx:end_idx]
        current_prompt_ids = prompt_ids[start_idx:end_idx]
        current_questions = questions[start_idx:end_idx]
        current_legacy_judgements = legacy_judgements[start_idx:end_idx]

        prompts = []
        swapped_prompts = []

        for j in range(batch_size):
            model_a, model_b = current_pairs[j]
            question = current_questions[j]
            legacy_judgement = current_legacy_judgements[j]
            if legacy_judgement and (
                "[[A]]" in legacy_judgement["judgement"]["judge_response"] or
                "[[B]]" in legacy_judgement["judgement"]["judge_response"] or
                "[[C]]" in legacy_judgement["judgement"]["judge_response"] or
                "[A]" in legacy_judgement["judgement"]["judge_response"] or
                "[B]" in legacy_judgement["judgement"]["judge_response"] or
                "[C]" in legacy_judgement["judgement"]["judge_response"]
            ):
                prompts.append(legacy_judgement["judgement"]["judge_prompt"])
                swapped_prompts.append(legacy_judgement["judgement"]["judge_prompt"])  # Placeholder for swapped
            else:
                response_a, response_b = fetch_responses(model_a, model_b, current_prompt_ids[j])
                prompt = create_prompt(question, response_a, response_b)
                swapped_prompt = create_prompt(question, response_b, response_a)  # Swapping responses
                prompts.append(prompt)
                swapped_prompts.append(swapped_prompt)

        # Running the model on both sets of prompts
        if judge_model != None: #HF
            judge_responses = run_hf_model(prompts, tokenizer, judge_model)
            swapped_judge_responses = run_hf_model(swapped_prompts, tokenizer, judge_model)
        else: #OPENAI
            judge_responses = run_openai_model(prompts, judge_name, client)

        # Process each response and compare
        for j, (response, swapped_response) in enumerate(zip(judge_responses, swapped_judge_responses)):
            model_a, model_b = current_pairs[j]
            winner = determine_winner(response, model_a, model_b)
            swapped_winner = determine_winner(swapped_response, model_b, model_a)

            if winner == swapped_winner:
                final_winner = winner  # Consistent result
            else:
                # Conduct a third judgement
                third_prompt = create_prompt(current_questions[j], response_a, response_b)  # Reusing the first prompt
                if judge_model != None: #HF
                    third_response = run_hf_model([third_prompt], tokenizer, judge_model)[0]
                else: #OPENAI
                    judge_responses = run_openai_model([third_prompt], judge_name, client)[0]
                third_winner = determine_winner(third_response, model_a, model_b)
                final_winner = winner if third_winner == winner else swapped_winner

            data_id = str(uuid.uuid4())
            update_voting_records(judge_name, model_a, model_b, current_prompt_ids[j], final_winner, data_id)
            if final_winner:
                update_transition_vector(judge_name, final_winner)

            judgement_data = {
                'data_id': data_id,
                'question_id': current_prompt_ids[j],
                'prompt': prompts[j],  # Record the initial prompt for consistency
                'judge_response': response
            }
            save_judgement(judge_name, judgement_data)


if __name__ == '__main__':
    fire.Fire(run_judging_trials)
