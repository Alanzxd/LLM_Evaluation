import os
import json

def fetch_responses(model_a, prompt_id):
    directory = f"model_responses/mt_bench_question_{prompt_id}"
    response_a = None
    for filename in os.listdir(directory):
        if model_a in filename and response_a == None:
            with open(os.path.join(directory, filename), 'r') as f:
                question_id = json.load(f)['question_id']
            with open(os.path.join(directory, filename), 'r') as f:
                response_a = json.load(f)['response'][0]
            with open(os.path.join(directory, filename), 'r') as f:
                question = json.load(f)['turns'][0]
    
    return response_a, question, question_id

cnt = 0
for i in range(81,120): 
    # model = "gpt4-1106"
    # model = "qwen1.5-32b-chat"
    # model = "vicuna-33b"
    # model = "mistral-7b-instruct-2"
    model = "llama3-8b-instruct"
    # model = "qwen1.5-14b-chat"
    
    a, q, qid = fetch_responses(model, i)
    if a == "": 
        cnt += 1
    print(qid)
    print(q)
    print("-"*30)
    print(a)
    print("="*50)
print(cnt)