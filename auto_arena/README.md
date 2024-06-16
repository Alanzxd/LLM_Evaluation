
## Pipeline
Start with a set of base questions and models. Collect responses from the models to the base questions. [Currently: 80 mt-bench questions, using legacy responses from Zhen.]

For a judge, randomly select a question_id and a pair of candidates. Then, check if this pair has been evaluated before on this question_id in the voting_records. If not, obtain the model's judgment. [Currently: first check Zhen's legacy judgments, and use them directly if available.]

Each judge model can be run concurrently using `python judge_responses.py --judge_name= --n_times=`, and all the records are stored in `judgements/{model_name}`, which includes: voting_records (pairs and questions that have been evaluated so far), judgments (actual judgment responses), and transition_vector (the judge's vote, updated in real-time).

To compile the final transition matrix, execute `python matrix_handling.py`. This script concatenates all the transition vectors together and saves them in the `transition_matrices` folder.

After obtaining the transition matrix, run `python pagerank.py --file_path=` to calculate the PageRank score, which outputs a ranking table and correlation.

## TODO
- Discuss the logic for adding a new model.
- Implement `baseq.py` to make the new model answer all base questions. Currently, this isn't implemented as we are using Zhen's legacy responses.
- Explore alternative algorithms.
- Investigate potential nuanced implementations that could accelerate the pipeline.

## To run judgement: 
python judge_responses.py --judge_name= --n_times= 

## To compile the newest transition matrix
python matrix_handling.py

## To see the pagerank ranks
python pagerank.py --file_path=


# SCRIPTS
*get_model_response.py* 
- to get a specific model's response to the specified prompt

*matrix_handling.py* 
- contains functions that update transition matrices and voting records
- initialize by directly running this script

*judge_responses.py* 
- main script for the creation of transition matrix
- to get a judge model's evaluation for two randomly sampled models to the same prompt
- calls *get_model_response.py*

*pagerank.py* 
- to calculate the pagerank score given the model evaluations 

*utils.py* 
- check for available cuda devices; existing supported model names and paths
- if any new models, please update model names and paths in this file

# RESULTS 
*voting_records.jsonl*
- contains all judged pairs
- look up judgement specifics using the data_id in judgements.jsonl

*judgements.jsonl*
- contains all judgements made 


# TODO
add swap verification

*answer_baseq.py* 
- ask models to answer the base questions
- calls *get_model_response.py*
- Now using the legacy responses collected by Zhen

*add_new_model.py*
- handles everything involved in adding a new model
- needs discussion