import argparse
import json

from concurrent.futures import ThreadPoolExecutor, as_completed

from fireworks.client import Fireworks
from openai import OpenAI
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm

from prompts import create_examples, answer_via_ensemble, get_prompt_for_not_diamond

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file-location", type=str, required=True, help="The location to output the training data.")
    parser.add_argument("--train-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the training data file.")
    parser.add_argument("--dev-data-file-location", type=str, default="data/questions/medqa_4_options_dev.jsonl", help="The location of the dev data file.")
    parser.add_argument("--gpt-cot-file-location", type=str, default="data/output/gpt-4o_training_cots.csv", help="The location of the gpt generated chain of thoughts.")
    parser.add_argument("--llama-cot-file-location", type=str, default="data/output/llama_training_cots.csv", help="The location of the llama generated chain of thoughts.")
    parser.add_argument("--similar_questions-file-location", type=str, default="data/output/similar_training_questions_to_dev.csv", help="The location of the file mapping questions in the dev set to similar training questions.")
    parser.add_argument("--model-id", type=str, choices=["gpt-4o-2024-08-06", "accounts/fireworks/models/llama-v3p1-405b-instruct"], help="The id of the model to use.")
    parser.add_argument("--client", type=str, choices=["openai", "fireworks"], help="The llm provider to use.")
    parser.add_argument("--use-cot", type=bool, default=False, help="Whether to use cot in the response.")

    args, _ = parser.parse_known_args()

    if args.client == "fireworks":
        client = Fireworks()
    elif args.client == "openai":
        client = OpenAI()

    # Only generate responses for training questions that are similar to test questions
    similar_training_questions = pd.read_csv(args.similar_questions_file_location)
    gpt_cot_df = pd.read_csv(args.gpt_cot_file_location)
    llama_cot_df = pd.read_csv(args.llama_cot_file_location)
    df = pd.read_json(args.dev_data_file_location, lines=True)
    train_df = pd.read_json(args.train_data_file_location, lines=True)

    temp = 1.0
    index_values = []
    prompts = []
    predicted_answers = []
    scores = []

    def process_row(row):
        similar_train_rows = similar_training_questions[similar_training_questions['test_idx'] == row.name]
        similar_train_rows = similar_train_rows['train_idx'].tolist()

        examples = create_examples(similar_train_rows, train_df, gpt_cot_df, llama_cot_df)
        votes_by_answer_idx = answer_via_ensemble(
            client,
            args.model_id,
            row['question'],
            row['options'],
            examples,
            temp,
            cot=args.use_cot,
            num_models=5
        )

        prompt = get_prompt_for_not_diamond(row['question'], row['options'], examples)
        return prompt, votes_by_answer_idx, row['answer_idx']


    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_row, row): i for i, row in df.iterrows() if i not in index_values}

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            row_index = future_to_row[future]
            try:
                prompt, votes, answer = future.result()
                index_values.append(row_index)
                prompts.append(prompt)
                predicted_answers.append(max(votes, key=votes.get))
                scores.append(votes[answer] / sum(votes.values()))
            except Exception as exc:
                print(f'Row {row_index} generated an exception: {exc}')

    print("Evaluation complete.")

    results = pd.DataFrame({
        'idx': index_values,
        'prompt': prompts,
        'predicted_answer': predicted_answers,
        'score': scores,
    })
    results.to_csv(args.output_file_location, index=False)
