import argparse
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

from prompts import answer_usmle_question


def generate_high_quality_reasoning(row, client, temp, num_reasoning_paths, model_id):
    num_trials = 4 * num_reasoning_paths
    backoff_time = 1
    actual_answer = row['answer_idx']
    correct_reasoning_paths = list()

    for i in range(num_trials):
        try:
            res = answer_usmle_question(
                client,
                row['question'],
                row['options'],
                [],
                temp=temp,
                model_id=model_id,
                cot=True
            )

            predicted_answer = res['answer']

            if predicted_answer != actual_answer:
                continue

            correct_reasoning_paths.append(res['reasoning'])
            if len(correct_reasoning_paths) >= num_reasoning_paths:
                break

        except Exception as exc:
            print(f'Row {row.name} generated an exception: {exc}')
            if "token rate limit" in str(exc):
                time.sleep(backoff_time)
                backoff_time *= 2
                backoff_time = min(backoff_time, 60)
            continue

    return correct_reasoning_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--similar_questions-file-location", type=str, default="data/generated/similar_training_questions.csv", help="The training questions that are similar to the test questions.")
    parser.add_argument("--output-file-location", type=str, default="data/generated/candidate_reasoning_paths.csv", help="The location to output the candidate reasoning paths.")
    parser.add_argument("--train-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the training data file.")
    parser.add_argument("--model-id", type=str, default="gpt-4o", help="The model to use.")
    parser.add_argument("--temp", type=float, default=0.7, help="The temperature to use.")
    parser.add_argument("--num-reasoning-paths", type=int, default=4, help="The number of reasoning paths to explore per question.")
    parser.add_argument("--max-workers", type=int, default=6, help="The number of threads to submit api requests.")
    args, _ = parser.parse_known_args()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    )

    similar_training_questions = pd.read_csv(args.similar_questions_file_location)
    training_indices = list(similar_training_questions['train_idx'].unique())
    df = pd.read_json(args.train_data_file_location, lines=True).iloc[training_indices]

    training_cots = list()
    cot_indices = list()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_row = {executor.submit(generate_high_quality_reasoning, row, client, args.temp, args.num_reasoning_paths, args.model_id): i for i, row in df.iterrows()}

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            row_index = future_to_row[future]
            reasonings = future.result()
            for reasoning in reasonings:
                training_cots.append(reasoning)
                cot_indices.append(row_index)

    print(f'Total explanations generated: {len(training_cots)}')

    results = pd.DataFrame({
        'train_idx': cot_indices,
        'cot': training_cots
    })
    results.to_csv(args.output_file_location, index=False)
