import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed

from fireworks.client import Fireworks
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from prompts import answer_usmle_question, llm_as_a_judge


def generate_high_quality_reasoning(row, client, model_id):
    actual_answer = row['answer_idx']
    temp = 0.10
    increase = 0.075

    best_score = 0
    best_cot = ''

    while temp <= 1.0:
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
                temp += increase
                continue

            judge_res = llm_as_a_judge(
                client,
                row['question'],
                row['options'],
                row['answer_idx'],
                res['reasoning']
            )
            score = judge_res['score']

            if score >= 9:
                return res['reasoning'], score
            elif score > best_score:
                best_score = score
                best_cot = res['reasoning']

            temp += increase

        except Exception as exc:
            print(f'Row {row.name} generated an exception: {exc}')
            return None, None

    return best_cot, best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file-location", type=str, required=True, help="The location to output the chain of thought results.")
    parser.add_argument("--train-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the training data file.")
    parser.add_argument("--similar_questions-file-location", type=str, default="data/generated/similar_training_questions.csv", help="The location of the file mapping questions in the dev set to similar training questions.")
    parser.add_argument("--model-id", type=str, choices=["gpt-4o-2024-08-06", "accounts/fireworks/models/llama-v3p1-405b-instruct"], help="The id of the model to use.")
    parser.add_argument("--client", type=str, choices=["openai", "fireworks"], help="The llm provider to use.")
    args, _ = parser.parse_known_args()

    if args.client == "fireworks":
        client = Fireworks()
    else:
        client = OpenAI()

    similar_training_questions = pd.read_csv(args.similar_questions_file_location)
    training_indices = set(similar_training_questions['train_idx'].unique())
    training_indices = list(training_indices)
    df = pd.read_json(args.train_data_file_location, lines=True).iloc[training_indices]

    training_cots = dict()
    llm_judge_scores = dict()

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_row = {executor.submit(generate_high_quality_reasoning, row, client, args.model_id): i for i, row in df.iterrows()}

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            row_index = future_to_row[future]
            reasoning, score = future.result()
            if reasoning is not None:
                training_cots[row_index] = reasoning
                llm_judge_scores[row_index] = score

    print(f'Total explanations generated: {len(training_cots)}')

    results = pd.DataFrame({
        'train_idx': training_cots.keys(),
        'cot': training_cots.values(),
        'score': llm_judge_scores.values()
    })
    results.to_csv(args.output_file_location, index=False)
