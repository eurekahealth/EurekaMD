import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

from prompts import answer_via_ensemble, create_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the test set to evaluate.")
    parser.add_argument("--test-data-file-location", type=str, default="data/questions/medqa_4_options_test.jsonl", help="The location of the test set to evaluate.")
    parser.add_argument("--similar-questions-file-location", type=str, default="data/generated/similar_training_questions_to_test.csv", help="The location of the file mapping training questions to similar test questions.")
    parser.add_argument("--cot-file-location", type=str, default="data/generated/best_reasoning_paths.csv", help="The location of the best reasoning paths selected by the LLM judge.")
    parser.add_argument("--temp", type=float, default=1.0, help="The temperature to use.")
    parser.add_argument("--model-id", type=str, default="gpt-4o", help="The model to use.")
    parser.add_argument("--num-examples", type=int, default=5, help="The number of examples to inject for in-context learning.")
    parser.add_argument("--num-shuffles", type=int, default=8, help="The number of shuffles to perform as part of choice shuffling.")
    parser.add_argument("--max-workers", type=int, default=6, help="The number of threads to submit api requests.")
    args, _ = parser.parse_known_args()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    )

    train_df = pd.read_json(args.train_data_file_location, lines=True)
    df = pd.read_json(args.test_data_file_location, lines=True)

    similar_training_questions = pd.read_csv(args.similar_questions_file_location)
    cot_df = pd.read_csv(args.cot_file_location).fillna('')

    predicted_answers = []
    actual_answers = []
    index_values = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_row = dict()
        for test_idx, row in df.iterrows():
            similar_train_rows = similar_training_questions[similar_training_questions['test_idx'] == row.name]

            examples = create_examples(
                similar_train_rows['train_idx'].tolist(),
                train_df,
                cot_df,
                max_examples=args.num_examples
            )

            future = executor.submit(
                answer_via_ensemble,
                client,
                args.model_id,
                row['question'],
                row['options'],
                examples,
                args.temp,
                num_models=args.num_shuffles,
                cot=True
            )
            future_to_row[future] = row

        for future in tqdm(as_completed(future_to_row), total=len(df)):
            row = future_to_row[future]
            try:
                votes_by_answer_idx = future.result()
                num_votes = sum(votes_by_answer_idx.values())
                if num_votes < args.num_shuffles:
                    print(f'Row {row.name} only has {num_votes} votes')
                    continue

                sorted_votes_by_answer = sorted(votes_by_answer_idx.items())
                predicted_answer = max(sorted_votes_by_answer, key=lambda item: item[1])[0]

                predicted_answers.append(predicted_answer)
                actual_answers.append(row['answer_idx'])
                index_values.append(row.name)
            except Exception as exc:
                print(f'Row {row.name} generated an exception: {exc}')

    print("Evaluation complete.")

    num_correct = sum([predicted_answers[i] == actual_answers[i] for i in range(len(predicted_answers))])
    pct_correct = num_correct / len(predicted_answers)
    print(f'Pct Correct,{round(pct_correct, 4)}')
    print(f'Num Correct,{num_correct}')
    print(f'Total,{len(predicted_answers)}')
