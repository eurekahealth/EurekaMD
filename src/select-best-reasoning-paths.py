import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm

from prompts import llm_judge_reasoning_ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the test set to evaluate.")
    parser.add_argument("--reasoning-paths-file-location", type=str, default="data/generated/data/generated/candidate_reasoning_paths.csv", help="The location of the candidate reasoning paths.")
    parser.add_argument("--output-file-location", type=str, default="data/generated/best_reasoning_paths.csv", help="The location to output the selected reasoning paths.")
    parser.add_argument("--model-id", type=str, default="gpt-4o", help="The model id to use as the judge.")
    parser.add_argument("--temp", type=float, default=1.0, help="The temperature to use.")
    parser.add_argument("--num-shuffles", type=int, default=12, help="The number of shuffles to perform as part of choice shuffling.")
    parser.add_argument("--max-workers", type=int, default=6, help="The number of threads to submit api requests.")
    args, _ = parser.parse_known_args()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    )

    df = pd.read_json(args.train_data_file_location, lines=True)
    cot_df = pd.read_csv(args.gpt_cot_file_location)

    future_to_index = dict()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for train_idx in cot_df['train_idx'].unique():
            row = df.iloc[train_idx]
            question = row['question']
            answer = row['answer_idx']
            choices = row['options']

            reasoning_rows = cot_df[cot_df['train_idx'] == train_idx]
            reasoning_paths_dict = dict()
            for i, (_, reasoning_row) in enumerate(reasoning_rows.iterrows()):
                reasoning_paths_dict[i + 1] = reasoning_row['cot']

            future = executor.submit(
                llm_judge_reasoning_ensemble,
                client,
                question,
                answer,
                choices,
                reasoning_paths_dict,
                args.num_shuffles,
                args.temp,
                args.model_id
            )
            future_to_index[future] = train_idx

        best_paths_dict = dict()
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            train_idx = future_to_index[future]
            reasoning = future.result()
            best_paths_dict[train_idx] = reasoning

    res_df = pd.DataFrame(list(best_paths_dict.items()), columns=['train_idx', 'cot'])
    res_df.to_csv(args.output_file_location, index=False)

