import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from vector_db import VectorDB


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-file-location", type=str, default="data/questions/medqa_4_options_train.jsonl", help="The location of the training data file.")
    parser.add_argument("--test-data-file-location", type=str, default="data/questions/medqa_4_options_test.jsonl", help="The location of the test data file.")
    parser.add_argument("--output-file-location", type=str, default="data/generated/similar_training_questions.csv", help="The location to output the similar indices.")
    parser.add_argument("--num_similar_questions", type=int, default=5, help="The number of similar training questions to store for each test question.")
    args, _ = parser.parse_known_args()

    client = OpenAI()
    db = VectorDB(client)

    train_df = pd.read_json(args.training_data_file_location, lines=True)
    train_questions = [q for q in train_df['question']]
    train_indices = train_df.index.tolist()
    db.save_db(train_questions, train_indices)
    db.load_data()

    test_df = pd.read_json(args.test_data_file_location, lines=True)
    sim_idx_list = [None] * len(test_df)

    def search_db(index, k=5):
        results = db.search(test_df.iloc[index]['question'], k=k, similarity_threshold=0)
        return index, [r['index'] for r in results]

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(search_db, i) for i in range(len(test_df))]

        # Collect results as they complete, showing progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            index, result = future.result()
            sim_idx_list[index] = result

    rows = []
    for test_idx, inner_list in enumerate(sim_idx_list):
        for train_idx in inner_list:
            rows.append({'test_idx': test_idx, 'train_idx': train_idx})

    df = pd.DataFrame(rows)
    df.to_csv(args.output_file_location, index=False)


