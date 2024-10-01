import os
import pickle

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


class VectorDB:
    def __init__(self, client, db_path="./data/generated/vector_db.pkl"):
        self.client = client
        self.embeddings = []
        self.questions = []
        self.indices = []
        self.db_path = db_path

    def load_data(self):
        if self.embeddings and self.questions and self.indices:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self._load_db()
            return

    def _load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)

        self.embeddings = data["embeddings"]
        self.questions = data["questions"]
        self.indices = data["indices"]

    def save_db(self, data, indices):
        self._embed_and_store(data, indices)
        data = {
            "embeddings": self.embeddings,
            "questions": self.questions,
            "indices": self.indices
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)
        print("Vector database loaded and saved.")

    def _embed_and_store(self, texts, indices):
        def generate_embedding(text):
            return self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            ).data[0].embedding

        with ThreadPoolExecutor(max_workers=32) as executor:
            # Submit all tasks and store futures with their original indices
            future_to_index = {executor.submit(generate_embedding, text): i for i, text in enumerate(texts)}

            # Initialize embeddings list with None values
            self.embeddings = [None] * len(texts)

            # As futures complete, store results in the correct position
            for future in tqdm(as_completed(future_to_index), total=len(texts)):
                index = future_to_index[future]
                self.embeddings[index] = future.result()

        self.questions = texts
        self.indices = indices

    def search(self, query, k=5, similarity_threshold=0.75):
        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "question": self.questions[idx],
                    "similarity": similarities[idx],
                    "index": self.indices[idx]
                }
                top_examples.append(example)

                if len(top_examples) >= k:
                    break

        return top_examples