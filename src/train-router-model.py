import argparse

import pandas as pd
from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit import CustomRouter
from notdiamond import NotDiamond

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-no-cot-file-location", type=str, default="data/generated/gpt-4o-no-cot_router_training_data.csv", help="The location of the gpt results on the dev set without cot.")
    parser.add_argument("--llama-no-cot-file-location", type=str, default="data/generated/llama-no-cot_router_training_data.csv", help="The location of the llama results on the dev set without cot.")
    args, _ = parser.parse_known_args()

    gpt_no_cot_df = pd.read_csv(args.gpt_no_cot_file_location)
    llama_no_cot_df = pd.read_csv(args.llama_no_cot_file_location)

    dataframes = [gpt_no_cot_df, llama_no_cot_df]

    index_intersection = set.intersection(*[set(df['idx']) for df in dataframes])

    # gpt_df = gpt_df[gpt_df['idx'].isin(index_intersection)]
    gpt_no_cot_df = gpt_no_cot_df[gpt_no_cot_df['idx'].isin(index_intersection)]

    # llama_df = llama_df[llama_df['idx'].isin(index_intersection)]
    llama_no_cot_df = llama_no_cot_df[llama_no_cot_df['idx'].isin(index_intersection)]

    gpt_no_cot_config = LLMConfig(
        provider="custom",
        model="gpt-4o-no-cot",
        is_custom=True,
        context_length=128000
    )

    llama_no_cot_config = LLMConfig(
        provider="custom",
        model="llama-405b-no-cot",
        is_custom=True,
        context_length=128000
    )

    provider_dict = {
        # 'openai/gpt-4o': gpt_df,
        gpt_no_cot_config: gpt_no_cot_df,
        # 'replicate/meta-llama-3.1-405b-instruct': llama_df,
        llama_no_cot_config: llama_no_cot_df
    }

    # Initialize the CustomRouter object for training
    trainer = CustomRouter(
        language="english",
        maximize=True
    )

    preference_id = trainer.fit(
        dataset=provider_dict,  # The dataset containing inputs, responses, and scores
        prompt_column="prompt",  # Column name for the input prompts
        response_column="predicted_answer",  # Column name for the model responses
        score_column="score"  # Column name for the scores
    )

    print("Custom router preference ID: ", preference_id)
