# EurekaPrompt

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Running Scripts](#running-scripts)
  - [Calculate Similar Training Questions](#calculate-similar-training-questions)
  - [Generate Chain of Thought Training Data](#generate-chain-of-thought-training-data)
  - [Generate Router Training Data](#generate-router-training-data)
  - [Train the Router Model](#train-the-router-model)
  - [Call the Router Model](#call-the-router-model)
  - [Calculate USMLE Accuracy](#calculate-usmle-accuracy)

## Overview

EurekaPrompt is an advanced medical prompting framework designed to enhance the accuracy and reasoning of LLMs in medical tasks. Building upon frameworks like MedPrompt, EurekaPrompt introduces new techniques such as LLM as a Judge and LLM Routing. These innovations allow for better selection of the strongest chain of thought reasoning and model-specific optimizations, ultimately leading to a 93.3% performance score on the USMLE.

This repository provides the scripts and workflows to replicate EurekaPrompt's training and evaluation process.

## Installation

Follow the steps below to configure your environment:

1. **Install Python 3.10 or above:**

2. **Set Up API Accounts:**
   The scripts in this repu use APIs from the following providers:
   - [OpenAI](https://platform.openai.com/)
   - [Fireworks](https://fireworks.ai/)
   - [NotDiamond](https://www.notdiamond.ai/)

4. **Set Up Environment Variables:**
   After creating accounts, obtain the API keys and set the following environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export FIREWORKS_API_KEY="your-fireworks-api-key"
   export NOTDIAMOND_API_KEY="your-notdiamond-api-key"
   ```

5. **Install Dependencies:**
   Use `pip` to install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Scripts

### Calculate Similar Training Questions

This script finds similar questions in the training set for each question in both the test set and the dev set. This process is crucial for the in-context learning aspect of the framework. The script uses OpenAI's `text-embedding-3-large` embedding model to calculate similarity.

#### To Run

1. For the dev set:

The script is run twice: once for the dev set and once for the test set.

```bash
python src/calculate-similar-training-questions.py --output-file-location data/output/similar_training_questions_to_dev.csv --test-data-file-location data/questions/medqa_4_options_dev.jsonl
```

2. For the test set

```bash
python src/calculate-similar-training-questions.py --output-file-location data/output/similar_training_questions_to_test.csv --test-data-file-location data/questions/medqa_4_options_test.jsonl
```

### Next Section

Based on the script you've provided, here's the updated section for the "Generate Chain of Thought Training Data" subsection in the README. Since the script is called twice (once for GPT and once for Llama), I'll include both examples.

---

### Generate Chain of Thought Training Data

This script generates the chain of thought (CoT) reasoning for each training question. The process involves querying multiple temperatures for each question using two models: GPT-4o and Llama 3.1 405b Instruct. An LLM judge (implemented with GPT-4o) evaluates the responses and selects the highest-quality reasoning for use in the training data.

#### To Run

1. Run the script for GPT-4o:

```bash
python src/generate-training-chain-of-thoughts.py --output-file-location data/generated/gpt-4o_training_cots.csv  --similar_questions-file-location data/output/similar_training_question_to_test.csv --model-id gpt-4o-2024-08-06 --client openai
```

2. Run the script for Llama:

```bash
python src/generate-training-chain-of-thoughts.py --output-file-location data/generated/llama_training_cots.csv  --similar_questions-file-location data/output/similar_training_question_to_tests.csv --model-id accounts/fireworks/models/llama-v3p1-405b-instruct --client fireworks
```

The output files (`gpt-4o_training_cots.csv` and `llama_training_cots.csv`) will contain chain of thought reasoning and scores for each training question. These files will be used in later steps to provide the model with in-context learning.

### Generate Router Training Data

Based on the script you've provided, here is the updated "Generate Router Training Data" subsection for the README. This script generates the router training data by using similar training questions and ensemble model predictions.

---

### Generate Router Training Data

This script generates the training data for the routing model by leveraging both GPT and Llama models, both with and without CoT reasoning. The training data includes ensemble-based predictions from similar training questions, using in-context learning to generate router prompts and scores for each question in the dev set.

#### To Run

1. Run the script for GPT with CoT:

```bash
python src/generate-router-training-data.py --output-file-location data/generated/gpt-4o_router_training_data.csv  --model-id gpt-4o-2024-08-06 --client openai --use-cot True
```

2. Run the script for GPT without CoT:

```bash
python src/generate-router-training-data.py --output-file-location data/generated/gpt-4o_router_training_data.csv  --model-id gpt-4o-2024-08-06 --client openai --use-cot False 
```

3. Run the script for Llama with CoT:

```bash
python src/generate-router-training-data.py --output-file-location data/generated/llama_router_training_data.csv  --model-id ccounts/fireworks/models/llama-v3p1-405b-instruct --client fireworks --use-cot True 
```

4. Run the script for Llama without CoT:

```bash
python src/generate-router-training-data.py --output-file-location data/generated/llama_router_training_data.csv  --model-id ccounts/fireworks/models/llama-v3p1-405b-instruct --client fireworks --use-cot False
```

The output files (`gpt-4o_router_training_data.cs` and `llama_router_training_data.cs`) will contain prompts, predicted answers, and scores for each development question. These files will be used in training the router model.

Here is the completed **Train the Router Model** section for your README based on the provided script.

---

### Train the Router Model

This script trains the routing model to determine which model (GPT or Llama, with or without CoT) provides the best response for each question. The router is trained on results from the dev set, where each LLM response is generated using choice shuffling. The trained router is used to select the most suitable model for future predictions based on prompt and model performance.

#### To Run

```bash
python src/train-router.py --gpt-file-location data/generated/gpt-4o-no-cot_router_training_data.csv --gpt-no-cot-file-location data/generated/gpt-4o_router_training_data.csv --llama-file-location data/generated/llama-no-cot_router_training_data.csv --llama-no-cot-file-location data/generated/llama_router_training_data.csv
```

The router model will be trained on the given dataset, using the input prompts, predicted responses, and model performance scores. Upon completion, a preference ID will be printed, which can be used to reference the router for subsequent inference tasks.

The following datasets are used:

- **GPT with CoT**: Results generated by GPT with chain-of-thought reasoning.
- **GPT without CoT**: Results generated by GPT without chain-of-thought reasoning.
- **Llama with CoT**: Results generated by Llama with chain-of-thought reasoning.
- **Llama without CoT**: Results generated by Llama without chain-of-thought reasoning.

The router uses this data to learn how to select the best model for future tasks.

### Call the Router Model

After training the router, you can use it to select the most appropriate model for each test question by comparing the performance of the available models. The router will choose whether to use GPT or Llama and whether to include CoT reasoning for each specific input.

#### To Run

```bash
python src/call-router.py --preference-id <YOUR_PREFERENCE_ID> --output-file-location data/generated/router_preferences.csv
```

Make sure to replace `<YOUR_PREFERENCE_ID>` with the actual preference ID obtained after training the router.

The results, including the model choice and whether chain-of-thought reasoning was used, will be saved to the specified output file.

Here is the completed **Calculate USMLE Accuracy** section for your README based on the provided code:

---

### Calculate USMLE Accuracy

Once you have the router preferences generated and stored, the final step is to evaluate the accuracy of the models on the USMLE test set. The questions evaluated are those in the [MedQA 4-options dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

#### To Run

```bash
python src/calculate-usmle-accuracy.py
```
