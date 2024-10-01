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

EurekaPrompt is an advanced medical prompting framework designed to enhance the accuracy and reasoning of LLMs in medical tasks. Building upon frameworks like MedPrompt and MedPrompt+, EurekaPrompt introduces new techniques such as LLM as a Judge and LLM Routing. These innovations allow for better selection of the strongest chain of thought reasoning and model-specific optimizations, ultimately leading to a 93.7% performance score on the USMLE, compared to the 91.3% achieved when running MedPrompt on comparable models.

This repository provides the scripts and workflows to replicate EurekaPrompt's training and evaluation process.

## Installation

Follow the steps below to configure your environment:

1. **Install Python 3.10 or above:**

2. **Set Up API Accounts:**
   The scripts in this repu use APIs from the following providers:
   - **OpenAI:** [Sign up here](https://platform.openai.com/)
   - **Fireworks:** [Sign up here](https://fireworks.ai/)
   - **NotDiamond:** [Sign up here](https://www.notdiamond.ai/)

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

This step involves finding similar questions in the training set for each question in both the test set and the dev set. This process is crucial for the in-context learning aspect of the framework. The script uses OpenAI's `text-embedding-3-large` embedding model to calculate similarity.

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

