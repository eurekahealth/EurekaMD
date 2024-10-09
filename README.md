# EurekaPrompt

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Running Scripts](#running-scripts)
  - [Determine Similar Training Questions](#determine-similar-training-questions)
  - [Generate Candidate Reasoning Paths](#generate-candidate-reasoning-paths)
  - [Select Best Reasoning Paths](#select-best-reasoning-paths)
  - [Calculate USMLE Accuracy](#calculate-usmle-accuracy)

## Overview

EurekaPrompt is an advanced medical prompting framework designed to enhance the accuracy and reasoning of LLMs in medical tasks. Building upon the MedPrompt framework, EurekaPrompt uses an LLM Judge to provide better selection of the strongest chain of thought reasoning, ultimately leading to a 93.3% performance score on the USMLE.

This repository provides the scripts and workflows to replicate EurekaPrompt's training and evaluation process.

## Installation

Follow the steps below to configure your environment:

1. **Install Python 3.10 or above:**

2. **Deploy gpt-4o via the Azure OpenAI Service:**
   The scripts in this repo use the [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview). Follow [these instructions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource) to deploy gpt-4o through the Azure OpenAI Service.

4. **Set Up Environment Variables:**
   After configuring the Azure OpenAI Service, set the following environment variables:
   ```bash
   export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
   export AZURE_OPENAI_ENDPOINT_URL="your-azure-openai-endpoint-url"
   ```

5. **Install Dependencies:**
   Use `pip` to install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Scripts

### Determine Similar Training Questions

This script finds similar questions in the training set for each question in the test set. The script uses OpenAI's `text-embedding-3-large` embedding model to calculate similarity.

#### To Run

```bash
python src/determine-similar-training-questions.py
```

---

### Generate Candidate Reasoning Paths

This script generates multiple reasoning paths for each question in the training set. The output file will contain multiple reasoning paths for each question in the training set.

#### To Run

```bash
python src/generate-candidate-reasoning-paths.py
```

---

### Select Best Reasoning Paths

This script uses an LLM Judge to select the best reasoning path for each question in the training set.

#### To Run

```bash
python src/select-best-reasoning-paths.py
```

---

### Calculate USMLE Accuracy

The final script evaluates the accuracy of EurekaPrompt on the USMLE test set, using the reasoning paths selected by the previous script as the paths to use for the few-shot learning. The questions evaluated are those in the [MedQA 4-options dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

#### To Run

```bash
python src/calculate-usmle-accuracy.py
```
