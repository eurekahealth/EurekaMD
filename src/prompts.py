from collections import defaultdict
from dataclasses import dataclass

import json
import random
import re
import time


def llm_judge_reasoning(
    client,
    question,
    answer,
    choices,
    reasoning_paths_dict,
    temp=0.0,
    model_id='gpt-4o'
):
    llm_judge_guidelines = """- Clarity: Assess whether the reasoning is expressed in a clear, concise, and understandable manner. Each step should be easily followed and free from unnecessary complexity.
- Logical Consistency: Verify that the reasoning avoids contradictions or logical fallacies and maintains consistency throughout.
- Coherence: Evaluate the logical flow between steps. Each step should logically follow from the previous one, with no gaps in reasoning or abrupt transitions.
- Depth of Explanation: Determine if the reasoning provides a detailed and thorough explanation for each step, including relevant assumptions and justifications where necessary.
- Accuracy: Check if the reasoning leads to a correct or reasonable conclusion based on the problem context. Ensure that each step aligns with the facts and information provided."""

    system_msg = f"""You will be presented with several potential reasoning paths to answer a question from the United States Medical Licensing Examination. Your task is to select the best reasoning path.

Evaluation Criteria:
{llm_judge_guidelines}
"""

    choices_str = '\n'.join([f"{letter}. {c}" for letter, c in choices.items()])
    reasoning_paths_str = '\n\n'.join([f"{num}. {r}" for num, r in reasoning_paths_dict.items()])

    user_prompt = f"""### Question:
{question}

### Choices:
{choices_str}

### Answer:
{answer}

### Reasoning Paths:
{reasoning_paths_str}
"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scratchpad": {
                        "type": "string",
                        "description": "Your thoughts and analysis on which reasoning path is best.",
                    },
                    "best_path": {
                        "type": "string",
                        "description": "The best reasoning path.",
                        "enum": [str(i+1) for i in range(len(reasoning_paths_dict))]
                    }
                },
                "required": ["scratchpad", "best_path"],
                "additionalProperties": False
            }
        }
    }

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temp,
        seed=42,
        response_format=response_format,
        max_tokens=8192
    ).choices[0].message.content

    response_obj = json.loads(response)
    return int(response_obj['best_path'])


def llm_judge_reasoning_ensemble(
    client,
    question,
    answer,
    choices,
    reasoning_paths_dict,
    ensemble_size=5,
    temp=1.0,
    model_id='gpt-4o'
):
    backoff_time = 1
    votes_by_reasoning = defaultdict(int)

    max_tries = ensemble_size * 2
    for trial in range(max_tries):
        if sum(votes_by_reasoning.values()) == ensemble_size:
            break

        reasoning_order_list = random.sample(list(reasoning_paths_dict.values()), len(reasoning_paths_dict))
        reasoning_order_dict = {i + 1: r for i, r in enumerate(reasoning_order_list)}

        try:
            shuffled_best_path_idx = llm_judge_reasoning(
                client,
                question,
                answer,
                choices,
                reasoning_order_dict,
                temp,
                model_id
            )
        except Exception as exc:
            print(f'Exception generated {exc}')
            if "token rate limit" in str(exc):
                print(f'Retrying in {backoff_time} seconds')
                time.sleep(backoff_time)
                backoff_time *= 2
                backoff_time = min(backoff_time, 60)
            continue

        best_reasoning = reasoning_order_dict[shuffled_best_path_idx]
        votes_by_reasoning[best_reasoning] += 1

    total_votes = sum(votes_by_reasoning.values())
    if total_votes != ensemble_size:
        return None

    sorted_votes_by_reasoning = sorted(votes_by_reasoning.items())
    selected_reasoning = max(sorted_votes_by_reasoning, key=lambda item: item[1])[0]

    return selected_reasoning


def get_usmle_response_format(cot=True, structured_output=True):
    properties = dict()
    if cot:
        properties["reasoning"] = {
            "type": "string",
            "description": "Your step-by-step reasoning that logically progresses from the question to the final answer.",
        }

    properties["answer"] = {
        "type": "string",
        "description": "The answer to the question.",
        "enum": ["A", "B", "C", "D"]
    }

    required = ["reasoning", "answer"] if cot else ["answer"]

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }

    if structured_output:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": schema
            }
        }
    else:
        return {
            "type": "json_object",
            "schema": schema
        }


@dataclass
class Example:
    question: str
    choices: dict[str, str]
    answer: str
    reasoning: str

    def get_input(self):
        example_options = '\n'.join([f"{letter}. {c}" for letter, c in self.choices.items()])
        return f"""{self.question}

{example_options}
"""

    def get_output(self):
        return {
            "reasoning": self.reasoning,
            "answer": self.answer
        }

    def to_object(self):
        return {
            "question": self.get_input(),
            "response": self.get_output()
        }


def answer_usmle_question(
    client,
    question,
    choices,
    examples,
    temp=0,
    model_id='gpt-4o',
    cot=True
):
    system_prompt = "You are a medical expert taking the United States Medical Licensing Examination."

    if examples:
        example_str = '\n'.join([json.dumps(ex.to_object()) for ex in examples])
        system_prompt += f"""
Here are examples that demonstrate how to reason about the question:
{example_str}"""

    option_str = '\n'.join([f"{letter}. {c}" for letter, c in choices.items()])
    user_prompt = f"""{question}

{option_str}
"""

    fmt = get_usmle_response_format(cot=cot, structured_output=True)

    res = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temp,
        response_format=fmt,
        max_tokens=8192
    ).choices[0].message.content

    res = res.strip()
    res = res.replace('{\n', '{').replace('\n}', '}').replace('\x08', '').replace('\n"answer"', ' "answer"').replace('\n, "answer"', ', "answer"')
    res = re.sub(r'(?<!\\)\n', r'\\n', res)
    try:
        return json.loads(res)
    except Exception as exc:
        print(f'Can not load json {res}')

    return None


def get_chain_of_thought(cot_df, train_idx):
    rows = cot_df[cot_df['train_idx'] == train_idx]
    return rows.iloc[0]['cot'] if len(rows) else ''


def create_examples(training_indices, train_df, cot_df, max_examples=5):
    examples = list()

    for train_idx in training_indices:
        reasoning = get_chain_of_thought(cot_df, train_idx)
        if not reasoning:
            continue

        example_row = train_df.iloc[train_idx]
        example = Example(
            example_row['question'],
            example_row['options'],
            example_row['answer_idx'],
            reasoning
        )
        examples.append(example)

    return examples[:max_examples]


def answer_via_ensemble(client, model_id, question, choices, examples, temp, num_models=5, cot=True):
    answer_to_idx = dict((v, k) for k, v in choices.items())
    votes_by_answer_idx = defaultdict(int)
    backoff_time = 1

    max_tries = num_models * 2
    for trial in range(max_tries):
        if sum(votes_by_answer_idx.values()) == num_models:
            break

        choice_order_list = random.sample(list(choices.values()), len(choices))
        choice_order_dict = {chr(65 + i): option for i, option in enumerate(choice_order_list)}

        examples = random.sample(examples, len(examples))

        try:
            response = answer_usmle_question(
                client,
                question,
                choice_order_dict,
                examples,
                temp,
                model_id,
                cot
            )
        except Exception as exc:
            print(f'Exception generated: {exc}')
            if "token rate limit" in str(exc):
                print(f'Retrying in {backoff_time} seconds')
                time.sleep(backoff_time)
                backoff_time *= 2
                backoff_time = min(backoff_time, 60)

            continue

        if not response:
            continue

        shuffled_predicted_answer_idx = response['answer']
        predicted_answer = choice_order_dict[shuffled_predicted_answer_idx]
        predicted_answer_idx = answer_to_idx[predicted_answer]
        votes_by_answer_idx[predicted_answer_idx] += 1

    return votes_by_answer_idx
