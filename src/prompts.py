from collections import defaultdict
from dataclasses import dataclass

import json
import random
import re

llm_judge_guidelines = """- Clarity: Assess whether the reasoning is expressed in a clear, concise, and understandable manner. Each step should be easily followed and free from unnecessary complexity.
- Logical Consistency: Verify that the reasoning avoids contradictions or logical fallacies and maintains consistency throughout.
- Coherence: Evaluate the logical flow between steps. Each step should logically follow from the previous one, with no gaps in reasoning or abrupt transitions.
- Depth of Explanation: Determine if the reasoning provides a detailed and thorough explanation for each step, including relevant assumptions and justifications where necessary.
- Accuracy: Check if the reasoning leads to a correct or reasonable conclusion based on the problem context. Ensure that each step aligns with the facts and information provided."""


def llm_as_a_judge(client, question, choices, answer, explanation, model_id='gpt-4o'):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "A step-by-step explanation of your thought process.",
                    },
                    "score": {
                        "type": "integer",
                        "description": "A numeric rating assessing the quality of the user's reasoning, ranging from 0 (lowest) to 10 (highest).",
                    }
                },
                "required": ["explanation", "score"],
                "additionalProperties": False
            }
        }
    }

    choices_str = '\n'.join([f"{letter}. {c}" for letter, c in choices.items()])

    user_prompt = f"""Evaluate the reasoning quality, focusing on its clarity, coherence, and logical progression. The reasoning should support an LLM's ability to follow a step-by-step chain of thought to arrive at the correct answer.

### Question:
{question}

### Choices:
{choices_str}

### Answer:
{answer}

### Reasoning:
{explanation}
"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": 'You are a medical expert.'},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        seed=42,
        response_format=response_format,
        max_tokens=8192
    )

    res = response.choices[0].message.content
    if not res:
        print(user_prompt)
        print('=' * 100)
        print(response.choices[0])

    res = res.strip()
    res = res.replace('{\n', '{').replace('\n}', '}').replace('\n"score"', ' "score"').replace('\n', '\\n')
    return json.loads(res)

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
    model_id='gpt-4o-2024-08-06',
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

    use_structured_output = 'gpt' in model_id
    fmt = get_usmle_response_format(cot=cot, structured_output=use_structured_output)

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


def get_chain_of_thought(gpt_df, llama_df, train_idx):
    gpt_rows = gpt_df[gpt_df['train_idx'] == train_idx]
    llama_rows = llama_df[llama_df['train_idx'] == train_idx]

    gpt_score = gpt_rows.iloc[0]['score'] if len(gpt_rows) else 0
    gpt_cot = gpt_rows.iloc[0]['cot'] if len(gpt_rows) else ''

    llama_score = llama_rows.iloc[0]['score'] if len(llama_rows) else 0
    llama_cot = llama_rows.iloc[0]['cot'] if len(llama_rows) else ''

    if not gpt_score and not llama_score:
        return None
    elif not gpt_score:
        return llama_cot
    elif not llama_score:
        return gpt_cot

    return gpt_cot if gpt_score >= llama_score else llama_cot


def create_examples(training_indices, train_df, gpt_cot_df, llama_cot_df, max_examples=5):
    examples = list()

    for train_idx in training_indices:
        reasoning = get_chain_of_thought(gpt_cot_df, llama_cot_df, train_idx)
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
            shuffled_predicted_answer_idx = response['answer']
            predicted_answer = choice_order_dict[shuffled_predicted_answer_idx]
            predicted_answer_idx = answer_to_idx[predicted_answer]
            votes_by_answer_idx[predicted_answer_idx] += 1
        except Exception as exc:
            continue

    return votes_by_answer_idx


# NotDiamond custom router doesn't support system prompts, so combining the sys and user prompts to approximate
# the actual llm calls that will occur when taking the usmle
def get_prompt_for_not_diamond(question, choices, examples):
    example_str = '\n'.join([json.dumps(ex.to_object()) for ex in examples])
    option_str = '\n'.join([f"{letter}. {c}" for letter, c in choices.items()])

    return f"""You are a medical expert taking the United States Medical Licensing Examination.

### Here are examples that demonstrate how to reason about the question:
{example_str}

### Question:
{question}

{option_str}
"""
