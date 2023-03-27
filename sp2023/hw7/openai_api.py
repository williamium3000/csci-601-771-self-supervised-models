import os
import openai
from datasets import load_dataset

dataset = load_dataset("boolq")
dataset = dataset.shuffle()  # shuffle the data
# construct prompt
num_true = 4
num_false = 4
pos_data = []
neg_data = []

for sample in dataset['train']:
    answer = sample["answer"]
    if answer == False:
        if num_false > 0:
            neg_data.append(sample)
            num_false -= 1
    else:
        if num_true > 0:
            pos_data.append(sample)
            num_true -= 1

prefix = "Consider the the problem with the following examples:\n"
for pos_sample, neg_sample in zip(pos_data, neg_data):
    passage = neg_sample["passage"]
    question = neg_sample["question"]
    answer = neg_sample["answer"]
    prefix += f"passage: {passage}\n question: {question} \n answer: {answer}\n"
    
    passage = neg_sample["passage"]
    question = neg_sample["question"]
    answer = neg_sample["answer"]
    prefix += f"passage: {passage}\n question: {question} \n answer: {answer}\n"
    
    prefix += "\n"


openai.api_key = "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"

num_test = 30

rec = num_test
num_correct = 0

for test_case in dataset['validation']:
    passage = test_case["passage"]
    question = test_case["question"]
    answer = test_case["answer"]
    
    prompt = prefix + f"passage: {passage}\n question: {question} \n answer: "
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    pred = bool(response["choices"][0]["text"].strip())
    num_correct += (pred == answer)
    rec -= 1
    if rec == 0:
        break
print(f"Test result: {num_correct} / {num_test}, accuracy {num_correct / num_test}")