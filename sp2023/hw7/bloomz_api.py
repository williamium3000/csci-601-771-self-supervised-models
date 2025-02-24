import os
import openai
import requests
from datasets import load_dataset

dataset = load_dataset("boolq")
dataset = dataset.shuffle()  # shuffle the data
# construct prompt
num_true = 2
num_false = 2
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


API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
num_test = 30

rec = num_test
num_correct = 0

for test_case in dataset['validation']:
    passage = test_case["passage"]
    question = test_case["question"]
    answer = test_case["answer"]
    
    prompt = prefix + f"passage: {passage}\n question: {question} \n answer: "
    response = output = query({
        "inputs": prompt,
    })
    pred = bool(response[0]["generated_text"][-5:])
    num_correct += (pred == answer)
    rec -= 1
    if rec == 0:
        break
print(f"Test result: {num_correct} / {num_test}, accuracy {num_correct / num_test}")


