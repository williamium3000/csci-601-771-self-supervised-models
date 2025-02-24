from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM
from datasets import load_dataset
import tqdm

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
for pos_sample, neg_sample in tqdm.tqdm(zip(pos_data, neg_data)):
    passage = neg_sample["passage"]
    question = neg_sample["question"]
    answer = neg_sample["answer"]
    prefix += f"passage: {passage}\n question: {question} \n answer: {answer}\n"
    
    passage = neg_sample["passage"]
    question = neg_sample["question"]
    answer = neg_sample["answer"]
    prefix += f"passage: {passage}\n question: {question} \n answer: {answer}\n"
    
    prefix += "\n"




MODEL_NAME = "bigscience/bloom-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model.cuda()
num_test = 30

rec = num_test
num_correct = 0
for test_case in tqdm.tqdm(dataset['validation'], total=num_test):
    passage = test_case["passage"]
    question = test_case["question"]
    answer = test_case["answer"]
    prompt = prefix + f"passage: {passage}\n question: {question} \n answer: "
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, max_new_tokens=1)
    outputs = tokenizer.decode(outputs[0])
    pred = bool(outputs[-5:])
    num_correct += (pred == answer)
    rec -= 1
    if rec == 0:
        break
print(f"Test result: {num_correct} / {num_test}, accuracy {num_correct / num_test}")


