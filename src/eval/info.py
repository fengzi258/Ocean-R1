import re
import os
import json
from tqdm import tqdm

question_file = "/data_train/code/mllm/minglingfeng/code/R1-V/src/eval/super_clevr/superCLEVR_questions_30k.json"


with open(question_file, "r")as f:
    data = json.load(f)

info = data["info"]
questions = data["questions"]

count_data = []
question_hash_list = []
for item in tqdm(questions):
    question = item["question"]
    image_filename = item["image_filename"]
    answer = item["answer"]
    question_hash = item["question_hash"]
    # if "count" in question_hash:
    if "How many" in question:
        count_data.append(item)
    question_hash_list.append(question_hash)
    # if type(answer) == int:
    #     print(item)

save_file = "/data_train/code/mllm/minglingfeng/code/R1-V/src/eval/prompts/super_clevr_test_5k.jsonl"

n_test = 5000
with open(save_file, "w")as f_write:
    for item in tqdm(count_data[-n_test:]):
        dd = {
            "image_path": item["image_filename"],
            "question": item["question"],
            "ground_truth": item["answer"]
        }
        f_write.write(json.dumps(dd)+ "\n")
# print(data)


# {"image_path": "./images/superCLEVR_new_025199.png", "question": "How many different items are there in the image?", "ground_truth": 3}
