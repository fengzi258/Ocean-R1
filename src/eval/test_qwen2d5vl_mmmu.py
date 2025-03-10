from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify
import os
import datasets
from utils import default_accuracy_reward
import copy
import argparse

max_new_tokens = 4096

def run():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path_list", default="/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct", type=str, help="model path")
    parser.add_argument("--output_dir", default='./src/eval/logs/eval/', type=str, help="save path")
    parser.add_argument("--dataset_name", default="MMMU/MMMU", type=str, help="dataset name")
    parser.add_argument("--ds_split", default="validation", type=str, help="test dataset name")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")

    # Parse the arguments
    args = parser.parse_args()

    MODEL_PATH_list = args.model_path_list.split(",")

    output_dir = f"{args.output_dir}"
    os.makedirs(output_dir, exist_ok = True)

    dataset_name = args.dataset_name
    ds_split = args.ds_split
    
    sub_dataset_name_list = ['Accounting', 'Agriculture', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Geography', 'Manage', 'Materials', 'Math', 'Mechanical_Engineering', 'Public_Health', 'Sociology', 'Architecture_and_Engineering', 'Art', 'Computer_Science', 'Energy_and_Power', 'Finance', 'History', 'Literature', 'Marketing', 'Music', 'Pharmacy', 'Physics', 'Psychology']


    for MODEL_PATH in MODEL_PATH_list:
        print(MODEL_PATH)
        BSZ = args.batch_size if "3B" in MODEL_PATH else args.batch_size//2

        ckpt_name = "_".join(MODEL_PATH.split("/")[-3:])
        
        OUTPUT_PATH=f"{output_dir}/mmmu_{ds_split}_grpo_{ckpt_name}.json"
    
        #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        try:
            if "Qwen2-VL" in MODEL_PATH:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                            MODEL_PATH,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            device_map="auto",
                        )
            elif "Qwen2.5-VL" in MODEL_PATH:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            MODEL_PATH,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            device_map="auto",
                        )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                            MODEL_PATH,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            device_map="auto",
                            )



            # default processer
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            # processor.tokenizer.padding_side = "left"
        except:
            continue

        Origin_QUESTION_TEMPLATE = "{Question}"
        SYSTEM_PROMPT = "Please solve the problem step by step and then answer it. If it is a multiple choice question, only provide the correct option letter, e.g., A, B, C, D, at the end.\n"
        QUESTION_TEMPLATE = "Question: {Question}\nOutput the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

        messages = []
        data = []
        for sub_dataset_name in sub_dataset_name_list:

            try:
                local_data_dir = "./src/eval/data"
                dataset_path =  os.path.join(local_data_dir, dataset_name, sub_dataset_name)
                ds = datasets.load_from_disk(dataset_path)
            except:
                ds = datasets.load_dataset(dataset_name, sub_dataset_name)

            sub_ds = ds[ds_split]
            for i in sub_ds:
                question = i["question"]
                answer = i["answer"]
                options = i["options"]
                if len(options) > 0:
                    choices = [f"{key}. {value}" for key,value in zip("ABCDEFG", eval(options))]
                    if len(choices)>0:
                        question = question + "\nChoices: " + "\n".join(choices)
                        i["question"] = question
                
                images = []
                for k in range(1,8):
                    if i[f"image_{k}"]:
                        images.append(i[f"image_{k}"])
                
                message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT  + QUESTION_TEMPLATE.format(Question=question)
                        }
                    ]
                }]
                
                for image in images:
                    message[0]["content"].append(
                            {
                            "type": "image", 
                            "image": image,
                        }
                    )
                messages.append(message)
                data.append(i)


        # minglingfeng fix
        # max_new_tokens = 2048 #2048 if "checkpoint" in MODEL_PATH else 1024
        repetition_penalty = 1.0 #2.0 if "checkpoint" in MODEL_PATH else 1.0
        default_eval_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    top_p=0.001,
                    top_k=1,
                    temperature=0.01,
                    repetition_penalty=repetition_penalty,
                )


        all_outputs = []  # List to store all answers

        # Process data in batches
        for i in tqdm(range(0, len(messages), BSZ)):
            batch_messages = messages[i:i + BSZ]
            batch_data = data[i:i + BSZ]
            
            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
            # generated_ids = model.generate(**inputs, **default_eval_kwargs)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)
            print(f"\033[31m{batch_messages[0]}\033[0m")
            print(f"\033[32m{batch_output_text[0]}\033[0m")
            print(f"ground_truth: {batch_data[0]['answer']}")
                
            print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")





        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data,all_outputs):
            original_output = model_output
            ground_truth = input_example['answer']
            reward,model_answer = default_accuracy_reward(original_output, ground_truth)
            if reward == 1.0:
                correct_number += 1
                is_correct = True
            else:
                is_correct = False
            
            result = copy.deepcopy(input_example)
            if isinstance(model_answer,list):
                model_answer = model_answer[0]
            for k in range(1,8):
                result[f"image_{k}"] = None
            try:
                result['ground_truth'] = ground_truth
                result['model_output'] =  original_output
                result['extracted_answer'] = str(model_answer) if model_answer else ""
                result['is_correct'] = is_correct

            except Exception as e:
                print("no answer parsed",e,model_answer)
                result['ground_truth'] = ground_truth
                result['model_output'] =  original_output
                result['extracted_answer'] = None
                result['is_correct'] = is_correct




            final_output.append(result)


        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(ds_split=ds_split, ckpt_name=ckpt_name)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run()

