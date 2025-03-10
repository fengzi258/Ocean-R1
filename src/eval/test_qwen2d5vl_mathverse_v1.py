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

# source /data_train2/mllm/anaconda3/bin/activate r1-v_dev
# CUDA_VISIBLE_DEVICES

# "xyliu6/mathverse_vision_dominant",
# "xyliu6/mathverse_4k"
dataset_name = "xyliu6/mathverse_vision_dominant"
sub_ds_name = "train"
max_new_tokens = 2048

MODEL_PATH_list = [

    "/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct",
    # "/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-1700",
    # "/checkpoint_mount/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-3850"

]

for MODEL_PATH in MODEL_PATH_list:
    print(MODEL_PATH)

    ckpt_name = "_".join(MODEL_PATH.split("/")[-3:])
    BSZ = 8 if "3B" in MODEL_PATH else 4
    OUTPUT_PATH="/data_train2/mllm/minglingfeng/code/R1-V/src/eval/logs/eval/mathverse_{ds_split}_grpo_{ckpt_name}.json"
 
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

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
    
    try:
        local_data_dir = "/data_train2/mllm/minglingfeng/mllm_data/eval_datasets"
        dataset_path =  os.path.join(local_data_dir, dataset_name.replace("/","_"))
        ds = datasets.load_from_disk(dataset_path)
    except:
        ds = datasets.load_dataset(dataset_name)


    Origin_QUESTION_TEMPLATE = "{Question}"
    SYSTEM_PROMPT = "Please solve the problem step by step and then answer it. If it is a multiple choice question, only provide the correct option letter, e.g., A, B, C, D, at the end.\n"
    QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."


    
    messages = []
    data = []

    sub_ds = ds[sub_ds_name]
    for i in sub_ds:
        question = i["problem"]
        answer = i["solution"]
        # options = i["options"]
        # if len(options) > 0:
        #     question = question + "\n".join(options)
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": i["image"],
                },
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT + QUESTION_TEMPLATE.format(Question=question)
                }
            ]
        }]
        messages.append(message)
        data.append(i)

    # minglingfeng fix
    # max_new_tokens = 512 #2048 if "checkpoint" in MODEL_PATH else 1024
    repetition_penalty = 2.0 #2.0 if "checkpoint" in MODEL_PATH else 1.0
    default_eval_kwargs = dict(
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                top_p=0.98,
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
        ground_truth = input_example['solution']
        reward,model_answer = default_accuracy_reward(original_output, ground_truth)
        if reward == 1.0:
            correct_number += 1
            is_correct = True
        else:
            is_correct = False
        
        if isinstance(model_answer,list):
            model_answer = model_answer[0]
        try:
            result = {
                'question': input_example["question"],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer':str(model_answer) if model_answer else "",
                'is_correct':is_correct
            }

        except Exception as e:
            print("no answer parsed",e,model_answer)
            result = {
                'question': input_example["question"],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer':None,
                'is_correct':is_correct
            }



        final_output.append(result)


    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH.format(ds_split=sub_ds_name, ckpt_name=ckpt_name)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")





