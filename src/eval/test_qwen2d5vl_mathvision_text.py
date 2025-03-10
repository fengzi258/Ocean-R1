from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify
import os
import datasets
from utils import default_accuracy_reward, evaluate_mathvision
import copy
import base64
# source /data_train2/mllm/anaconda3/bin/activate r1-v_dev
# CUDA_VISIBLE_DEVICES

dataset_name = "MathLLMs/MathVision"
ds_split_list = ["test"]#["testmini", "test"]

MODEL_PATH_list = [
    # "/checkpoint_mount/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-3700",
    "/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct",
    # "/global_data/mllm/minglingfeng/models/Qwen2.5-VL-7B-Instruct",

    "/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-1600",
    "/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-1800",
    "/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-2000",

]

for MODEL_PATH in MODEL_PATH_list:
    print(MODEL_PATH)

    ckpt_name = "_".join(MODEL_PATH.split("/")[-3:])
    BSZ = 32 if "3B" in MODEL_PATH else 16
    OUTPUT_PATH="/data_train2/mllm/minglingfeng/code/R1-V/src/eval/logs/eval/mathvision_{ds_split}_grpo_{ckpt_name}.json"
 
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
        ds = datasets.load_data(dataset_name)


    Origin_QUESTION_TEMPLATE = "{Question}\nAnswer the question using a single word or phrase."
    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."


    for data_split in ds_split_list:
        messages = []
        data = []

        sub_ds = ds[data_split]
        for example in sub_ds:
            question = example["question"]
            answer = example["answer"]
            options = ""
            if len(example['options']) > 0 and ''.join(example['options']) != 'ABCDE':
                assert len(example['options']) == 5, example
                options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"

            input_q = f"{question}\n{options}"
            example["input_q"] = input_q

            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": example["decoded_image"],
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=input_q),## if "checkpoint" in MODEL_PATH else Origin_QUESTION_TEMPLATE.format(Question=input_q)
                    }
                ]
            }]
            messages.append(message)
            data.append(example)

        # minglingfeng fix
        max_new_tokens = 2048 #2048 if "checkpoint" in MODEL_PATH else 1024
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
            # generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            generated_ids = model.generate(**inputs, **default_eval_kwargs)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)
            print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")





        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['answer']
            reward,model_answer = evaluate_mathvision(original_output, ground_truth, input_example)
            if reward == 1.0:
                correct_number += 1
                is_correct = True
            else:
                is_correct = False
            
            result = copy.deepcopy(input_example)
            try:
                result["decoded_image"] = "" #base64.b64encode(result["decoded_image"])
                result['ground_truth'] = ground_truth
                result['model_output'] =  original_output
                result['extracted_answer'] = str(model_answer) if model_answer else ""
                result['is_correct'] = is_correct

            except Exception as e:
                print("no answer parsed",e,model_answer)
                result["decoded_image"] = ""#base64.b64encode(result["decoded_image"])
                result['ground_truth'] = ground_truth
                result['model_output'] =  original_output
                result['extracted_answer'] = None
                result['is_correct'] = is_correct



            final_output.append(result)


        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(ds_split=data_split, ckpt_name=ckpt_name)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")





