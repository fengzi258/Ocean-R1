from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch
import json
import tqdm
from math_verify import parse, verify
import argparse
import pandas as pd
from torch.multiprocessing import Process, set_start_method, Manager
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

import datasets
import os
import copy
from utils import default_accuracy_reward

max_new_tokens = 4096
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 1. get evaluation configuration <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_eval_config():
    parser = argparse.ArgumentParser(description="Inference script for GeoQA evaluation.")
    parser.add_argument("--model_path_list", type=str, default="/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct", help="Path to the model checkpoint (e.g., qwen2vl model or a fine-tuned model).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference. Reduce if GPU OOM (default: 50).")
    parser.add_argument("--output_dir", type=str, default="。/src/eval/logs/eval/", help="Path to save inference result (e.g., JSON file).")
    parser.add_argument("--dataset_name",  type=str, default="MathLLMs/MathVision", help="Path to the prompts JSONL file for GeoQA evaluation.")
    parser.add_argument("--data_split",  type=str, default="test", help="Path to the prompts JSONL file for GeoQA evaluation.")
    all_gpu = ",".join(map(str, range(torch.cuda.device_count())))
    parser.add_argument("--gpu_ids", default=all_gpu, help="comma-separated list of GPU IDs to use")
    args = parser.parse_args()
    return args

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 2. load testset <<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def prepare_test_messages(dataset_name, data_split):
    try:
        local_data_dir = "./src/eval/data"
        dataset_path =  os.path.join(local_data_dir, dataset_name)
        ds = datasets.load_from_disk(dataset_path)
    except:
        ds = datasets.load_dataset(dataset_name)
    
    SYSTEM_PROMPT = """Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n"""
    QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    
    tested_messages = []
    test_data = []

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
                    "text": SYSTEM_PROMPT + "Question: " + QUESTION_TEMPLATE.format(Question=input_q),
                }
            ]
        }]
        tested_messages.append(message)
        test_data.append(example)
    return test_data, tested_messages




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 3. use several GPUs to accelerate inference at testset <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def init_model(model_path, gpu_id):
    """init a model(args.model_path) on a specific gpu"""
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    if "Qwen2-VL" in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=f"cuda:{gpu_id}",
                )
    elif "Qwen2.5-VL" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=f"cuda:{gpu_id}",
                )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=f"cuda:{gpu_id}",
                    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return model, processor

def answer_a_batch_question_qwen(batch_messages, model, processor):
    """ let qwen answer a batch of questions """
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
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False) # do_sample=False
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(f"\033[31m{batch_messages[0]}\033[0m")
    # print(f"\033[32m{batch_output_text[0]}\033[0m")
    return batch_output_text

def infer_on_single_gpu(model_path, device_id, chunk_of_tested_messages, batch_size, results=None):
    """init model on this single gpu and let it answer asign chunk of questions"""
    model, processor = init_model(model_path, device_id)
    
    ### split batch
    responses = []
    batch_messages_list = [chunk_of_tested_messages[start: start + batch_size] 
               for start in range(0, len(chunk_of_tested_messages), batch_size)]

    for batch_messages in tqdm.auto.tqdm(batch_messages_list, desc=f"GPU {device_id} progress", position=device_id, leave=False):
        batch_output_text = answer_a_batch_question_qwen(batch_messages, model, processor)
        
        responses.extend(batch_output_text)
    
    results[device_id] = responses
    return
        
        
def multi_gpu_inference(prompts, gpu_ids, model_path, batch_size):
    """ let each gpu (along with a model) answer a chunk of questions """
    set_start_method("spawn", force=True)
    manager = Manager()
    gpu_id2result = manager.dict()

    gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    chunk_size = len(prompts) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(prompts)
        chunk = prompts[start_idx: end_idx]
        process = Process(target=infer_on_single_gpu, args=(model_path, gpu_id, chunk, batch_size, gpu_id2result))
        process.start()
        processes.append(process)

    # for process in tqdm.auto.tqdm(processes, desc="Inference progress", position=num_gpus, leave=True):
    for process in processes:
        process.join()

    all_predicts = []
    for gpu_id in gpu_ids:
        all_predicts.extend(gpu_id2result[gpu_id])

    return all_predicts

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 4. compute metrics <<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def compute_metrics(testset_data, all_predicts, args):
    final_output = []
    correct_number = 0

    for input_example, model_output in zip(testset_data, all_predicts):
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
    accuracy = correct_number / len(testset_data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # Save results to a JSON file
    ckpt_name = "_".join(args.model_path.split("/")[-3:])
    output_path=f"{args.output_dir}/{args.dataset_name.replace('/','_')}_{args.data_split}_grpo_{ckpt_name}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")



if __name__ == "__main__":
    args = get_eval_config()
    print(args)
    model_path_list = args.model_path_list.split(",")
    for model_path in model_path_list:
        args.model_path = model_path
        print(args.model_path)
        testset_data, tested_messages = prepare_test_messages(args.dataset_name, args.data_split)
        all_predicts = multi_gpu_inference(tested_messages, args.gpu_ids, args.model_path, args.batch_size)
        compute_metrics(testset_data, all_predicts, args)
