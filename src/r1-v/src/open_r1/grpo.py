# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from reward import accuracy_reward, format_reward, reasoning_steps_reward, get_cosine_scaled_reward, get_repetition_penalty_reward, len_reward

import math
import random
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
IOU_QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
  

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "cosine": get_cosine_scaled_reward(
        min_value_wrong=0,
        max_value_wrong=-0.5,
        min_value_correct=0.5,
        max_value_correct=1,
        max_len=1000,
    ),
    "repetition_penalty": get_repetition_penalty_reward(
        ngram_size=3,
        max_penalty=-1.0,
    ),
    "length": len_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # Load the dataset
    is_local = os.environ.get("IS_LOCAL", False)
    if is_local:
        data_path = f"/data_train2/mllm/minglingfeng/mllm_data/processed_data/r1_data/{script_args.dataset_name}"
        dataset = load_from_disk(data_path)
        # shuffle by seed
        try:
            random_number = random.randint(1, 100)
            dataset = dataset.shuffle(seed=random_number)
            print(f"Shuffle with seed: {random_number}")
        except Exception as e:
            print(e)
        print(dataset)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        if "has_image" in example:
            if example["has_image"]:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": IOU_QUESTION_TEMPLATE.format(Question=example["problem"]) if "reward_func" in example and example["reward_func"] == "iou" else QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ]
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": example["problem"]},
                    ],
            }
        else:
            return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ],
                }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        try:
            dataset = dataset.map(make_conversation_image, num_proc=32)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
        except Exception as e:
            print(e)
            print("Make Conversation Image Later")

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation, num_proc=32)
        try:
            dataset = dataset.remove_columns("messages")
        except:
            print("Pure Text Data ...")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
