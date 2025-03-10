# OpenV-R1


> OpenV-R1 investigates the efficacy of GRPO training in VLM using solely textual inference data, and achieves significant performance improvements across multiple tasks by incorporating multimodal data. 

**Resources:** 

[🤗 OpenV-R1-3B-Instruct](https://huggingface.co/minglingfeng/OpenV_R1_3B_Instruct)

[🤗 OpenV-R1 Training Visual Dataset](https://huggingface.co/datasets/minglingfeng/OpenV_R1_collected_visual_data)

[🤗 OpenV-R1 Training Text Dataset](https://huggingface.co/datasets/minglingfeng/Openv_R1_collected_text_data)


**OpenV-R1 Team:** 

[LingFeng Ming](https://github.com/fengzi258) · 

<!-- **Contributors**:

<a href="https://github.com/Deep-Agent/R1-V/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Deep-Agent/R1-V&max=30" />
</a> -->



---

### News

- 2025-03-10: We upload the evaluation script and polish the README. 
- 2025-03-10: We upload the training codebase.
- 2025-03-10: We release the OpenV-R1 repo.

---

### Our Findings
![Image](./assets/openv-r1.png)

- *In our experiments, when training exclusively with textual inference data, the model's performance on strong visual perception tasks significantly decreased. For instance, in the Grounding task (refcoco/+/g), average performance dropped from 75.3 to 2.4. This indicates that enhancing specific capabilities through focused training can impair other aspects of the model.*
- *However, on other tasks such as geometric reasoning and mathematical tasks, we observed varying degrees of performance improvement. This suggests that incorporating textual inference data enhances the VLM model's reasoning capabilities. Additionally, improvements were noted in counting tasks and general-purpose tasks, demonstrating that enhanced reasoning abilities can generalize to general tasks.*
- *When trained with multimodal data, the model demonstrates significant improvements across a variety of tasks, including counting, geometric reasoning, grounding, mathematical tasks, and general-purpose tasks.*




<!-- 
![image](https://github.com/user-attachments/assets/f5191b1e-dde2-42b7-9ec9-10f7f6213c12) -->


## Setup

```bash
conda create -n openv_r1 python=3.11 
conda activate openv_r1

bash setup.sh
```

> [!NOTE] 
> If you meet bug when running the script, first try align your environments with `./src/requirements.txt`


<!-- ### Supported Models

1. Qwen2-VL
2. Qwen2.5-VL 

### Supported Training Datasets

1. [🤗 R1V Training Dataset: CLEVR-70k-Counting](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train): Item Counting Problems

2. [🤗 R1V Training Dataset: CLEVR-70k-Complex](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_70K_Complex): Number Related Reasoning 

3. [🤗 R1V Training Dataset: GEOQA-8k](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K): Geometry Reasoning -->


<!-- ### Supported Evaluations

1. [SuperClevr-200](https://github.com/Deep-Agent/R1-V?tab=readme-ov-file#superclevr): Item Counting Problems
2. [GeoQA-Test-Direct-Answer-735](https://github.com/Deep-Agent/R1-V?tab=readme-ov-file#geoqa): Geometry Reasoning -->

## Training

### GRPO
- ./src/scripts/run_grpo_qwen2d5vl.sh
- ./src/scripts/run_grpo_vllm_qwen2d5vl_openv_r1_visual_data.sh

```bash
cd src/r1-v

HF_DATASET="minglingfeng/OpenV_R1_collected_visual_data" 

export FORMAT_REWARD_FACTOR=1.0
export IS_LOCAL="true" ## load_from_disk or load_dataset from huggingface: minglingfeng/OpenV_R1_collected_visual_data
export DEBUG_MODE="true"
export LOG_PATH=./src/logs/debug_qwen2p5_vl_3b_${HF_DATASET}.log
# export WANDB_API_KEY="xxxxx"
export WANDB_PROJECT="OpenV-R1"

QWEN_PATH=/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct
OUTPUT_DIR=./src/r1-v/src/outputs/exp-Qwen2.5-VL-3B/${HF_DATASET}
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME=3B-$HF_DATASET
DS_CONFIG="./src/r1-v/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.

# vLLM NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ./src/r1-v/src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.8 \
    --deepspeed ${DS_CONFIG} \
    --num_generations 7 
    # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance 

```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. 
> 2. If you meet **OOM Error**, you can try reduce `--num_generations` or set `gradient_checkpointing` as `true`.



### SFT

We also provide SFT code, please follow the script and edit the config to customize the sft task.

```bash
accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft.py --config src/r1-v/configs/qwen2vl_sft_config.yaml 
```

## Evaluation

*Todo*: 
- 结果表格展示
- Typical examples

### Counting: SuperCLEVR

```bash
cd ./src/eval/data
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip

# change image dir and the model path in the scripts
python ./src/eval/test_qwen2d5vl_counting_superclevr_5k.py
python ./src/eval/test_qwen2d5vl_counting_superclevr.py

```

### Geo Reasoning: GEOQA

<!-- <img width="379" alt="截屏2025-02-11 13 38 50" src="https://github.com/user-attachments/assets/0282872d-bfe5-40fa-ac00-8986450a0b1e" />
<img width="379" alt="截屏2025-02-11 14 54 16" src="https://github.com/user-attachments/assets/053ebb99-5f19-4599-be51-a7c335ab2b8b" /> -->



We provide the example script to evaluate on the test set (direct answer form) of [GEOQA](https://arxiv.org/abs/2312.11370).


```bash
# prepare images for testing
cd ./src/eval/data
git lfs install
git clone https://huggingface.co/datasets/Luckyjhg/Geo170K
cd Geo170K
unzip images.zip


# change image dir and the model path in the scripts
python ./src/eval/test_qwen2d5vl_geoqa.py

# To enable faster inference with multiple GPUs, you could also use the script in 
python ./src/eval/test_qwen2d5vl_geoqa_multigpu.py
```

### Referring Expression Comprehension (REC): RefCOCO/+/g and RefGTA
> 1. Download the [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip) and unzip it, and we refer to the image dir as `<your_image_root>`.

> 2. Download the [RefCOCO/+/g and RefGTA Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) and unzip it (RefGTA [Images](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/refgta.zip) is used for out-of-domain evaluation).

```bash
# Remember to change the model path, image root, and annotation path in the script
python ./src/eval/test_qwen2d5vl_rec.py
```

### Math: MathVision, MathVerse, and OlympiadBench
```bash
# Remember to change the model path, image root, and annotation path in the script
python ./src/eval/test_qwen2d5vl_mathvision_multigpu.py
python ./src/eval/test_qwen2d5vl_mathverse_multigpu.py
python ./src/eval/test_qwen2d5vl_olympiadbench_multigpu.py
```

### General: MMMU
```bash
python ./src/eval/test_qwen2d5vl_mmmu.py
```



## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V) (our initial codebase), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR), [G-LLAVA](https://arxiv.org/abs/2312.11370), [RefCOCO](https://github.com/lichengunc/refer), and [RefGTA](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2) for providing open source resources and to build the project. 
<!-- Special thanks to [Kimi](https://kimi.moonshot.cn/), [bAInance Labs](https://bainancelabs.com/) for supporting computation resources and [Yuxin Wu](https://scholar.google.com/citations?user=mJQI-gUAAAAJ&hl=en), [Xinyu Zhou](https://scholar.google.com/citations?user=Jv4LCj8AAAAJ&hl=en), [Baobao Chang](https://scholar.google.com.au/citations?user=LaKNyhQAAAAJ&hl=en) for their valuable advice. -->

## Citation

```bib
@misc{ming2025openvr1,
  author       = {Ming},
  title        = {OpenV-R1},
  howpublished = {\url{https://github.com/fengzi258/OpenV-R1}},
  note         = {Accessed: 2025-03-10},
  year         = {2025}
}
```



