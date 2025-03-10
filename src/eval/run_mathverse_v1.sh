# MODEL_PATH_list=/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct,/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-6250,/data_train2/mllm/minglingfeng/code/R1-V/model_ckpts/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-8500
MODEL_PATH_list=/data_train2/mllm/minglingfeng/code/R1-V/model_ckpts/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-6250,/checkpoint_mount/r1-v-3b-concat-visual-data-w-rec-135k-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_135k_v2/checkpoint-5800
batch_size=8
dataset_name=xyliu6/mathverse_4k
# "xyliu6/mathverse_vision_dominant",
# "xyliu6/mathverse_4k"
data_split=train

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_mathverse_multigpu_v1.py --model_path_list $MODEL_PATH_list --dataset_name $dataset_name --data_split $data_split --batch_size $batch_size

# /data_train2/mllm/minglingfeng/code/R1-V/model_ckpts/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-6250

