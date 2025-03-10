# MODEL_PATH_list=/checkpoint_mount/r1-v-3b-text-math-collected-44k-v1/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-1600,/checkpoint_mount/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-8500,/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct,/checkpoint_mount/r1-v-3b-concat-visual-data-w-rec-135k-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_135k_v2/checkpoint-3200
MODEL_PATH_list=/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct,/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-6250,/checkpoint_mount/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-8500
batch_size=4
data_split=test_en

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_olympiadbench_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size