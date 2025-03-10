MODEL_PATH_list=/checkpoint_mount/r1-v-3b-text-math-collected-44k-v0/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-6100,/checkpoint_mount/r1-v-3b-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_139k/checkpoint-8500,/checkpoint_mount/r1-v-3b-text-math-collected-44k-v1/exp-Qwen2.5-VL-3B-orz_math_57k_collected_44k/checkpoint-1600,/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct
batch_size=1
data_split=testmini

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_mathverse_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size