MODEL_PATH_list=/checkpoint_mount/r1-v-3b-concat-visual-data-w-rec-135k-v2/exp-Qwen2.5-VL-3B-concat_visual_data_w_rec_135k_v2/checkpoint-4600

batch_size=8
dataset_name=xyliu6/mathverse_vision_dominant
# "xyliu6/mathverse_vision_dominant",
# "xyliu6/mathverse_4k"
data_split=train

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_mathverse_multigpu_v1.py --model_path_list $MODEL_PATH_list --dataset_name $dataset_name --data_split $data_split --batch_size $batch_size



batch_size=4
data_split=test

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_mathvision_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size



batch_size=4
data_split=test_en

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_olympiadbench_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size





batch_size=8
dataset_name=xyliu6/mathverse_4k
# "xyliu6/mathverse_vision_dominant",
# "xyliu6/mathverse_4k"
data_split=train

/data_train2/mllm/anaconda3/envs/r1-v_dev/bin/python /data_train2/mllm/minglingfeng/code/R1-V/src/eval/test_qwen2d5vl_mathverse_multigpu_v1.py --model_path_list $MODEL_PATH_list --dataset_name $dataset_name --data_split $data_split --batch_size $batch_size