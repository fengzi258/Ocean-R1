MODEL_PATH_list=/global_data/mllm/minglingfeng/models/Qwen2.5-VL-3B-Instruct
batch_size=8


dataset_name=xyliu6/mathverse_vision_dominant
# "xyliu6/mathverse_vision_dominant",
# "xyliu6/mathverse_4k"
data_split=train

python ./src/eval/test_qwen2d5vl_mathverse_multigpu.py --model_path_list $MODEL_PATH_list --dataset_name $dataset_name --data_split $data_split --batch_size $batch_size



batch_size=4
data_split=test

python ./src/eval/test_qwen2d5vl_mathvision_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size



batch_size=4
data_split=test_en

python ./src/eval/test_qwen2d5vl_olympiadbench_multigpu.py --model_path_list $MODEL_PATH_list --data_split $data_split --batch_size $batch_size
