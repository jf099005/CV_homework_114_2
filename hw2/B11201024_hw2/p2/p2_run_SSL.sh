### example for running the python files

threshold=.95
project_root=./
dataset_root=../../../hw2_data/p2_data
### Training (We would only run for checking your code is executable)
label=0417

# model_path=./checkpoint/resnet18_SSL_reinforcement_${label}.pth

model_path=./checkpoint/MyNet_SSL_reinforcement_${label}.pth

echo "training with model_path" ${model_path}

# lr=0.001

python p2_train.py\
    --dataset_dir ${dataset_root}\
    --save_path ${model_path}


for n_iter in {1..5};do
    echo iteration ${n_iter}
    python p2_gen_pseudo_label.py\
        --dataset_path ${dataset_root}/unlabel\
        --output_annotations_path ${project_root}/pseudo_labels.json\
        --val_dataset_path ${dataset_root}/val\
        --threshold ${threshold}\
        --model_path ${model_path}

    python p2_train.py\
        --dataset_dir ${dataset_root}\
        --extra_data_annotations ${project_root}/pseudo_labels.json\
        --checkpoint ${model_path}\
        --save_path ${model_path}
done