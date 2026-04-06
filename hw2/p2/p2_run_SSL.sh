### example for running the python files

threshold=.9
project_root=/mnt/20F408ADF408876E/114_2/computer_vision/homeworks/hw2/p2
dataset_root=/mnt/20F408ADF408876E/114_2/computer_vision/hw2_data/p2_data
### Training (We would only run for checking your code is executable)
# python3 p2_train.py --dataset_dir /mnt/20F408ADF408876E/114_2/computer_vision/hw2_data/p2_data
python p2_gen_pseudo_label.py\
    --dataset_path ${dataset_root}/unlabel\
    --output_annotations_path ${project_root}/pseudo_labels.json\
    --val_dataset_path ${dataset_root}/val\
    --threshold ${threshold}

python p2_train.py\
    --dataset_dir ${dataset_root}\
    --extra_data_annotations ${project_root}/pseudo_labels.json