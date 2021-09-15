python main.py --gpus "0," --max_epochs 5  \
    --data_class REDataset \
    --litmodel_class SEQLitModel \
    --model_class BertForRelationClassification \
    --task_name seq \
    --batch_size 16 \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 256 \
    --check_val_every_n_epoch 1 \
    --data_dir ./dataset/ske 