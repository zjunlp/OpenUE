python setup.py install
python main.py --gpus "0," --max_epochs 1  --num_workers=16 \
    --data_class REDataset \
    --litmodel_class INFERLitModel \
    --model_class Inference \
    --task_name interactive \
    --batch_size 16 \
    --model_name_or_path bert-base-chinese \
    --ner_model_name_or_path output/ske/ner/epoch=0-Eval \
    --seq_model_name_or_path output/ske/seq/epoch=2-Eval \
    --max_seq_length 256 \
    --check_val_every_n_epoch 1 \
    --data_dir ./dataset/ske