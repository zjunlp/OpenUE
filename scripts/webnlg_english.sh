# python setup.py install
dataset="WebNLG"



python main.py --gpus "0," --max_epochs 10  \
    --data_class REDataset \
    --litmodel_class RELitModel \
    --model_class BertForNER \
    --task_name ner \
    --batch_size 16 \
    --model_name_or_path bert-large-uncased \
    --max_seq_length 256 \
    --check_val_every_n_epoch 1 \
    --data_dir ./dataset/${dataset}  \
    --overwrite_cache \
    --lr 1e-5 


python main.py --gpus "0," --max_epochs 20  \
    --data_class REDataset \
    --litmodel_class SEQLitModel \
    --model_class BertForRelationClassification \
    --task_name seq \
    --batch_size 16 \
    --model_name_or_path bert-large-uncased \
    --max_seq_length 256 \
    --check_val_every_n_epoch 1 \
    --data_dir ./dataset/${dataset} \
    --overwrite_cache \
    --lr 3e-5 