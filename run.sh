

python run_seq.py --model_name_or_path bert-base-chinese --data_dir ./dataset --output_dir ./output_seq --save_steps 5000 --num_train_epochs 30 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --do_train 
python run_ner.py --model_name_or_path bert-base-chinese --data_dir ./dataset --output_dir ./output_ner --save_steps 2000 --num_train_epochs 20 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --do_train 
python Interactive.py --seq_model_path ./seq_model/ --ner_model_path ./ner_model/ --output_dir ./output/ --data_dir ./dataset --per_device_eval_batch_size 1 --do_predict  --task interactive
