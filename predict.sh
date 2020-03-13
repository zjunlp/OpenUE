python run_classification.py \
  --task_name=classification \
  --do_predict=true \
  --data_dir=openue/classification/classification_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs6/model.ckpt-6000 \
  --max_seq_length=128 \
  --output_dir=./output/predicate_infer_out/epochs6/ckpt27000
python openue/classification/prepare_data_for_labeling_infer.py
python run_sequnce_labeling.py \
  --task_name=sequence_labeling \
  --do_predict=true \
  --data_dir=openue/sequence_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-85304 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/epochs9/ckpt22000
