python export_classification.py \
  --task_name=classification \
  --do_predict=true \
  --data_dir=openue/classification/classification_data \
  --vocab_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/classification_model/epochs6/model.ckpt-6000 \
  --max_seq_length=128 \
  --output_dir=output/classification_model/wwm/ \
  -export_model_dir=./export/classification
