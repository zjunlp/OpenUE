python export_sequnce_labeling.py \
  --do_predict=true \
  --task_name=sequence_labeling \
  --data_dir=openue/sequence_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-85304 \
  --max_seq_length=128 \
  --output_dir=output/sequnce_labeling_model/wwm/ \
  --export_model_dir=export/seq
