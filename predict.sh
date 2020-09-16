python run_classification.py \
  --task_name=classification \
  --do_predict=true \
  --data_dir=openue/classification/classification_data/$1 \
  --vocab_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/classification_model/wwm/model.ckpt-32457 \
  --max_seq_length=128 \
  --output_dir=output/predicate_infer_out/wwm/epoch6/
python openue/classification/prepare_data_for_labeling_infer.py $1
python run_sequnce_labeling.py \
  --task_name=sequence_labeling \
  --do_predict=true \
  --data_dir=openue/sequence_labeling/sequence_labeling_data/$1 \
  --vocab_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/wwm/model.ckpt-85304 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/wwm/epoch9/
