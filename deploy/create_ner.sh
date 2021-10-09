torch-model-archiver --model-name BERTForNER_en  \
	--version 1.0 --serialized-file ./ner_en/pytorch_model.bin \
	--handler ./deploy/handler.py \
	--extra-files "./ner_en/config.json,./ner_en/setup_config.json,./ner_en/vocab.txt,./deploy/model.py" -f


sudo cp ./BERTForNER_en.mar /home/model-server/model-store/

# ./start_server.sh
# delete the exist model
curl -X DELETE http://localhost:3001/models/BERTForNER_en/
# deploy the model
curl -v -X POST "http://localhost:3001/models?initial_workers=1&synchronous=false&url=BERTForNER_en.mar&batch_size=1&max_batch_delay=200"
