torch-model-archiver --model-name BERTForNER  \
	--version 1.0 --serialized-file ./ner_model/pytorch_model.bin \
	--handler ./handler.py \
	--extra-files "./ner_model/config.json,./ner_model/setup_config.json,./ner_model/vocab.txt,./model.py" -f


sudo cp ./BERTForNER.mar /home/model-server/model-store/

# ./start_server.sh
# delete the exist model
# curl -X DELETE http://localhost:3001/models/BERTForNER/
# deploy the model
curl -v -X POST "http://localhost:3001/models?initial_workers=1&synchronous=false&url=BERTForNER.mar&batch_size=1&max_batch_delay=200"