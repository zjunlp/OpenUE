torch-model-archiver --model-name BERTForSEQ_en  \
	--version 1.0 --serialized-file ./seq_en/pytorch_model.bin \
	--handler ./deploy/handler_seq.py \
	--extra-files "./seq_en/config.json,./seq_en/setup_config.json,./seq_en/vocab.txt,./deploy/model.py" -f


sudo cp ./BERTForSEQ_en.mar /home/model-server/model-store/

# ./start_server.sh

curl -X DELETE http://localhost:3001/models/BERTForSEQ_en/
# deploy the model
curl -v -X POST "http://localhost:3001/models?initial_workers=1&synchronous=false&url=BERTForSEQ_en.mar&batch_size=1&max_batch_delay=200"