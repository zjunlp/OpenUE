torch-model-archiver --model-name BERTForSEQ  \
	--version 1.0 --serialized-file ./seq_model/pytorch_model.bin \
	--handler ./handler_seq.py \
	--extra-files "./seq_model/config.json,./seq_model/setup_config.json,./seq_model/vocab.txt,./model.py" -f


sudo cp ./BERTForSEQ.mar /home/model-server/model-store/

# ./start_server.sh

# curl -X DELETE http://localhost:3001/models/BERTForSEQ/
# deploy the model
curl -v -X POST "http://localhost:3001/models?initial_workers=1&synchronous=false&url=BERTForSEQ.mar&batch_size=1&max_batch_delay=200"