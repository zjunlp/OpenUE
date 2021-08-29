sudo docker run --rm -it -p 3000:8080 -p 3001:8081 --name ner \
 -v /home/model-server/model-store:/home/model-server/model-store \
pytorch/torchserve:latest