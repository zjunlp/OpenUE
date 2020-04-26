sudo docker run --name openue --network=host -p 8500:8500 -p 8501:8501 -v "$(pwd)/export/classification:/models/classification" -v "$(pwd)/export/seq:/models/seq"  -t tensorflow/serving
