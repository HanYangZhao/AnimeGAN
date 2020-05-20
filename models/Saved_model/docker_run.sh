
# Serves the model with tensorflow/serving

docker run -p 8501:8501 --mount type=bind,source=$(pwd)/animegan_hayao/,target=/models/animegan_hayao -e MODEL_NAME=animegan_hayao -t tensorflow/serving