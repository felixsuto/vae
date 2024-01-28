docker run -p 8888:8888 \
    -v $HOME/felix/vae:/host --gpus all \
    --rm -it vae:latest 