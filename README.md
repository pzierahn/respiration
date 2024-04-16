# Respiratory Rate Estimation

Master thesis: "Investigation of Image Processing Techniques for Camera-Based Respiratory Rate Measurement with Machine
Learning"

## Setup development environment

```shell
# Connect to the remote machine with port forwarding
ssh -L 8888:localhost:8888 user@remote-machine

# Set the data directory
cd data;
ln -s /media/hdd2/07_Datenbank_Smarthome/Testaufnahmen/ subjects;

# Start jupyter notebook
cd notebooks;
jupyter notebook --no-browser --port=8888

# Docker build a new image
docker build -t respiratory-rate-estimation .

# Run the docker container
docker run -it --rm \
  -v $(pwd):/app \
  -p 8888:8888 \
  respiratory-rate-estimation
```