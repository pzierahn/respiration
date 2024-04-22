# Respiratory Rate Estimation

Master thesis: "Investigation of Image Processing Techniques for Camera-Based Respiratory Rate Measurement with Machine
Learning"

## Install dependencies

```shell
# Get the YOLO model for object detection
mkdir -p data/yolo;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.txt -O data/yolo/yolov3.txt;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.cfg -O data/yolo/yolov3.cfg;
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo/yolov3.weights;

# Get the MTTS-CAN pretrained model
mkdir -p data/mtts_can;
wget https://github.com/xliucs/MTTS-CAN/raw/main/mtts_can.hdf5 -O data/mtts_can/mtts_can.hdf5;

# Get the BigSmall model
mkdir -p data/rPPG-Toolbox;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_BigSmall_Multitask_Fold3.pth -O data/rPPG-Toolbox/BP4D_BigSmall_Multitask_Fold3.pth;
```

## Setup development environment

```shell
# Connect to the remote machine with port forwarding
ssh -L LOCAL_PORT:localhost:JUPYTER_PORT user@remote-machine

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
  -v $(DATASET):/app/data/subjects \
  -p JUPYTER_PORT:8888 \
  respiratory-rate-estimation
```