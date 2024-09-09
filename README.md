# Respiratory Rate Measurement

This repository deals with the **"Investigation of Image Processing Techniques for Camera-Based Respiratory Rate
Measurement with Machine Learning"**. The goal is to compare different methods for respiratory rate estimation from
videos.

## Project structure

```
.
├── data
│   ├── mtts_can             <- MTTS-CAN model
│   ├── rPPG-Toolbox         <- Models from rPPG-Toolbox
│   ├── VitalCamSet          <- VitalCamSet dataset
│   └── yolo                 <- YOLO model
├── evaluation               <- Evaluation data
├── figures                  <- Figures for the report
├── models                   <- Trained models
├── notebooks
│   ├── 00-Fine-Tuning       <- Fine-tuning of the models
│   ├── 01-Extractors        <- Demos of all extractors
│   ├── 02-Experiments       <- Extract signals from videos
│   ├── 03-Analysis          <- Analysis of experiment results
│   └── 04-Support           <- Miscellaneous notebooks like data exploration
└── respiration              <- Python package
    ├── analysis             <- Extract frequencies from signals
    ├── dataset              <- Data loading
    ├── extractor            <- Extractors for the respiratory rate
    ├── preprocessing        <- Preprocessing of the signals, e.g. filtering
    ├── roi                  <- Region of interest detection
    └── utils                <- Utility functions like video loading and transformation
```

## Get pretrained models

Download the pretrained models from the following repositories:

```shell
cd data;
sh download_dependencies.sh;
```

## Setup development environment

**Native setup:**

```shell
# Create a virtual environment
pip install virtualenv;
virtualenv .venv;
source .venv/bin/activate;

# Install the dependencies
pip install --upgrade pip torch torchvision torchaudio;
pip install --upgrade -r requirements.txt;

# Start jupyter notebook as a demon
nohup jupyter notebook --no-browser --port=$JUPYTER_PORT 1>jupyter.log 2>jupyter.log &
```

**Docker setup:**

```shell
# Connect to the remote machine with port forwarding
ssh -L LOCAL_PORT:localhost:$JUPYTER_PORT user@remote-machine

# Set the data directory
cd data;
ln -s /media/hdd2/07_Datenbank_Smarthome/Testaufnahmen/ VitalCamSet;

# Start jupyter notebook
jupyter notebook --no-browser --port=$JUPYTER_PORT
nohup jupyter notebook --no-browser --port=$JUPYTER_PORT 1>jupyter.log 2>jupyter.log &

# Docker build a new image
docker build -f Dockerfile.cuda -t respiration-jupyter .

# Run the docker container
docker run -d --gpus all --rm \
  -v $(pwd):/app \
  -v $DATASET:/app/data/VitalCamSet \
  -p $JUPYTER_PORT:8888 \
  respiration-jupyter
```