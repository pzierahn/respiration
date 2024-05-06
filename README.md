# Respiratory Rate Estimation

Master thesis: "Investigation of Image Processing Techniques for Camera-Based Respiratory Rate Measurement with Machine
Learning"

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
│   ├── analyse              <- Analysis of experiment results
│   ├── experiments          <- Extract signals from videos
│   ├── extractors           <- Demo of all extractors
│   ├── fine_tuning          <- Fine-tuning of the models
│   └── misc                 <- Miscellaneous notebooks like data exploration
└── respiration              <- Python package
    ├── analysis             <- Extract frequencies from signals
    ├── dataset              <- Data loading
    ├── extractor            <- Extractors for the respiratory rate
    ├── preprocessing        <- Preprocessing of the signals, e.g. filtering
    ├── roi                  <- Region of interest detection
    └── utils                <- Utility functions like video loading and transformation
```

## Install dependencies

* YOLO model for object detection.
* MTTS-CAN and BigSmall models for respiration rate estimation.

```shell
# YOLO model
mkdir -p data/yolo;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.txt -O data/yolo/yolov3.txt;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.cfg -O data/yolo/yolov3.cfg;
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo/yolov3.weights;

# MTTS-CAN model
mkdir -p data/mtts_can;
wget https://github.com/xliucs/MTTS-CAN/raw/main/mtts_can.hdf5 -O data/mtts_can/mtts_can.hdf5;

# BigSmall model
mkdir -p data/rPPG-Toolbox;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_BigSmall_Multitask_Fold3.pth -O data/rPPG-Toolbox/BP4D_BigSmall_Multitask_Fold3.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/PURE_EfficientPhys.pth -O data/rPPG-Toolbox/PURE_EfficientPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/UBFC-rPPG_EfficientPhys.pth -O data/rPPG-Toolbox/UBFC-rPPG_EfficientPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_PseudoLabel_EfficientPhys.pth -O data/rPPG-Toolbox/BP4D_PseudoLabel_EfficientPhys.pth;
```

## Setup development environment

```shell
# Connect to the remote machine with port forwarding
ssh -L LOCAL_PORT:localhost:JUPYTER_PORT user@remote-machine

# Set the data directory
cd data;
ln -s /media/hdd2/07_Datenbank_Smarthome/Testaufnahmen/ subjects;

# Start jupyter notebook
jupyter notebook --no-browser --port=JUPYTER_PORT

# Docker build a new image
docker build -t respiratory-rate-estimation .

# Run the docker container
docker run -it --rm \
  -v $(pwd):/app \
  -v $(DATASET):/app/data/VitalCamSet \
  -p JUPYTER_PORT:8888 \
  respiratory-rate-estimation
```