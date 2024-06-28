# Respiratory Rate Measurement

This repository deals with the **"Investigation of Image Processing Techniques for Camera-Based Respiratory Rate
Measurement with Machine Learning"**. The goal is to compare different methods for respiratory rate estimation from
videos.

The following methods and models are compared:

* **Pixel intensity changes**: Analyze the pixel intensity changes in the chest region.
* **Optical flow**: Analyze the optical flow in the chest region.
* **MTTS-CAN**: The MTTS-CAN (Multi-Task Temporal Shift Convolutional Attention Network) model utilizes temporal shift
  modules and attention mechanisms to perform on-device, real-time measurement of cardiopulmonary signals using video
  input, capable of extracting vital signs like heart and respiration rates.
* **BigSmall**: The BigSmall model efficiently integrates diverse spatial and temporal scales through a dual-branch
  architecture for concurrent multi-task physiological measurement from video, leveraging techniques like Wrapping
  Temporal Shift Modules for enhanced temporal feature representation.
* **Fine-tuned EfficientPhys**: EfficientPhys is designed to extract PPG signals. The model is fine-tuned on the
  VitalCamSet dataset to estimate the respiratory rates.

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

This projects needs the following pretrained models:

### YOLO model

The YOLO model is used for person detection in the videos

```shell
# YOLO model
mkdir -p data/yolo;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.txt -O data/yolo/yolov3.txt;
wget https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.cfg -O data/yolo/yolov3.cfg;
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo/yolov3.weights;
```

### MTTS-CAN

```shell
# MTTS-CAN model
mkdir -p data/mtts_can;
wget https://github.com/xliucs/MTTS-CAN/raw/main/mtts_can.hdf5 -O data/mtts_can/mtts_can.hdf5;
```

### BigSmall model

```shell
mkdir -p data/rPPG-Toolbox;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_BigSmall_Multitask_Fold3.pth -O
data/rPPG-Toolbox/BP4D_BigSmall_Multitask_Fold3.pth;
```

### TS-CAN model

```shell
mkdir -p data/rPPG-Toolbox;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_PseudoLabel_TSCAN.pth -O data/rPPG-Toolbox/BP4D_PseudoLabel_TSCAN.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/MA-UBFC_tscan.pth -O data/rPPG-Toolbox/MA-UBFC_tscan.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/PURE_TSCAN.pth -O data/rPPG-Toolbox/PURE_TSCAN.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/SCAMPS_TSCAN.pth -O data/rPPG-Toolbox/SCAMPS_TSCAN.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/UBFC-rPPG_TSCAN.pth -O data/rPPG-Toolbox/UBFC-rPPG_TSCAN.pth;
```

### EfficientPhys models

```shell
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_PseudoLabel_EfficientPhys.pth -O data/rPPG-Toolbox/BP4D_PseudoLabel_EfficientPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/MA-UBFC_efficientphys.pth -O data/rPPG-Toolbox/MA-UBFC_efficientphys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/PURE_EfficientPhys.pth -O data/rPPG-Toolbox/PURE_EfficientPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/SCAMPS_EfficientPhys.pth -O data/rPPG-Toolbox/SCAMPS_EfficientPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/UBFC-rPPG_EfficientPhys.pth -O data/rPPG-Toolbox/UBFC-rPPG_EfficientPhys.pth;
```

### DeepPhys models

```shell
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/BP4D_PseudoLabel_DeepPhys.pth -O data/rPPG-Toolbox/BP4D_PseudoLabel_DeepPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/MA-UBFC_deepphys.pth -O data/rPPG-Toolbox/MA-UBFC_deepphys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/PURE_DeepPhys.pth -O data/rPPG-Toolbox/PURE_DeepPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/SCAMPS_DeepPhys.pth -O data/rPPG-Toolbox/SCAMPS_DeepPhys.pth;
wget https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/UBFC-rPPG_DeepPhys.pth -O data/rPPG-Toolbox/UBFC-rPPG_DeepPhys.pth;
```

### Mediapipe

```shell
mkdir -p data/mediapipe;
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite -O data/mediapipe/detector.tflite
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O data/mediapipe/face_landmarker_v2_with_blendshapes.task
```

### FlowNet2

To install the `FlowNet2-SD` flow net model, download the pre-trained model from
the [official repository](https://github.com/NVIDIA/flownet2-pytorch) or
click [here](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing) to download the model.
Place the model under `data/flownet`.

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

# Docker build a new image
docker build -f Dockerfile.cuda -t respiration-jupyter .

# Run the docker container
docker run -d --gpus all --rm \
  -v $(pwd):/app \
  -v $DATASET:/app/data/VitalCamSet \
  -p $JUPYTER_PORT:8888 \
  respiration-jupyter
```