# Respiratory Rate Estimation

Master thesis project for estimating respiratory rate from a video stream.

Title: "Investigation of Image Processing Techniques for Camera-Based Respiratory Rate Measurement with Machine
Learning"

## Motivation

Respiratory monitoring is crucial because it serves as a vital indicator of health, enabling the early identification of
serious health issues and improving patient outcomes through timely interventions [10]. Traditional methods for
monitoring respiration, such as spirometry, chest straps, or contact-based sensors like electrocardiograms (ECG), though
accurate, are often cumbersome and intrusive [3]. They may interfere with a patient's natural respiratory patterns or
cause discomfort during long-term monitoring, especially in sensitive populations such as infants, the elderly, or those
with chronic conditions [3]. Additionally, in certain environments or situations like sleep studies, intensive care
units, or remote health monitoring, non-intrusive methods are highly preferred to ensure patient comfort while
maintaining the ability to continuously monitor vital respiratory parameters [2].

## Problem Description

This Master Thesis aims to develop a state of the art machine learning algorithm that can extract respiratory signals
from videos. The development of video-based respiratory signal extraction technologies seeks to eliminate the need for
physical contact, offering a more comfortable and safer alternative for patients [1]. Moreover, video-based systems can
be deployed using existing hardware, such as smartphones, making it a cost-effective, scalable, and accessible option
for a wide range of applications [6]. Approaches proposed by previous studies [1, 2, 4, 5] rely heavily on Convolutional
Neural Network (CNN) for signal extraction. Since the advent of CNNs, new network architectures have emerged, promising
improved generalizability, reduced bias, and a decreased need for training data and computational resources. These
advances stem from both architectural innovations and novel training methodologies that aim to optimise neural network
performance, even in data-scarce or computationally constrained environments [9].

**Research Question: How can novel machine learning algorithms improve the extraction of respiratory signals from
videos?**

## Approach

In the first phase, respiration data will be matched with video data, preprocessed for quality and compatibility, and
explored to understand the dataset and guide future research steps. The dataset used is VitalCamSet[11] by Timon
Blöcher.
The second phase will focus on researching and replicating state-of-the-art model architectures from previous works in
the field of video-based respiratory signal extraction. Additionally, new model architectures will be explored, for
their potential to improve the accuracy and robustness of respiratory signal extraction.

To evaluate the performance and generalizability of the developed models, k-fold cross-validation will be employed. This
technique minimises the risk of overfitting and provides a robust assessment of the model's effectiveness in extracting
respiratory signals from videos.

## Bibliography

1. Chen, W. and McDuff, D., 2018. Deepphys: Video-based physiological measurement using convolutional attention
   networks. In Proceedings of the european conference on computer vision (ECCV) (pp. 349-365).
2. Chaichulee, Sitthichok, Mauricio Villarroel, Joao Jorge, Carlos Arteta, Kenny McCormick, Andrew Zisserman, and Lionel
   Tarassenko. "Cardio-respiratory signal extraction from video camera data for continuous non-contact vital sign
   monitoring using deep learning." Physiological measurement 40, no. 11 (2019): 115001.
3. Janssen, Rik, Wenjin Wang, Andreia Moço, and Gerard De Haan. "Video-based respiration monitoring with automatic
   region of interest detection." Physiological measurement 37, no. 1 (2015): 100.
4. Rocque, Mukul. "Fully automated contactless respiration monitoring using a camera." In 2016 IEEE International
   Conference on Consumer Electronics (ICCE), pp. 478-479. IEEE, 2016.
5. Zhan, Qi, Jingjing Hu, Zitong Yu, Xiaobai Li, and Wenjin Wang. "Revisiting motion-based respiration measurement from
   videos." In 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC),
   pp. 5909-5912. IEEE, 2020.
6. McDuff, Daniel J., Justin R. Estepp, Alyssa M. Piasecki, and Ethan B. Blackford. "A survey of remote optical
   photoplethysmographic imaging methods." In 2015 37th annual international conference of the IEEE engineering in
   medicine and biology society (EMBC), pp. 6398-6404. IEEE, 2015.
7. Selva, Javier, Anders S. Johansen, Sergio Escalera, Kamal Nasrollahi, Thomas B. Moeslund, and Albert Clapés. "Video
   transformers: A survey." IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
8. Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information
   processing systems 28 (2015).
9. Tavanaei, Amirhossein, Masoud Ghodrati, Saeed Reza Kheradpisheh, Timothée Masquelier, and Anthony Maida. "Deep
   learning in spiking neural networks." Neural networks 111 (2019): 47-63.
10. AL‐Khalidi, Farah Q., Reza Saatchi, Derek Burke, Heather Elphick, and Stephen Tan. "Respiration rate monitoring
    methods: A review." Pediatric pulmonology 46, no. 6 (2011): 523-529.
11. Blöcher, Timon, Simon Krause, Kai Zhou, Jennifer Zeilfelder, and Wilhelm Stork. "VitalCamSet-a dataset for
    Photoplethysmography Imaging." In 2019 IEEE Sensors Applications Symposium (SAS), pp. 1-6. IEEE, 2019.

## Setup development environment

```shell
# Connect to the remote machine with port forwarding
ssh -L 8888:localhost:8888 zierahn@ess-barclay.fzi.de

# Start jupyter notebook
jupyter notebook --no-browser --port=8888
```