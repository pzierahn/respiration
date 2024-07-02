#!/bin/bash

#
# Note install gdown with `pip install gdown`
# Source: https://github.com/NVIDIA/flownet2-pytorch
#

# FlowNet2
gdown 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da
ln -s FlowNet2_checkpoint.pth.tar FlowNet2_checkpoint.pth

# FlowNet2-C
gdown 1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE
ln -s FlowNet2-C_checkpoint.pth.tar FlowNet2-C_checkpoint.pth

# FlowNet2-CS
gdown 1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS
ln -s FlowNet2-CS_checkpoint.pth.tar FlowNet2-CS_checkpoint.pth

# FlowNet2-CSS
gdown 157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8
ln -s FlowNet2-CSS_checkpoint.pth.tar FlowNet2-CSS_checkpoint.pth

# FlowNet2-CSS-ft-sd
gdown 1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u
ln -s FlowNet2-CSS-ft-sd_checkpoint.pth.tar FlowNet2-CSS-ft-sd_checkpoint.pth

# FlowNet2-S
gdown 1V61dZjFomwlynwlYklJHC-TLfdFom3Lg
ln -s FlowNet2-S_checkpoint.pth.tar FlowNet2-S_checkpoint.pth

# FlowNet2-SD
gdown 1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn
ln -s FlowNet2-SD_checkpoint.pth.tar FlowNet2-SD_checkpoint.pth
