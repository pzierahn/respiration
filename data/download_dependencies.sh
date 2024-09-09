#!/bin/sh

# shellcheck disable=SC2164
# shellcheck disable=SC2103

cd flownet;
sh download.sh;
cd ..;

cd mediapipe;
sh download.sh;
cd ..;

cd mtts_can;
sh download.sh;
cd ..;

cd rPPG-Toolbox;
sh download.sh;
cd ..;

cd yolo;
sh download.sh;
cd ..;
