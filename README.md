# NetMind-Face
This repo is the demo for real-time "Obama" face swap by using webcam

## Preparing
1 clone the github repo


2 install deps:

conda create --name demo_stgan python=3.8

conda activate demo_stgan

conda install matplotlib

pip install opencv-python

pip install pyyaml

pip install torch torchvision torchaudio

pip install torchsummary

pip install Cython

bash build.sh

3 download pretrained [weights](https://drive.google.com/file/d/1cmYMB0a8Hd_hySJSKa1gPwfW-npQ6erq/view?usp=sharing) and put it into ./weights/face2face/ 


## demo
python demo_face2face.py

## showcase

![showcase](https://github.com/protagolabs/NetMind-FacialAttributeEditing/blob/main/demo_stgan.gif)

from left -> right: Input, landmarks, mask, synthesis, output

## acknowledgement
[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)
