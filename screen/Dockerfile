
# FROM ubuntu:latest
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
RUN apt-get update
RUN apt-get install -y emacs
RUN apt-get install -y tmux
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y zip


RUN pip3 install --upgrade pip
RUN pip3 install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm wandb
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install timm

