FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN apt-get update
RUN apt-get install -y emacs
RUN apt-get install -y tmux
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm wandb
RUN pip3 install timm