# Do not copy github repo code here

# FROM ubuntu:latest
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
RUN apt-get update
RUN apt-get install -y emacs
RUN apt-get install -y tmux
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm wandb
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install timm

# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# # Install packages
# RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
# RUN yum clean all && yum makecache
# RUN yum update -y
# RUN yum install -y tmux 
# RUN yum install -y emacs
# RUN yum install -y python3 
# RUN yum install -y python3-pip

# RUN pip3 install --upgrade pip
# RUN pip3 install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm wandb
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# RUN pip3 install timm

