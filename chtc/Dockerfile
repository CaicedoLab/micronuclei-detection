# build install based on python
# FROM python:latest
# WORKDIR /app
# COPY . /app
# RUN pip3 install -r requirements.txt

# RUN pip install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip3 install timm

# Do not copy github repo code here



FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y tmux
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install pandas numpy matplotlib scikit-image Pillow scipy scikit-learn tqdm
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install timm

