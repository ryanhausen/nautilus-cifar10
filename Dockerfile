# Maybe move this to latest?
FROM tensorflow/tensorflow:1.8.0-devel-gpu-py3

# Add a workspace
RUN mkdir /root/src
RUN mkdir /root/data
WORKDIR /root/src

# Packages that I use
RUN pip install --upgrade pip
RUN pip install astropy
RUN pip install keras
RUN pip install comet-ml