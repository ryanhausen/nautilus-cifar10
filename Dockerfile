FROM tensorflow/tensorflow:1.8.0-devel-gpu-py3

RUN pip install
    astropy
    keras
    coemt-ml
