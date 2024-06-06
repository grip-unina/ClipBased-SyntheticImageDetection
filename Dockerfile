FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils wget unzip git
RUN pip install --upgrade pip
RUN pip install huggingface-hub>=0.23.0 timm>=0.9.10 scikit-learn pandas open_clip_torch tqdm

ADD ./ /workdir
WORKDIR /workdir

ENTRYPOINT [ "python", "main.py" ]
