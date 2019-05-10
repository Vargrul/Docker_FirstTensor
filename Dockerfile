FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="ksla@create.aau.dk"

EXPOSE 7102

RUN apt update \
    && apt install -y \
        libglib2.0 \
        libsm6 \
        libxext6 \
        libxrender-dev

COPY requirements.txt .

RUN pip3 install -r requirements.txt
# RUN python3 -m pip install --upgrade pip

COPY ./src /app

# Set the working directory to /app
WORKDIR /app


# VOLUME [ "/data", "app" ]

# Copy the current directory contents into the container at /app

# CMD [ "python3", "data_prep_helper.py" ]
CMD [ "python3", "alexnet_v1.py" ]
# ENTRYPOINT [ "python3", "tensortest.py" ]
