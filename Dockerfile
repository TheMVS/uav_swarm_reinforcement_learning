#Download base image ubuntu 20.04
FROM ubuntu:20.04

# Author information
LABEL maintainer="a.puentec@udc.es"
LABEL version="1.0"
LABEL description="UAV SWARM PATH PLANNING WITH \ REINFORCEMENT LEARNING FOR FIELD PROSPECTING: \ PRELIMINARY RESULTS"

# Set root
USER root

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Install pip
RUN apt install -y python3-pip vim

COPY . main/

# Install Python packages and add py file
RUN pip3 install numpy==1.19.2 shapely tensorflow keras matplotlib
CMD ["python3", "Program.py"]
CMD ["vim", "Config.py"]
CMD ["vim", "data.json"]

WORKDIR main/