FROM python:3.5

WORKDIR /Test

RUN apt update
RUN apt install python3-pip python3-dev -y
RUN pip3 install --upgrade --user tensorflow==1.13.1
RUN pip3 install keras==2.2.4
RUN pip3 install opencv-python==3.4.1.15
RUN pip3 install matplotlib==3.0.3

#COPY . /Test
