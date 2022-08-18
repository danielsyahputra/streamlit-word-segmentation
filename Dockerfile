FROM python:3.8

EXPOSE 8501

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV NAME improc

CMD streamlit run App.py