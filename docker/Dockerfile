FROM hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:1.0.1

USER root 

COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

