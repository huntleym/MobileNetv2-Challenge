# Dockerfile (blueprint for building Images), Image (template for running containers), Container (actual running process for projects)
FROM python:3.8

ADD mobilenetv2-challenge.py .

RUN pip install matplotlib numpy tensorflow

CMD [ "python", "./mobilenetv2-challenge.py"]
