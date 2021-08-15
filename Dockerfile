# Dockerfile (blueprint for building Images), Image (template for running containers), Container (actual running process for projects)
FROM python:3.8

WORKDIR /usr/src/app
RUN mkdir -p /usr/src/app/model

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "mobilenetv2-challenge.py"]
