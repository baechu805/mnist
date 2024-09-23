FROM python:3.11

WORKDIR /code

RUN apt update
RUN apt install -y cron
COPY ml-work-cronjob /etc/cron.d/ml-work-cronjob
RUN crontab /etc/cron.d/ml-work-cronjob

COPY src/mnist/main.py /code/

RUN pip install --no-cache-dir --upgrade git+https://github.com/dMario24/mnist.git@0.5.0

CMD service cron start;uvicorn main:app --host 0.0.0.0 --port 8080 --reload
