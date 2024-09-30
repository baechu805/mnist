FROM datamario24/py311tf:0.1.1

WORKDIR /code

RUN apt update
RUN apt install -y cron
COPY ml-work-cronjob /etc/cron.d/ml-work-cronjob
RUN crontab /etc/cron.d/ml-work-cronjob

COPY src/mnist/main.py /code
COPY run.sh /code/run.sh

RUN pip install --no-cache-dir git+https://github.com/baechu805/mnist.git@0.4.3

CMD ["sh", "run.sh"]
