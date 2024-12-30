FROM python:3.9.5-slim

WORKDIR /app

COPY ./bot /app

RUN pip install -r requirements.txt

EXPOSE 5000


CMD python bot.py & python main.py
