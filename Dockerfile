FROM python:3.13-slim

WORKDIR /app

RUN apt update -y && apt install awscli -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "app.py"]