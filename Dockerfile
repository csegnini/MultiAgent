FROM python:3.11-slim-bullseye
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --disabled-password --gecos "" myuser && \
    chown -R myuser:myuser /app

COPY . .

USER myuser

ENV PORT=8000
ENV PATH="/home/myuser/.local/bin:$PATH"

EXPOSE ${PORT}

CMD {"uvicorn", "main:app", "--host", "0.0.0.0"}