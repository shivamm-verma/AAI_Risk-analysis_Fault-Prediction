FROM python:3.11-slim

WORKDIR /app
COPY Server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY Server /app/Server
COPY Model /app/Model

WORKDIR /app/Server
ENV PORT=5000
EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
