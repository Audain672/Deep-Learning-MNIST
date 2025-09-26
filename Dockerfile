FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY mnist_model.h5 .

EXPOSE 5000

CMD ["python", "app.py"]