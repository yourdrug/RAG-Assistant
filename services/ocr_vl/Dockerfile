FROM python:3.12-slim
WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir 'paddleocr[doc-parser]==3.6.0' fastapi==0.115.12 uvicorn==0.34.2
COPY service.py tables.py ./
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
