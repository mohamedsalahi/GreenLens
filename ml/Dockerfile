# Headless training environment for the classifier: python 3.12 plus the
# pinned packages from requirements.txt. The desktop app is not containerized.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /work
COPY requirements.txt /work/ml/requirements.txt
RUN pip install --no-cache-dir -r /work/ml/requirements.txt

COPY . /work/ml

CMD ["python", "/work/ml/train.py", "--help"]
