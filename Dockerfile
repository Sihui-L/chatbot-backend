FROM python:3.9.13-slim-buster

USER root

# Create non root user convergence_user
RUN adduser --quiet --disabled-password \
    --home /home/convergence_user \
    --shell /bin/bash convergence_user
RUN adduser convergence_user sudo

# Set working directory.
WORKDIR /srv

# Set PYTHONPATH
ENV PYTHONPATH="/srv"

# Install dependencies used on dev (includes testing packages)
COPY ./requirements.dev.txt .
COPY ./requirements/PIP_VERSION .

RUN pip install pip-tools

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip=="$(cat PIP_VERSION)" \
 && pip install --no-cache-dir -r requirements.dev.txt

RUN apt-get update && apt-get install make

COPY . /srv

USER convergence_user

EXPOSE 8081

CMD ["uvicorn", "src.main:application", "--host", "0.0.0.0", "--port", "8081"]
