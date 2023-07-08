FROM python:3.8-slim-buster

WORKDIR /setup

RUN mkdir -p /dagster_home 

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV DAGSTER_HOME=/
ENV PYTHONPATH=/src

# Copy dagster instance YAML to $DAGSTER_HOME
COPY dagster.yaml /
COPY workspace.yaml /

WORKDIR /

COPY src/ src/.

EXPOSE 3000

ENTRYPOINT ["dagit", "-w", "workspace.yaml", "-h", "0.0.0.0", "-p", "3000"]