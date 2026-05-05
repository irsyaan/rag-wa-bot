FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY wheelhouse ./wheelhouse

RUN pip install --no-index --find-links=./wheelhouse -r requirements.txt

COPY app ./app
COPY migrations ./migrations

RUN useradd --create-home --shell /bin/bash botuser

USER botuser

CMD ["python", "-m", "app.main"]