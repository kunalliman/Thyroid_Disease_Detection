FROM python:3.8

# Install necessary packages for building Python packages
# RUN apk add --no-cache musl-dev gcc make

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py
