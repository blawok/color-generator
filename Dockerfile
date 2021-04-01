# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY . .
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN set -ex && pip3 install -r ./requirements.txt

# Run the web server
EXPOSE 5000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py