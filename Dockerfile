# Start with the official Miniconda image
FROM continuumio/miniconda:latest

# updating conda
RUN conda update -n base conda
RUN conda install python=3.10

# Set the working directory to /app
WORKDIR /app

# Copy the environment.yml file to the container
COPY . .

# Create a new environment
RUN conda env create -f env.yml

# Activate the new environment
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]

WORKDIR /app
