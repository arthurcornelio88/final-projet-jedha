# Base image using Miniconda
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /home

# Copy the environment.yml file into the container
COPY environment.yml .

# Install the conda environment
RUN conda env create -f environment.yml

# Activate the environment by default
RUN echo "conda activate mlops" >> ~/.bashrc
ENV PATH /opt/conda/envs/mlops/bin:$PATH

# Set PYTHONPATH to include the / directory
ENV PYTHONPATH /home

# Copy the application code into the container
COPY . .

# Expose the port for the API (change if necessary)
EXPOSE 5000

# Run the API (or entry point)
CMD ["python", "./home/app/mlflow_train.py"]