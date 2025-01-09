# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /

# Create a new conda environment
COPY environment.yml .
RUN conda env create --file environment.yml && conda clean --all --yes

# Activate the environment by default
SHELL ["conda", "run", "-n", "mlops", "/bin/bash", "-c"]

# Copy the application code
COPY . .

# Ensure the environment variables are set properly
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/

# Expose the port for the API
EXPOSE 5000

# Command to run the MLflow model serving
CMD ["conda", "run", "-n", "mlops", "python", "./app/mlflow_train.py"]
