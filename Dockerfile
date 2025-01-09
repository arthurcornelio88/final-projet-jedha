# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Create a new conda environment and activate it
RUN conda create --name myenv python=3.9 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install dependencies using conda or pip
COPY environment.yml .
RUN conda env update --file environment.yml --name myenv && conda clean --all --yes

# Copy the application code
COPY . .

# Expose the port for the API
EXPOSE 5000

# Command to run the MLflow model serving
CMD ["conda", "run", "-n", "myenv", "mlflow", "models", "serve", "-m", "/app/model", "-p", "5000"]
