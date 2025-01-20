# Use a lightweight Miniconda base image
FROM condaforge/miniforge3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Install dependencies using Mamba for faster resolution and cleanup unnecessary files
RUN mamba env create -f environment.yml && conda clean -afy && rm -f environment.yml

# Activate the environment in the container
ENV PATH="/opt/conda/envs/mlops/bin:$PATH"

# Copy only the application files
COPY app /app

# Expose the port for MLflow
EXPOSE 5000

# Command to run the MLflow tracking server
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
