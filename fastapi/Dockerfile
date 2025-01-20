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

# Expose the default FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
