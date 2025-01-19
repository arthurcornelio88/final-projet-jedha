FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml
RUN echo "conda activate mlops" >> ~/.bashrc
ENV PATH=/opt/conda/envs/mlops/bin:$PATH
COPY app/ /app
EXPOSE 8000
CMD ["uvicorn", "app.fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
