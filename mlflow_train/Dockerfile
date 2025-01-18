FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml
RUN echo "conda activate mlops" >> ~/.bashrc
ENV PATH /opt/conda/envs/mlops/bin:$PATH
COPY app/ /app
CMD ["python", "mlflow_train.py"]
