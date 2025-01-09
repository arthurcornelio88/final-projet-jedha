FROM continuumio/miniconda3
WORKDIR /home
COPY . .
RUN apt update -y 
RUN apt upgrade -y 
RUN apt install nano
RUN pip install -r requirements.txt
CMD ["-m", "pytest", "tests/"]