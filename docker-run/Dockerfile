FROM python:3.10-bullseye

WORKDIR /home/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
            vim nano curl


COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN pip install virtualenv

COPY . .

#Local
CMD ["python" , "/home/mlflowcalifornia-housing.py"]

# S3
# CMD ["mlflow" ,"ui" , "-h" , "0.0.0.0" ,"--default-artifact-root", "s3://..../"]



