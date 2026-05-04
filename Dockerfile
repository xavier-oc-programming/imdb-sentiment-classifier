FROM public.ecr.aws/lambda/python:3.12

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080
ENV AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /var/task

COPY requirements-lambda.txt .
RUN pip install -r requirements-lambda.txt --no-cache-dir

COPY app.py .
COPY templates/ templates/
COPY models/ models/

ENTRYPOINT ["python", "app.py"]
