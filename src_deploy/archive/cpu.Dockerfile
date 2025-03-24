FROM python:3.10-slim

WORKDIR /app
RUN mkdir -p /app/data
RUN mkdir -p /app/data/prompts


COPY ./src_deploy/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY ./src_deploy/ ./

# Expose the port
EXPOSE 8080

# Run the service
CMD ["python", "run_server.py"]