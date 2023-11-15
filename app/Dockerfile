# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Set the environment variable for FastAPI debug mode
ENV FASTAPI_DEBUG=1

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app", "main:app", "--debug"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 
