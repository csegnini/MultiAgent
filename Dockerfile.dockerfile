# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

# Set the working directory in the container
WORKDIR $APP_HOME

# Install system dependencies needed for libraries like Matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    # Add any other system dependencies your Python packages might need
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container
# Cloud Run will automatically use this port.
EXPOSE 8080

# Define the command to run your app using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app