# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./app/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# We use the gunicorn.conf.py for configuration
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:create_app()"]
