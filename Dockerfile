# # Use the official Python image from the Docker Hub
# FROM public.ecr.aws/lambda/python:3.10

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Set the working directory in the container
# WORKDIR ./app_API

# # Copy the requirements file to the container
# COPY requirements.txt .

# # Install the dependencies
# RUN pip install -r requirements.txt --target "${mlops_assignment2}"

# # Copy the rest of the application code to the container
# COPY app.py .

# COPY random_forest_model.joblib .

# # Expose the port the app runs on
# EXPOSE 8080
# # Command to run the application
# CMD ["python", "app.py"]


# FROM python:3.10.4
# EXPOSE 8080
# ADD . /app.API
# WORKDIR /app_API
# RUN pip install -r requirements.txt --target "${mlops_assignment2}"
# CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "-w", "4"]


# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY / .

# Set environment variables if necessary (optional)
# ENV YOUR_ENV_VAR=value

# Command to run your application (modify as necessary)
CMD ["python", "app.py"]  
