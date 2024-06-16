FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update and upgrade the system
RUN apt-get update; \
    apt-get upgrade -y; \
    apt-get dist-upgrade -y

# Install OpenCV
RUN apt-get install -y git nano python3-opencv; \
    apt-get clean

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Start the Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]