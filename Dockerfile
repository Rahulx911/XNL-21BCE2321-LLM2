# # Use the official Python image
# FROM python:3.10

# # Set the working directory inside the container to /app
# WORKDIR /app

# # Copy everything from the TASK-SUMM-LLM folder into /app
# COPY . .

# # Switch to the StreamlitApp folder where your app is located
# WORKDIR /app/StreamlitApp

# # Install Python dependencies from requirements.txt
# RUN pip install -r requirements.txt

# # Expose the port for Streamlit
# EXPOSE 8501

# # *Remove HEALTHCHECK for now* (Can be added later)
# # HEALTHCHECK --interval=30s --timeout=10s --start-period=4000s --retries=3 \
# #   CMD curl --fail http://localhost:8501/ || exit 1

# # Command to run the app with CORS disabled and headless mode enabled
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS", "false", "--server.headless", "true", "--server.fileWatcherType=none"]



# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN  pip install -r requirements.txt

# Copy entire content of StreamlitApp.py folder
COPY . .

# Expose the port used by Streamlit (default 8501)
EXPOSE 8501

# Command to run the app with CORS disabled and headless mode enabled
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
