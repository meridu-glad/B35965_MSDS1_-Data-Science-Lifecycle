# 1. Use a lightweight Python image as a base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# 4. Upgrade pip and install dependencies
# We do this before copying the rest of the code to save time on rebuilds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
# This will copy your main.py AND your 'models' folder
COPY . .

# 6. Expose the port the API will run on
EXPOSE 8000

# 7. Command to run the FastAPI app using Uvicorn
# We use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

