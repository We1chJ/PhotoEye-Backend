# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements if you have them
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Use gunicorn for production instead of Flask development server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "30", "server:app"]

# CMD ["sh"]

# Alternative if you want to stick with Python directly:
# CMD ["python", "server.py"]