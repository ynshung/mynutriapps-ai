FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python main.py

EXPOSE 5001

# Command to run the application
CMD ["waitress-serve", "--call", "main:create_app"]