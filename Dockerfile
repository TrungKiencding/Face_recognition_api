FROM python:3.9-slim

WORKDIR /app

# Cài đặt các gói phụ thuộc
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements.txt và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và model
COPY . .

# Cấu hình ngrok
COPY ngrok.yml /root/.config/ngrok/ngrok.yml

# Expose port
EXPOSE 8000

# Khởi động ứng dụng với ngrok
CMD ["sh", "-c", "ngrok http 8000 & uvicorn app:app --host 0.0.0.0 --port 8000"]
