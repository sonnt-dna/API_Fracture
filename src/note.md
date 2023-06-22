
# Step 1: Chuyển tới Folder "src" (chứa source code, dockerfile, requirements.txt)

cd src

# Step 2: Build Docker Image trên máy Local (Lưu ý phải mở Docker Container trước khi build)

  - Thông thường sẽ build Docker bằng lệnh:

    docker build -t my_image .

  - Nếu gặp lỗi "need ARM64" (Các thư viện cho Query Datamart như ODBC hoặc các trường hợp khác,...) thì build Docker bằng lệnh sau:

    docker build -t my_image --platform=linux/amd64 .

# Step 3: Run Docker Container

docker run -p 8017:3500 my_image uvicorn app.main:app --reload --host 0.0.0.0 --port 3500   