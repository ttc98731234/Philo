# 使用官方 Python 鏡像作為基礎鏡像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 防止 Python 在寫入 pyc 文件時緩存
ENV PYTHONDONTWRITEBYTECODE 1
# 防止 Python 在 stdout 和 stderr 上緩存
ENV PYTHONUNBUFFERED 1

# 更新包管理器並安裝 PostgreSQL 客戶端依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    # 清理 apt 緩存
    && rm -rf /var/lib/apt/lists/*

# 安裝 pipenv
# RUN pip install --upgrade pip
# RUN pip install pipenv

# 創建一個非 root 用戶
RUN addgroup --system app && adduser --system --ingroup app app

# 設置權限
RUN chown -R app:app /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 更改為非 root 用戶
USER app

# 複製應用程序代碼
COPY . .

# 暴露端口
EXPOSE 8000

# 運行應用程序
# 在生產環境中，可能需要使用 Gunicorn 等生產級服務器
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 