# upload_minio_test.py
import io
from datetime import timedelta

from minio import Minio

# ==== 基本参数 ====
# 本地：
ENDPOINT = "localhost:9000"
SECURE = False
# 服务器（示例）：把上面两行改为
# ENDPOINT = "s3.kusime.icu"; SECURE = True

ACCESS_KEY = "admin"  # 取自你的 .env
SECRET_KEY = "change_this_strong_password"
BUCKET = "tts-pipeline"

# ==== 初始化客户端 ====
client = Minio(
    ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=SECURE,
)

# 确保桶存在（不存在就创建）
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

# ==== 上传一个最小文件（bytes）====
data = b"hello minio"
client.put_object(
    BUCKET,
    "hello.txt",
    io.BytesIO(data),
    length=len(data),
    content_type="text/plain",
)



# ==== 生成预签名下载链接（1 小时有效）====
url = client.presigned_get_object(BUCKET, "hello.txt", expires=timedelta(hours=1))
print("Presigned URL:", url)
