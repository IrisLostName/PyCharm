import base64
import json
import os

import requests

# 创建一个会话对象
session = requests.Session()

login_url = "https://ehall.zjvtit.edu.cn/ywtb-portal/cusStandard/index.html?browser=no#/home/home"
# 假设的登录参数
login_data = {
    "username": "my_username",
    "password": "my_password"
}

try:
    # 发送登录请求
    # 注意：实际网站可能有CSRF token、复杂的加密参数或验证码，仅靠简单的post可能无法登录
    response = session.post(login_url, data=login_data)

    # 检查登录是否成功
    if response.status_code == 200:
        print("登录请求发送成功")

        # 登录成功后，session 内部已经保存了 cookie
        # 使用同一个 session 访问受保护的资源
        api_url = "http://example.com/api/getData"
        api_response = session.get(api_url)

        print(api_response.text)
    else:
        print("登录失败")

except Exception as e:
    print(f"发生错误: {e}")
# 模拟的 JSON 响应
response_text = """
{
    "success": true,
    "result": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
}
"""

def save_base64_image(json_data, filename, save_dir):
    try:
        data = json.loads(json_data)
        base64_str = data.get("result", "")

        # 检查并分离头部信息
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        # 解码
        img_data = base64.b64decode(base64_str)

        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, filename)

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(img_data)
        print(f"图片已保存: {file_path}")

    except Exception as e:
        print(f"解码或保存失败: {e}")

# 使用示例
save_base64_image(response_text, "student_001.png", "./photos")
