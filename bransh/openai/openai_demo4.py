# 加载环境变量
import openai
import os

os.environ["http_proxy"] = "http://localhost:1087"
os.environ["https_proxy"] = "http://localhost:1087"


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_key = ''
print(openai.api_key)


# 基于 prompt 生成文本
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message["content"]


# 任务描述
instruction = """

"""

# 用户输入
input_text = """

"""

# 这是系统预置的 prompt。魔法咒语的秘密都在这里
prompt = f"""
请解释下下面的 shell 命令 `cut -d':' -f6-`

"""

response = get_completion(prompt)
print(response)