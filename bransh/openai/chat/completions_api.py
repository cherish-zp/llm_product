import os
from openai import OpenAI
import dotenv as load_env

load_env.load_dotenv("/env/.env")

client = OpenAI()

messages = [
    {
        "role": "user",
        "content": "Hello!"
    }
]

data = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
print(data)

data = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="生成可执行的快速排序 Python 代码",
    max_tokens=1000,
    temperature=0
)

text = data.choices[0].text
print(text)


