import dotenv as load_env
from openai import OpenAI
from pprint import pprint
import constant as constant

load_env.load_dotenv("/env/.env")

client = OpenAI()

messages = [
    {
        "role": "system",
        "content": constant.system_message
    },
    {
        "role": "user",
        "content": "张鹏,未婚,华为，康居园"
    }
]

data = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    #stream=True
)

for d in data.choices:
    print(d.message.role + ": " + d.message.content)


