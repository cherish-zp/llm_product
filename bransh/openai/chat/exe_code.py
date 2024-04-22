import os
from openai import OpenAI
import dotenv as load_env

load_env.load_dotenv("/env/.env")

client = OpenAI()

# 将模型 ID 传入 retrieve 接口
gpt_3 = client.models.retrieve("gpt-3.5-turbo")

data = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="生成可执行的快速排序 Python 代码",
    max_tokens=1000,
    temperature=0
)

text = data.choices[0].text
print(text)
# `exec` 函数会执行传入的字符串作为 Python 代码。
# 在这个例子中，我们使用 `exec` 来定义了一个 `quick_sort` 函数，然后你就可以调用这个函数了。
# 请注意，`exec` 可以执行任何 Python 代码，因此在使用它的时候一定要小心，特别是当你执行的代码来自不可信的来源时。
exec(text)
# quick_sort 是 text 里面返回的函数名
print(quick_sort([19, 123, 123, 234, 2343, 4534, 64, 561, 2, 3, 4, 5, 6, 7]))
