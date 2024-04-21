# 加载环境变量
import openai
import os

os.environ["http_proxy"] = "http://localhost:1087"
os.environ["https_proxy"] = "http://localhost:1087"



from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_key = os.getenv("OPENAI_API_KEY")

print(openai.api_key)

# gpt-4
# gpt-3.5-turbo
# 基于 prompt 生成文本
def get_completion(prompt, model="gpt-4"):
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
使用shell 命令行
在hdfs如何查询hdfs总是否有数据 hdfs的路径如下 `hdfs dfs -cat   /apps/hive/warehouse/datasecurity/regular_mode/16/attachment/2024-01-11/*/*  | wc -l`
写的所有文件，`2024-01-11` 改为动态获取今天的日期，用5中方式获取 今天的日期
"""

# 这是系统预置的 prompt。魔法咒语的秘密都在这里
prompt = f"""
{instruction}

用户输入：
{input_text}
"""

response = get_completion(prompt)
print(response)
