from langchain_community.llms import ChatGLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ChatGLM 私有化部署的 Endpoint URL
endpoint_url = "http://127.0.0.1:8001"

# 实例化 ChatGLM 大模型
llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[
        ["你是一个专业的销售顾问", "欢迎问我任何问题。"]
    ],
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)
