from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

model_path = "/Users/zhangpeng/bigmodel/Qwen-7B-Chat"

chatLLM = ChatTongyi(
    name="Qwen-7B-Chat",
    model_path=model_path,
    streaming=True,
)

message = HumanMessage(content="What is the capital of China?")

res = chatLLM.stream([message], streaming=True)
for r in res:
    print("chat resp:", r)
