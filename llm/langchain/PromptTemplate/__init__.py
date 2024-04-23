from langchain.prompts import ChatPromptTemplate
import llm.langchain.model_io.constant as constant
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pprint import pprint

load_dotenv("/Users/zhangpeng/PycharmProjects/llm-project/env/.env")

if __name__ == '__main__':

    template = ChatPromptTemplate.from_messages([
        ("system", "{system_input}"),
        ("human", "{user_input}的分类是什么"),
    ])

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)

    # 生成提示
    messages = template.format_messages(
        system_input=constant.system_person_message,
        user_input="张鹏"
    )

    chat_result = chat_model.invoke(messages)
    pprint(chat_result.content)
