from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from pprint import pprint
from dotenv import load_dotenv
import  constant
load_dotenv("/env/.env")

system_message = constant.system_person_message

messages = [SystemMessage(content=system_message),
            HumanMessage(content="Who won the world series in 2020?"),
            AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
            HumanMessage(content="Where was it played?")]

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
chat_model.stream(True)

pprint(messages)

chat_result = chat_model.invoke(messages)
type(chat_result)
pprint(chat_result)
while True:
    __input = input()
    messages.append(HumanMessage(content=__input))
    chat_result = chat_model.invoke(messages)
    pprint(chat_result)
