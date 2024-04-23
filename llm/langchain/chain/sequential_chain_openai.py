from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import dotenv as loca_env

loca_env.load_dotenv("/Users/zhangpeng/.bash_profile")

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

prompt = PromptTemplate(
    input_variables=["product"],
    template="给制造{product}的有限公司取10个好名字，并给出完整的公司名称",
)

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)

chain.verbose = True

print(chain.run({
    'product': "性能卓越的GPU"
    }))
