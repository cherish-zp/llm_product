from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import dotenv as loca_env

loca_env.load_dotenv("/Users/zhangpeng/.bash_profile")

# 这是一个 LLMChain，用于根据剧目的标题撰写简介。

llm = OpenAI(temperature=0.7, max_tokens=1000)

template = """你是一位剧作家。根据戏剧的标题，你的任务是为该标题写一个简介。

标题：{title}
剧作家：以下是对上述戏剧的简介："""

prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)
synopsis_chain.verbose = True

# 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。
# llm = OpenAI(temperature=0.7, max_tokens=1000)
template = """你是《纽约时报》的戏剧评论家。根据剧情简介，你的工作是为该剧撰写一篇评论。

剧情简介：
{synopsis}

以下是来自《纽约时报》戏剧评论家对上述剧目的评论："""

prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)
review_chain.verbose = True

# 这是一个SimpleSequentialChain，按顺序运行这两个链
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

review = overall_chain.run("三体人不是无法战胜的")
print(review)

