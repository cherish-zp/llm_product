from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv as loca_env

loca_env("/Users/zhangpeng/.bash_profile")

# 加载长文本
document_path = 'test/state_of_the_union.txt'
raw_documents = TextLoader(document_path).load()
# 实例化文本分割器
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# 分割文本
documents = text_splitter.split_documents(raw_documents)
print(documents)

embeddings_model = OpenAIEmbeddings()

# 将分割后的文本，使用 OpenAI 嵌入模型获取嵌入向量，并存储在 Chroma 中
db = Chroma.from_documents(documents, embeddings_model)

query = "What did the president say about Ketanji Brown Jackson"
# 使用文本进行寓意搜索
docs = db.similarity_search(query)
print("##################")
print(docs[0].page_content)

# 使用嵌入向量进行语义相似度搜索
embedding_vector = embeddings_model.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
