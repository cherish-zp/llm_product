from langchain_community.document_loaders import TextLoader

docs = TextLoader('/Users/zhangpeng/bigmodel/openai-quickstart/langchain/jupyter/tests/state_of_the_union.txt').load()

print(docs[0])
print(docs[0].metadata)
