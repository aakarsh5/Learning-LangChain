from langchain_community.document_loaders import TextLoader

loader = TextLoader('text.txt',encoding = 'utf-8')

docs = loader.load()

print(docs[0].page_content)

print(docs[0].metadata)
