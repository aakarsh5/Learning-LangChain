from langchain_community.document_loaders import WebBaseLoader

url = 'https://timesofindia.indiatimes.com/sports/cricket/asia-cup/asia-cup-indias-complete-15-member-squad-for-the-eight-team-tournament/articleshow/123382194.cms'
loader = WebBaseLoader(url)

docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)