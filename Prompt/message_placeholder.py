from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system','you are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])


#load chat history
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

#create prompt
# print(chat_history)

prompt = chat_template.invoke({'chat_history':chat_history,'query':'What is the status ?'})

print(prompt)