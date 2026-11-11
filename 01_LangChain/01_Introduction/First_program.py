"""
     Template the prompt
"""

from langchain_core.prompts import ChatPromptTemplate

#####Create a prompt Template

prompt = ChatPromptTemplate.from_template(
    'Why People in this world make many {topic} and they don not admit.'
)



###Providing the Input

result = prompt.invoke({
    #'topic': 'Retrieval Argumented Generation'
    # 'topic': 'Embeddings'
    'topic': 'Mistakes'

})

# print(result)
# print("Message:", result.messages[0].content)
# print("Message type:", type(result.messages[0]))
# print("Additional kwargs:", result.messages[0].additional_kwargs)
# print("Response metadata:", result.messages[0].response_metadata)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are an expert {role}'),
    ('ai', 'Explain {topic} with an example')
])

result = prompt.invoke({
    'role': 'AI Engineer',
    'topic' : 'RAG'
})

print(result)