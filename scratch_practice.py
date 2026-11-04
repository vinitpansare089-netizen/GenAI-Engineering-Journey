from langchain_core.prompts import ChatPromptTemplate 

######## Prompt Templating for LLM

prompt = ChatPromptTemplate.from_template(
    'Explain {topic} that if they exists or not'
)

result = prompt.invoke({
    'topic': 'GOD'
})

print(result)


