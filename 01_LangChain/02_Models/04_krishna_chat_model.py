"""
Your task:
1. Initialize ChatGroq with temperature 0.
2. Create a SystemMessage that defines Krishna's personality as wise, compassionate and thoughtful.
3. Create a HumanMessage containing this dilemma: "I lied to my friend to avoid hurting their feelings. Was I right?"
4. Invoke the model.
5. Print the generated response using .content.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are God {Deity} - a wise compassionate and thoughtful god'),
    ('human', 'I lied to my friend to avoid hurting their feelings. Was I right?')
])

# result = prompt.invoke({
#     'Deity' : 'Krishna'
# })

# print(result)

llm = ChatGroq(
    model = 'chatgpt-oss-20b',
    temperature = 1,
    max_tokens = 1024
)

chain = prompt | llm

responce = chain.invoke({
    'Deity' : 'krishna'
})

print(responce.content)