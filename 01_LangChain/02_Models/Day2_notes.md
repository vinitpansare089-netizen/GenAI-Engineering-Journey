# Day 2 — Chat Models (LangChain)

## 1. What are Chat Models?

Chat Models are LangChain components that allow applications to communicate with conversational language models.

Unlike a simple text-completion interface, Chat Models work with structured messages such as:
- SystemMessage
- HumanMessage
- AIMessage

They accept conversational input and generate conversational responses.

## 2. Understanding Groq, GPT-OSS-20B and ChatGroq

These three are different:

| Component | Purpose |
|---|---|
| Groq | Provides inference infrastructure to run AI models |
| GPT-OSS-20B | The actual language model |
| ChatGroq | LangChain integration for communicating with models through Groq |

Example:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
```

Here, ChatGroq connects our LangChain application to the selected model through Groq.

## 3. Invoking a Chat Model

The `.invoke()` method sends input to the model and receives its response.

```python
response = llm.invoke("Explain RAG")

print(response.content)
```

- `llm.invoke()` executes the model.
- `response` stores the returned AIMessage.
- `response.content` extracts the generated text.

## 4. Understanding AIMessage

Chat Models return an AIMessage object rather than just a normal Python string.

An AIMessage can contain:
- `content`: Generated text.
- `response_metadata`: Provider response information.
- `additional_kwargs`: Additional provider-specific information, such as tool-call data.

Example:

```python
response = llm.invoke("Explain embeddings")

print(response.content)
print(response.response_metadata)
print(type(response))
```

## 5. Understanding Prompt and Model Chaining

LangChain uses LCEL (LangChain Expression Language) to connect components using the pipe operator `|`.

Example:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {field}."),
    ("human", "Explain {tech} using real-life examples.")
])

chain = prompt | llm

result = chain.invoke({
    "field": "AI",
    "tech": "RAG"
})

print(result.content)
```

Execution flow:

1. Prompt template receives input variables.
2. Prompt formats the messages.
3. The formatted messages are passed to the Chat Model.
4. The model generates a response.
5. The chain returns an AIMessage.

## 6. Direct Invocation vs Chaining

**Direct invocation:**

```python
response = llm.invoke("Explain RAG")
```

**Invocation through a chain:**

```python
chain = prompt | llm

response = chain.invoke({
    "field": "AI",
    "tech": "RAG"
})
```

Both approaches return an AIMessage when the final component is a Chat Model.

To extract only the generated text:

```python
print(response.content)
```

## 7. Understanding the Return Types

| Expression | Return type |
|---|---|
| `llm.invoke("Explain RAG")` | AIMessage |
| `(prompt \| llm).invoke({...})` | AIMessage |
| `(prompt \| llm).invoke({...}).content` | String |

## 8. Key Takeaways

- Groq is an inference provider, not the language model itself.
- ChatGroq is a LangChain integration.
- Chat Models work with structured conversational messages.
- `.invoke()` executes the model or chain.
- AIMessage represents the model's response.
- `.content` extracts the generated text.
- LCEL's `|` operator connects components into a chain.
- Direct model invocation and prompt-model chaining are both useful.

**Day 2 — Chat Models: Completed (Fundamentals).**