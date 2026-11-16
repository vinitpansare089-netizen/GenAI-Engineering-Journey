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

# Day 2 — Model Parameters (LangChain)

## 1. What are Model Parameters?

Model parameters are configuration options that control how a language model generates its responses.

In LangChain, we can configure parameters while initializing a Chat Model.

Example:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024
)
```

Here:
- `model` specifies which language model to use.
- `temperature` controls randomness and variation.
- `max_tokens` sets the maximum completion-token budget.

## 2. Temperature

Temperature controls the randomness of token selection during generation.

### Temperature = 0

- Generally produces more predictable responses.
- Useful when consistency is important.
- Suitable as a starting point for structured tasks.

### Temperature = 0.5

- Provides moderate variation.
- Can balance consistency and creativity.

### Temperature = 1

- Generally produces more varied responses.
- Useful when exploring creative or diverse outputs.

**Important:** Temperature does not directly control accuracy, intelligence or response length. A temperature of zero does not guarantee identical answers in every situation.

### Practical experiment

I tested the same question multiple times:

| Temperature | Observation |
|---|---|
| 0 | Responses were the same across my three test runs |
| 1 | Responses varied across my three test runs |

### Application in PaapPunyaAI

For the initial Krishna personality implementation, temperature 0 is a reasonable starting point because consistency and controlled responses are important.

However, consistency also depends on prompt engineering, retrieval quality and response validation.

## 3. max_tokens

`max_tokens` controls the maximum number of completion tokens the model can generate for a response.

Example:

```python
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024
)
```

If we reduce the limit:

```python
max_tokens=100
```

The model has a much smaller completion-token budget.

It may produce a short answer or stop before completing its response.

### My practical experiment

I tested two different token limits:

| max_tokens | Observation |
|---|---|
| 1024 | The model had a larger generation budget |
| 100 | The model reached the limit and did not provide a useful visible answer in my test |

### Important observation: Reasoning tokens

My GPT-OSS-20B response metadata showed:

```python
{
    "completion_tokens": 100,
    "completion_tokens_details": {
        "reasoning_tokens": 98
    },
    "finish_reason": "length"
}
```

This showed that 98 of the 100 completion tokens were attributed to reasoning.

Therefore, very little budget remained for visible answer content.

**Engineering insight:** Some reasoning models consume completion tokens for internal reasoning as well as visible output. A small token budget can prevent a useful answer from being generated.

## 4. Understanding Response Metadata

LangChain Chat Models can return metadata alongside the generated response.

Example:

```python
response = llm.invoke("Explain RAG")

print(response.content)
print(response.response_metadata)
```

Important metadata fields:

| Field | Meaning |
|---|---|
| `prompt_tokens` | Tokens used by the input |
| `completion_tokens` | Tokens used during generation |
| `total_tokens` | Total input and completion tokens |
| `reasoning_tokens` | Tokens attributed to reasoning, when reported |
| `finish_reason` | Why generation stopped |
| `model_name` | Model used for generation |

### finish_reason

Common values include:

- `stop`: Generation finished normally.
- `length`: Generation stopped because a token limit was reached.

Exact metadata and finish reasons depend on the provider and model.

## 5. Engineering Applications

| Application | Starting temperature choice |
|---|---|
| Structured question answering | Low |
| Classification | Low |
| Consistent assistant personality | Low |
| Creative writing | Higher |
| Brainstorming | Higher |

These are starting points, not strict rules. Always evaluate the actual model's behavior.

## 6. Key Takeaways

1. Model parameters configure model generation.
2. Temperature controls randomness and variation.
3. Lower temperature generally produces more predictable responses.
4. Higher temperature generally produces more varied responses.
5. `max_tokens` sets a completion-token budget.
6. Reasoning tokens can consume part of that budget.
7. Response metadata helps diagnose generation behavior.
8. `finish_reason="length"` indicates a length-based stopping condition.

**Day 2 — Model Parameters: Fundamentals completed.**