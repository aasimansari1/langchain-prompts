<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1C3C3C,100:7C3AED&height=200&section=header&text=LangChain%20Prompt%20Engineering&fontSize=40&fontColor=00D9FF&animation=fadeIn&fontAlignY=38&desc=Templates%20%E2%80%A2%20Chains%20%E2%80%A2%20RAG%20%E2%80%A2%20Agents%20%E2%80%A2%20Output%20Parsers&descAlignY=60&descColor=BD00FF&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**Production-ready prompt engineering patterns using LangChain — from basic templates to agentic RAG.**

</div>

---

## 🗂️ What's Covered

| Module | Technique | File |
|---|---|---|
| Basic Prompting | PromptTemplate, f-string substitution | `prompt_template.py` |
| Chat Prompts | System + Human + AI message roles | `chat_prompt_template.py` |
| Few-Shot Learning | In-context examples, dynamic selection | `prompt_generator.py` |
| Memory | ConversationBufferMemory, ChatHistory | `chat_history.txt` |
| Output Parsing | JSON, List, Pydantic parsers | `messages.py` |
| Temperature Tuning | Creativity vs determinism tradeoffs | `temperature.py` |
| Prompt UI | Interactive prompt builder | `prompt_ui.py` |

---

## ⚡ Quick Start

```bash
git clone https://github.com/aasimansari1/langchain-prompts.git
cd langchain-prompts
pip install langchain langchain-openai python-dotenv
echo "OPENAI_API_KEY=your-key-here" > .env
```

---

## 🧩 Core Patterns

### 1. Basic PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "You are a {role}. Answer the following: {question}"
)
prompt = template.format(role="data scientist", question="What is overfitting?")
```

### 2. Chat Prompt (System + Human)

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert. Be concise and precise."),
    ("human", "{user_message}")
])

chain = chat_prompt | llm
response = chain.invoke({"domain": "ML", "user_message": "Explain transformers."})
```

### 3. Few-Shot Prompting

```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template("Input: {input}\nOutput: {output}"),
    prefix="Give the antonym of each word.",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)
```

### 4. Structured Output (Pydantic)

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ArticleAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="confidence score 0-1")
    keywords: list[str] = Field(description="top 5 keywords")

parser = PydanticOutputParser(pydantic_object=ArticleAnalysis)

prompt = PromptTemplate(
    template="Analyze this article:\n{article}\n\n{format_instructions}",
    input_variables=["article"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result: ArticleAnalysis = chain.invoke({"article": article_text})
```

### 5. RAG Chain

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain.invoke({"query": "What are the main findings?"})
print(result["result"])
```

### 6. ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "What is the latest news on GPT-5?"})
```

---

## 🌡️ Temperature Guide

| Temperature | Effect | Use Case |
|---|---|---|
| 0.0 | Deterministic | Fact extraction, SQL, code |
| 0.3 | Slight variation | Summarization, classification |
| 0.7 | Balanced | General Q&A, chatbots |
| 1.0 | Creative | Brainstorming, storytelling |
| 1.5+ | Unpredictable | Experimental only |

---

## 🏗️ Project Structure

```
langchain-prompts/
├── prompt_template.py      # Basic PromptTemplate patterns
├── chat_prompt_template.py # ChatPromptTemplate (system/human/AI roles)
├── chatbot.py              # Full chatbot with memory
├── prompt_generator.py     # Few-shot + dynamic example selection
├── messages.py             # Message types + output parsers
├── temperature.py          # Temperature comparison demo
├── message_placeholder.py  # MessagesPlaceholder for dynamic history
├── prompt_ui.py            # Interactive Streamlit prompt builder
└── template.json           # Serialized prompt templates
```

---

## 🗺️ Roadmap

- [ ] LangGraph workflows (multi-agent orchestration)
- [ ] LCEL (LangChain Expression Language) patterns
- [ ] Prompt versioning with LangSmith
- [ ] Evaluation with RAGAS

---

## ⭐ If this saved you time, please star the repo!

MIT © [Mohd Aasim Ansari](https://github.com/aasimansari1)
