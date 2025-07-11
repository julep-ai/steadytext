# Migrating from LangChain to SteadyText

Simplify your AI stack by replacing complex LangChain abstractions with SteadyText's straightforward, deterministic approach.

## Why Migrate from LangChain?

| **LangChain** | **SteadyText** |
|---------------|----------------|
| Complex abstractions | Simple, direct API |
| Multiple dependencies | Single lightweight library |
| Non-deterministic chains | 100% deterministic outputs |
| Verbose configuration | Zero configuration |
| External LLM costs | Free local execution |
| Debugging nightmares | Predictable behavior |

## Quick Comparison

### Text Generation

**Before (LangChain):**
```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(temperature=0, api_key="sk-...")
template = """Summarize the following text:
{text}

Summary:"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

with get_openai_callback() as cb:
    summary = chain.run(text="Long text here...")
    print(f"Cost: ${cb.total_cost}")
```

**After (SteadyText):**
```python
import steadytext

summary = steadytext.generate(f"Summarize the following text: {text}")
# No chains, no templates, no callbacks, no costs!
```

### Embeddings and Vector Stores

**Before (LangChain):**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Complex setup
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key="sk-...")
db = FAISS.from_documents(docs, embeddings)

# Search
query = "What is the main topic?"
docs = db.similarity_search(query, k=4)
```

**After (SteadyText + PostgreSQL):**
```sql
-- Everything in SQL!
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024)
);

-- Load and embed documents
INSERT INTO documents (content, embedding)
SELECT 
    content,
    steadytext_embed(content)::vector
FROM load_text_file('document.txt');

-- Search
SELECT content, 1 - (embedding <=> steadytext_embed('What is the main topic?')::vector) AS score
FROM documents
ORDER BY embedding <=> steadytext_embed('What is the main topic?')::vector
LIMIT 4;
```

## Common LangChain Pattern Migrations

### 1. Prompt Templates → Direct Formatting

**Before:**
```python
from langchain import PromptTemplate, FewShotPromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+3", "output": "6"}
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve math problems:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)
result = chain.run(input="5+5")
```

**After:**
```python
def solve_math(problem):
    prompt = """Solve math problems:
Input: 2+2
Output: 4
Input: 3+3
Output: 6
Input: {problem}
Output:""".format(problem=problem)
    
    return steadytext.generate(prompt, max_tokens=10)

result = solve_math("5+5")  # Deterministic: always "10"
```

### 2. Chains → Simple Functions

**Before:**
```python
from langchain.chains import SimpleSequentialChain

# First chain: summarize
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Second chain: translate
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[summarize_chain, translate_chain],
    verbose=True
)

result = overall_chain.run(long_text)
```

**After:**
```python
def summarize_and_translate(text, target_lang="Spanish"):
    # Step 1: Summarize
    summary = steadytext.generate(f"Summarize: {text}", max_tokens=100)
    
    # Step 2: Translate
    translation = steadytext.generate(
        f"Translate to {target_lang}: {summary}", 
        max_tokens=150
    )
    
    return translation

result = summarize_and_translate(long_text)
```

### 3. Agents → Direct Logic

**Before:**
```python
from langchain.agents import load_tools, initialize_agent, AgentType

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("What is the weather in NYC and what is 234 * 432?")
```

**After:**
```python
def answer_question(question):
    # Determine what's needed
    needs = steadytext.generate_json(
        f"What tools are needed for: {question}",
        schema={
            "needs_search": {"type": "boolean"},
            "needs_calculation": {"type": "boolean"},
            "calculation": {"type": "string"}
        }
    )
    
    results = []
    
    if needs["needs_calculation"]:
        # Direct calculation
        calc_result = eval(needs["calculation"])  # In production, use safe eval
        results.append(f"Calculation: {calc_result}")
    
    if needs["needs_search"]:
        # Your search logic here
        results.append("Search: [Results]")
    
    return steadytext.generate(
        f"Answer based on: {results}\nQuestion: {question}"
    )
```

### 4. Document QA → SQL Queries

**Before:**
```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

loader = DirectoryLoader('docs/', glob="*.txt")
index = VectorstoreIndexCreator(
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000)
).from_loaders([loader])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever()
)

answer = qa.run("What is the main topic?")
```

**After:**
```sql
-- Create document chunks table
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255),
    chunk_number INTEGER,
    content TEXT,
    embedding vector(1024)
);

-- Load and chunk documents
INSERT INTO document_chunks (document_name, chunk_number, content, embedding)
SELECT 
    filename,
    chunk_number,
    chunk_text,
    steadytext_embed(chunk_text)::vector
FROM chunk_documents('docs/*.txt', 1000);

-- Question answering function
CREATE FUNCTION answer_question(question TEXT)
RETURNS TEXT AS $$
DECLARE
    context TEXT;
BEGIN
    -- Get relevant chunks
    SELECT string_agg(content, E'\n\n') INTO context
    FROM (
        SELECT content
        FROM document_chunks
        WHERE embedding <=> steadytext_embed(question)::vector < 0.3
        ORDER BY embedding <=> steadytext_embed(question)::vector
        LIMIT 4
    ) relevant_chunks;
    
    -- Generate answer
    RETURN steadytext_generate(
        format('Based on this context: %s\n\nAnswer: %s', 
               context, question)
    );
END;
$$ LANGUAGE plpgsql;
```

### 5. Output Parsers → Structured Generation

**Before:**
```python
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")

parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

prompt = PromptTemplate(
    template="Extract person info:\n{format_instructions}\n{text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run(text="John is 30 years old")
person = fixing_parser.parse(output)  # Might fail and retry!
```

**After:**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Guaranteed to return valid Person object!
result = steadytext.generate("Extract: John is 30 years old", schema=Person)
# Parses automatically from: "...<json-output>{"name": "John", "age": 30}</json-output>"
```

## Memory and State Management

### Conversation Memory

**Before (LangChain):**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response1 = conversation.predict(input="Hi, my name is John")
response2 = conversation.predict(input="What's my name?")
```

**After (Simple Python):**
```python
class SimpleConversation:
    def __init__(self):
        self.history = []
    
    def chat(self, user_input):
        # Build context from history
        context = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in self.history[-5:]  # Keep last 5 turns
        ])
        
        prompt = f"{context}\nUser: {user_input}\nAssistant:"
        response = steadytext.generate(prompt, max_tokens=100)
        
        # Save to history
        self.history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response

conv = SimpleConversation()
response1 = conv.chat("Hi, my name is John")
response2 = conv.chat("What's my name?")  # Will remember "John"
```

## Cost and Performance Comparison

### LangChain + OpenAI Costs
```python
# Typical LangChain application
def calculate_langchain_costs(daily_requests):
    # Multiple LLM calls per request due to chains
    avg_calls_per_request = 3  # Chain steps
    tokens_per_call = 500
    cost_per_1k_tokens = 0.002  # GPT-3.5
    
    daily_cost = (daily_requests * avg_calls_per_request * 
                  tokens_per_call / 1000 * cost_per_1k_tokens)
    
    print(f"Daily: ${daily_cost:.2f}")
    print(f"Monthly: ${daily_cost * 30:.2f}")
    print(f"Yearly: ${daily_cost * 365:.2f}")
```

### SteadyText Costs
```python
# SteadyText: Always $0 after installation
print("Daily: $0")
print("Monthly: $0") 
print("Yearly: $0")
print("Plus: 100x faster, 100% deterministic!")
```

## Testing Strategies

### Making Tests Deterministic

**Before (LangChain - Flaky):**
```python
def test_qa_chain():
    # This test might fail randomly!
    qa_chain = create_qa_chain()
    answer = qa_chain.run("What is the capital of France?")
    assert "Paris" in answer  # Sometimes fails!
```

**After (SteadyText - Reliable):**
```python
def test_qa_function():
    # Always passes with deterministic output
    answer = answer_question("What is the capital of France?")
    assert answer == "The capital of France is Paris."  # Exact match!
```

## Migration Strategy

### Phase 1: Replace Simple Chains
```python
# Start with single-step operations
# Replace: LLMChain → steadytext.generate()
# Replace: embedding + vectorstore → PostgreSQL + pgvector
```

### Phase 2: Simplify Complex Chains
```python
# Convert multi-step chains to simple functions
# Remove unnecessary abstractions
# Use straightforward Python logic
```

### Phase 3: Eliminate External Dependencies
```python
# Remove API-based tools
# Implement simple alternatives
# Use PostgreSQL for persistence
```

## Common Pitfalls & Solutions

### 1. Over-Engineering
```python
# ❌ LangChain habit: Creating chains for everything
chain = LLMChain(llm=llm, prompt=PromptTemplate(...))

# ✅ SteadyText: Just call the function
result = steadytext.generate("Your prompt here")
```

### 2. Callback Complexity
```python
# ❌ LangChain: Complex callback systems
callbacks = [StreamingStdOutCallbackHandler(), CustomCallback()]

# ✅ SteadyText: Simple iteration
for token in steadytext.generate_iter("Your prompt"):
    print(token, end="", flush=True)
```

### 3. Configuration Overload
```python
# ❌ LangChain: Tons of configuration
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100,
    model_kwargs={"top_p": 0.9},
    callbacks=callbacks,
    cache=True
)

# ✅ SteadyText: Sensible defaults
result = steadytext.generate("prompt", max_tokens=100)
```

## Real-World Migration Example

Here's a complete migration of a document QA system:

### Before (LangChain - 150+ lines)
```python
# Complex setup with multiple files, classes, and configurations
# Vector stores, embeddings, chains, callbacks, etc.
```

### After (SteadyText - 20 lines)
```sql
-- Complete QA system in PostgreSQL
CREATE OR REPLACE FUNCTION qa_system(question TEXT)
RETURNS TEXT AS $$
DECLARE
    context TEXT;
BEGIN
    -- Find relevant content
    SELECT string_agg(content, E'\n') INTO context
    FROM documents
    WHERE embedding <=> steadytext_embed(question)::vector < 0.3
    ORDER BY embedding <=> steadytext_embed(question)::vector
    LIMIT 3;
    
    -- Generate answer
    RETURN steadytext_generate(
        format('Context: %s\n\nQuestion: %s\n\nAnswer:', 
               context, question)
    );
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT qa_system('What is the main topic?');
```

## Migration Checklist

- [ ] List all LangChain components in use
- [ ] Identify which can be replaced with simple functions
- [ ] Install SteadyText
- [ ] Migrate embeddings to PostgreSQL
- [ ] Replace chains with direct function calls
- [ ] Remove prompt templates (use f-strings)
- [ ] Simplify memory management
- [ ] Update tests for deterministic outputs
- [ ] Remove all API key management
- [ ] Delete unused dependencies
- [ ] Celebrate simpler code! 🎉

## Next Steps

- [PostgreSQL Extension Guide →](../postgresql-extension-hero.md)
- [Simple Examples →](../examples/index.md)
- [API Reference →](../api/index.md)

---

!!! success "The Beauty of Simplicity"
    Most LangChain applications can be reduced to 10% of their original code size while gaining determinism, speed, and reliability. Less abstraction = fewer bugs!