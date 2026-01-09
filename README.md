# Generative AI LangChain Journey

A comprehensive learning repository exploring **LangChain**, **Large Language Models (LLMs)**, **embeddings**, and **prompt engineering** with practical Python examples. This project demonstrates core concepts from basic chat models to advanced structured outputs using Pydantic.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Prerequisites & Setup](#prerequisites--setup)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [File-by-File Documentation](#file-by-file-documentation)
- [How to Run Each Script](#how-to-run-each-script)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This repository is a step-by-step journey through LangChain and generative AI concepts:

1. **Chat Models** – Interact with LLMs (Groq)
2. **Embeddings** – Convert text to numerical vectors
3. **Semantic Search** – Find similar documents using embeddings
4. **Dynamic Prompts** – Build flexible, reusable prompts
5. **Message Handling** – Work with chat history and message types
6. **Chatbots** – Create interactive conversational agents
7. **Structured Outputs** – Extract structured data from LLM responses
8. **Data Validation** – Use Pydantic for type-safe data models

---

## 📁 Directory Structure

```
Generative-AI-Langchain/
├── 01_chat_models_langchain.py         # Basic LLM chat model usage
├── 02_emb_models_langchain.py          # Text embeddings with HuggingFace
├── 03_document_similarity_langchain.py # Semantic similarity search
├── 04_dynamic_prompt.py                # Streamlit UI for dynamic prompts
├── 05_prompt_generator.py              # Create and save prompt templates
├── 06_messages.py                      # Message types (System, Human, AI)
├── 07_chatbot.py                       # Interactive chatbot with history
├── 08_chat_prompt_template.py          # Chat prompt templating
├── 09_message_placeholder.py           # Advanced message handling with placeholders
├── 10_with_structured_output_pydantic.py # LLM output as structured TypedDict
├── 11_pydantic_demo.py                 # Pydantic model validation demo
├── template.json                       # Saved prompt template
├── chat_history.txt                    # Chat history storage
├── requirements.txt                    # Python dependencies
└── .env                                # Environment variables (not in repo)
```

---

## 🔧 Prerequisites & Setup

### Requirements
- **Python 3.8+** (3.10+ recommended)
- **pip** or **conda** for package management
- **GROQ API Key** (free at [console.groq.com](https://console.groq.com))
- **HuggingFace account** (optional, for model downloads)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DebjitDebjit2004/gen-ai-langchain-journey.git
   cd Generative-AI-Langchain
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file:**
   ```bash
   cp .env.example .env  # Or create manually
   ```
   Add your GROQ API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

---

## 🔐 Environment Variables

### Required Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `GROQ_API_KEY` | API key for Groq LLM service | [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | (Optional) HuggingFace token for gated models | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Example `.env` File

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

The `load_dotenv()` function in most scripts automatically loads these variables from the `.env` file.

---

## 📦 Dependencies

### Core Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | Latest | LLM orchestration framework |
| `langchain-groq` | Latest | Groq LLM integration |
| `langchain-huggingface` | Latest | HuggingFace embeddings integration |
| `langchain-core` | Latest | Core LangChain primitives |
| `pydantic` | Latest | Data validation and type hints |
| `python-dotenv` | Latest | Load `.env` files |
| `scikit-learn` | Latest | Cosine similarity calculations |
| `streamlit` | Latest | Web UI for demo apps |
| `email-validator` | Latest | Email validation for Pydantic |

### Install All Dependencies

```bash
pip install -r requirements.txt
```

### Manual Installation (if no requirements.txt)

```bash
pip install langchain langchain-groq langchain-huggingface langchain-core \
    pydantic python-dotenv scikit-learn streamlit email-validator
```

---

## 📚 File-by-File Documentation

### **01_chat_models_langchain.py**

**Purpose:** Introduction to LangChain's chat models using the Groq LLM.

**Key Concepts:**
- Initializing a chat model
- Setting temperature and max_tokens
- Invoking models with prompts
- Parsing responses

**Code Breakdown:**
```python
from langchain_groq import ChatGroq

# Initialize the model with Groq's llama-3.1-8b-instant
model = ChatGroq(
    model="llama-3.1-8b-instant",  # Small, fast model
    temperature=0.5,                # Moderate creativity (0-1 scale)
    max_tokens=500                  # Max response length
)

# Send a prompt and get a response
response = model.invoke("tell me about indian cricket. answer with one line")
print(response.content)  # Extract and print the text content
```

**What Happens:**
1. Loads `.env` for API credentials
2. Creates a ChatGroq instance
3. Sends a one-liner prompt about Indian cricket
4. Prints the LLM's single-line response

**Expected Output:**
```
Indian cricket is characterized by a passionate fan base, a rich history of legendary players, and a dominant position in world cricket with multiple ICC trophies.
```

**Temperature Explanation:**
- `0.0` = Deterministic (same answer every time)
- `0.5` = Balanced creativity
- `1.0` = Maximum randomness

---

### **02_emb_models_langchain.py**

**Purpose:** Generate numerical embeddings from text using HuggingFace models.

**Key Concepts:**
- Embeddings convert text to fixed-size vectors
- `sentence-transformers/all-MiniLM-L6-v2` is efficient and lightweight
- Embeddings enable semantic search and similarity comparisons
- Both documents and queries need embeddings

**Code Breakdown:**
```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384-dim vectors
)

# Sample documents
documents = [
    "India has one of the fastest growing economies.",
    "LangChain is a framework for building LLM-powered applications.",
    "Machine learning enables computers to learn from data."
]

# Convert all documents to vectors
vectors = embeddings.embed_documents(documents)
print(vectors)  # 3 vectors of 384 dimensions each
```

**What Happens:**
1. Downloads the MiniLM model (on first run, ~100MB)
2. Converts 3 documents into 384-dimensional vectors
3. Each vector captures semantic meaning

**Output Structure:**
```python
[
    [0.123, -0.456, 0.789, ...],  # Document 1 embedding
    [0.234, 0.567, -0.123, ...],  # Document 2 embedding
    [0.345, -0.678, 0.234, ...]   # Document 3 embedding
]
```

**Model Details:**
- **all-MiniLM-L6-v2**: 22M parameters, 384-dimensional output
- **Lightweight**: ~100MB, runs CPU-fast
- **General Purpose**: Works well for most semantic tasks

---

### **03_document_similarity_langchain.py**

**Purpose:** Find the most semantically similar document to a query using cosine similarity.

**Key Concepts:**
- Embeddings enable semantic (meaning-based) search
- Cosine similarity measures vector closeness (0-1 scale)
- Higher similarity = more semantically related
- Real-world use: FAQ matching, document retrieval

**Code Breakdown:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Documents about famous cricketers
documents = [
    "Virat Kohli is known for his aggressive batting...",
    "Sachin Tendulkar is regarded as one of the greatest...",
    "MS Dhoni is famous for his calm leadership...",
    "Rohit Sharma is known for his elegant batting...",
    "Jasprit Bumrah is recognized for his unorthodox bowling..."
]

# Query
query = "Who is the best keeper"

# Generate embeddings
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
# scores = [0.45, 0.52, 0.89, 0.48, 0.33]

# Find highest scoring document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
# index=2 (MS Dhoni), score=0.89

print(f"Query: {query}")
print(f"Best Match: {documents[index]}")
print(f"Similarity: {score:.2f}")
```

**What Happens:**
1. Creates embeddings for 5 cricket-related documents
2. Creates embedding for the query "Who is the best keeper"
3. Calculates cosine similarity between query and all documents
4. Returns the document with highest similarity score
5. MS Dhoni (document index 2) has the highest score because he's mentioned as a keeper/leader

**Expected Output:**
```
Query: Who is the best keeper
Best Match: MS Dhoni is famous for his calm leadership and led India to multiple ICC trophies.
similarity score is: 0.89
```

**Similarity Score Interpretation:**
- `1.0` = Identical meaning
- `0.5-0.9` = High semantic similarity
- `0.3-0.5` = Moderate similarity
- `0-0.3` = Low similarity

---

### **04_dynamic_prompt.py**

**Purpose:** Streamlit web UI for a research paper summarization tool with dynamic prompts.

**Key Concepts:**
- Streamlit creates interactive web dashboards
- Allows users to customize prompts with dropdowns
- `load_prompt()` reads saved prompt templates
- Chaining: template | model creates a pipeline
- `invoke()` runs the chain with specific variables

**Code Breakdown:**
```python
import streamlit as st
from langchain_core.prompts import load_prompt

st.header('Research Tool')

# Dropdown selections
paper_input = st.selectbox("Select Research Paper Name", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly",
    "Technical",
    "Code-Oriented",
    "Mathematical"
])

length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)"
])

# Load template
prompt = load_prompt("template.json")

# Process when button clicked
if st.button('Summarize'):
    chain = prompt | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
```

**What Happens:**
1. Displays a Streamlit interface with three dropdowns
2. User selects a paper, style, and length
3. Clicks "Summarize" button
4. Loads `template.json` prompt
5. Chains prompt with Groq model
6. Invokes with selected values
7. Displays LLM response on webpage

**How to Run:**
```bash
streamlit run 04_dynamic_prompt.py
```

Opens at `http://localhost:8501`

**Notes:**
- Requires `template.json` to exist (generated by script 05)
- Demonstrates LLM chaining: `prompt | model`
- Shows Streamlit selectbox and button widgets

---

### **05_prompt_generator.py**

**Purpose:** Create and save a reusable prompt template as JSON.

**Key Concepts:**
- `PromptTemplate` defines prompt structure with placeholders
- `input_variables` specifies required variables
- `validate_template=True` checks for mismatches
- `.save()` exports to JSON for reuse across scripts

**Code Breakdown:**
```python
from langchain_core.prompts import PromptTemplate

# Define a flexible prompt template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True  # Ensure all placeholders match input_variables
)

# Save to JSON
template.save('template.json')
```

**What Happens:**
1. Defines a multi-variable prompt with instructions
2. Specifies three required variables: `paper_input`, `style_input`, `length_input`
3. Validates template has matching placeholders
4. Saves to `template.json` for reuse

**Generated template.json:**
```json
{
  "template": "Please summarize the research paper titled \"{paper_input}\"...",
  "input_variables": ["paper_input", "style_input", "length_input"]
}
```

**Use Case:**
- Allows other scripts to load and reuse the same prompt
- Keeps prompt logic separate from code logic
- Easy to modify without code changes

---

### **06_messages.py**

**Purpose:** Demonstrate different message types in LangChain (System, Human, AI).

**Key Concepts:**
- `SystemMessage`: Set AI behavior/role
- `HumanMessage`: User input
- `AIMessage`: LLM response
- Messages form conversation history
- Different roles enable proper context handling

**Code Breakdown:**
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Build message list
messages = [
    SystemMessage(content='You are a knowledgeable doctor'),
    HumanMessage(content='Tell me about some acidity relief tablets name')
]

# Invoke model with message history
result = model.invoke(messages)

# Add AI response to history
messages.append(AIMessage(content=result.content))

print(messages)
```

**Message Flow:**
```python
[
    SystemMessage(content='You are a knowledgeable doctor'),
    HumanMessage(content='Tell me about some acidity relief tablets name'),
    AIMessage(content='Some common acidity relief tablets include...')
]
```

**What Happens:**
1. Creates a system message defining the AI's role as a doctor
2. Creates a human message asking about acidity tablets
3. Invokes the model with this message history
4. Appends the AI's response to the history
5. Prints the full conversation message list

**Message Types Explained:**

| Type | Purpose | Example |
|------|---------|---------|
| `SystemMessage` | Set AI's behavior and role | "You are a helpful doctor" |
| `HumanMessage` | User input/questions | "What is diabetes?" |
| `AIMessage` | LLM responses | "Diabetes is a metabolic disorder..." |

**Expected Output:**
```python
[
    content='You are a knowledgeable doctor' role='system',
    content='Tell me about some acidity relief tablets name' role='user',
    content='Some common acidity relief tablets include Antacid tablets, Omeprazole, Ranitidine...' role='assistant'
]
```

---

### **07_chatbot.py**

**Purpose:** Interactive multi-turn chatbot with persistent chat history.

**Key Concepts:**
- Maintain chat history across turns
- System message sets AI personality
- `while True` loop for continuous conversation
- 'exit' command to terminate
- Demonstrates stateful conversation

**Code Breakdown:**
```python
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# Initialize with system message
chat_history = [
    SystemMessage(content='You are a helpful and knowledgeable doctor.')
]

while True:
    # Get user input
    user_msg = input('you: ')
    chat_history.append(HumanMessage(content=user_msg))
    
    # Exit condition
    if user_msg == 'exit':
        break

    # Get AI response
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    # Display response
    print("=============================================")
    print("AI: ", result.content)
    print("=============================================")

print(chat_history)
```

**Conversation Flow:**

```
you: What is diabetes?
AI: Diabetes is a chronic metabolic disorder...
=============================================

you: What are its symptoms?
AI: Common symptoms include increased thirst, frequent urination...
=============================================

you: exit
```

**Chat History Evolution:**
```python
[
    SystemMessage(...),
    HumanMessage('What is diabetes?'),
    AIMessage('Diabetes is...'),
    HumanMessage('What are its symptoms?'),
    AIMessage('Common symptoms...')
]
```

**What Happens:**
1. Initializes with system message (doctor role)
2. Enters infinite loop for multi-turn conversation
3. Each turn: append human message, get AI response, append to history
4. All previous messages sent to model (provides context)
5. Type 'exit' to quit
6. Prints full conversation history at end

**Key Feature:**
- **Memory**: Each turn includes ALL previous messages
- **Context Awareness**: AI knows entire conversation history
- **Realistic Chat**: Feels like real dialogue

---

### **08_chat_prompt_template.py**

**Purpose:** Create and use chat prompt templates with variable substitution.

**Key Concepts:**
- `ChatPromptTemplate`: Combines system and human messages
- Tuples: `('role', 'message_template')` format
- Variables in curly braces `{variable}`
- `.invoke()` fills variables and returns message list

**Code Breakdown:**
```python
from langchain_core.prompts import ChatPromptTemplate

# Create chat template with two message types
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Expert in simple terms, what is {topic}')
])

# Fill variables
prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'googly'})

# prompt is now a list of messages:
# [
#     SystemMessage('You are a helpful cricket expert'),
#     HumanMessage('Expert in simple terms, what is googly')
# ]

# Invoke model with prepared messages
response = model.invoke(prompt)
print(response.content)
```

**What Happens:**
1. Defines a chat template with system and human roles
2. Specifies two variables: `{domain}` and `{topic}`
3. Invokes template with domain='cricket', topic='googly'
4. Generates proper message objects
5. Sends to model
6. Prints the explanation

**Expected Output:**
```
A googly is a type of bowling action in cricket where the ball is bowled with a leg-break action but spins in the opposite direction. It's an off-break that appears to be a leg-break...
```

**Benefits Over Raw Strings:**
- Type-safe message creation
- Variable validation
- Easy to reuse and compose
- Clean separation of prompt structure

---

### **09_message_placeholder.py**

**Purpose:** Advanced chat templating with message placeholders for dynamic history injection.

**Key Concepts:**
- `MessagesPlaceholder`: Inject entire message lists into prompts
- Load chat history from file
- Useful for inserting conversation context without hardcoding
- Separates template structure from content

**Code Breakdown:**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create template with placeholder for chat history
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Load history from file
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# Interactive loop
while True:
    user_query = input('You: ')
    if user_query == 'exit':
        break
    
    # Inject history and current query
    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_query
    })
    
    response = model.invoke(prompt)
    print('AI: ', response.content)
```

**Template Structure:**
```
System: "You are a helpful customer support agent"
[... injected chat_history messages ...]
Human: user's current query
```

**What Happens:**
1. Loads `chat_history.txt` (lines as strings)
2. Each iteration: injects history + new query into template
3. Sends complete prompt to model
4. Model has context of entire conversation
5. Type 'exit' to quit

**File Format (chat_history.txt):**
```
User asked about refund policy
Agent explained 30-day policy
User asked about processing time
Agent said 5-7 business days
```

**Use Cases:**
- Customer support chatbots
- Document Q&A with context
- Reference conversations during new queries

---

### **10_with_structured_output_pydantic.py**

**Purpose:** Extract structured data from unstructured LLM responses using TypedDict and Pydantic.

**Key Concepts:**
- `TypedDict`: Defines expected output structure with type hints
- `Annotated`: Adds descriptions for LLM guidance
- `with_structured_output()`: Constrains LLM to return valid structure
- Ensures JSON-parseable, type-safe responses
- Real-world use: Review analysis, data extraction

**Code Breakdown:**
```python
from typing import TypedDict, Annotated, Optional, Literal

# Define expected output structure
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all key themes in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment: positive or negative"]
    pros: Annotated[Optional[list[str]], "Write down all pros in a list"]
    cons: Annotated[Optional[list[str]], "Write down all cons in a list"]
    name: Annotated[Optional[str], "Name of the reviewer"]

# Create model that returns structured output
structured_model = model.with_structured_output(Review)

# Pass unstructured text
result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra...
[long review text]
""")

# Result is a dict matching Review structure
print(result)
# {
#     'key_themes': ['processor', 'camera', 'battery', ...],
#     'summary': 'Excellent phone but expensive...',
#     'sentiment': 'pos',
#     'pros': ['fast processor', 'great camera', ...],
#     'cons': ['expensive', 'heavy', ...],
#     'name': 'Nitish Singh'
# }
```

**Output Structure:**
```python
{
    'key_themes': ['processor performance', 'camera quality', 'battery life', 'weight'],
    'summary': 'Samsung Galaxy S24 Ultra is a powerful flagship phone with exceptional camera and processor, though expensive and heavy.',
    'sentiment': 'pos',
    'pros': [
        'Insanely powerful processor (great for gaming and productivity)',
        'Stunning 200MP camera with incredible zoom capabilities',
        'Long battery life with fast charging',
        'S-Pen support is unique and useful'
    ],
    'cons': [
        'Weight and size make it uncomfortable for one-handed use',
        'Samsung One UI comes with bloatware',
        '$1,300 price tag is very high'
    ],
    'name': 'Nitish Singh'
}
```

**What Happens:**
1. Defines `Review` TypedDict with annotated fields
2. Creates model constrained to return this structure
3. Passes raw product review text
4. LLM automatically extracts and structures data
5. Returns validated dict with correct types

**Benefits:**
- No manual JSON parsing needed
- Type-safe: guaranteed structure
- Validation: LLM understands field meanings
- Easy to use in downstream processing

---

### **11_pydantic_demo.py**

**Purpose:** Demonstrate Pydantic for data validation, type safety, and serialization.

**Key Concepts:**
- `BaseModel`: Define data schemas with validation
- Field validators: Constraints on values (gt, lt, etc.)
- Type hints: Enforce types
- Email validation: Built-in email checks
- Serialization: Convert to dict/JSON
- Default values: Fallback values if not provided

**Code Breakdown:**
```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# Define Student model with validation
class Student(BaseModel):
    name: str = 'nitish'                          # Default value
    age: Optional[int] = None                     # Optional field
    email: EmailStr                               # Must be valid email
    cgpa: float = Field(
        gt=0,                                     # Greater than 0
        lt=10,                                    # Less than 10
        default=5,                                # Default value
        description='CGPA of the student'         # Documentation
    )

# Create instance with partial data
new_student = {'email': 'example@gmail.com', 'cgpa': 9.5}
student = Student(**new_student)

# Access as attributes
print(student.name)      # 'nitish' (default)
print(student.age)       # None (not provided)
print(student.email)     # 'example@gmail.com'
print(student.cgpa)      # 9.5

# Convert to dict
student_dict = dict(student)
# {'name': 'nitish', 'age': None, 'email': 'example@gmail.com', 'cgpa': 9.5}

# Convert to JSON
student_json = student.model_dump_json()
# '{"name":"nitish","age":null,"email":"example@gmail.com","cgpa":9.5}'

print(student_json)
```

**Validation Rules:**

| Field | Type | Validation | Default |
|-------|------|-----------|---------|
| `name` | `str` | Any string | 'nitish' |
| `age` | `Optional[int]` | Any integer or None | None |
| `email` | `EmailStr` | Valid email format | Required |
| `cgpa` | `float` | 0 < value < 10 | 5 |

**What Happens:**
1. Defines Student model with 4 fields
2. Creates instance with just email and cgpa
3. Name fills from default, age remains None
4. CGPA is validated (must be 0-10)
5. Email is validated (must be valid format)
6. Converts to dict and JSON

**Expected Output:**
```json
{"name":"nitish","age":null,"email":"example@gmail.com","cgpa":9.5}
```

**Validation Examples:**
```python
# Valid
Student(email="user@example.com", cgpa=7.5)
Student(name="John", email="john@example.com", cgpa=8.0)

# Invalid - email format
Student(email="invalid-email", cgpa=7.5)  # ValidationError

# Invalid - cgpa out of range
Student(email="user@example.com", cgpa=15)  # ValidationError (>10)
Student(email="user@example.com", cgpa=-1)  # ValidationError (<0)
```

**Pydantic Benefits:**
- **Type Safety**: Guarantees correct types
- **Validation**: Enforces business rules
- **Documentation**: Field descriptions
- **Serialization**: Easy JSON conversion
- **Error Messages**: Clear validation errors

---

## 🚀 How to Run Each Script

### Basic Execution (Non-Interactive Scripts)

#### 01_chat_models_langchain.py
```bash
python 01_chat_models_langchain.py
```
**Output:** One-line response about Indian cricket

#### 02_emb_models_langchain.py
```bash
python 02_emb_models_langchain.py
```
**Output:** Prints embeddings as lists of floats (384 dimensions each)

#### 03_document_similarity_langchain.py
```bash
python 03_document_similarity_langchain.py
```
**Output:**
```
Query: Who is the best keeper
Best Match: MS Dhoni is famous for his calm leadership and led India to multiple ICC trophies.
similarity score is: 0.89
```

#### 05_prompt_generator.py
```bash
python 05_prompt_generator.py
```
**Output:** Creates `template.json` (no console output)

#### 06_messages.py
```bash
python 06_messages.py
```
**Output:** Prints message list with system, human, and AI messages

#### 08_chat_prompt_template.py
```bash
python 08_chat_prompt_template.py
```
**Output:** Explanation of "googly" from an LLM

#### 10_with_structured_output_pydantic.py
```bash
python 10_with_structured_output_pydantic.py
```
**Output:** Structured dict with themes, summary, sentiment, pros, cons

#### 11_pydantic_demo.py
```bash
python 11_pydantic_demo.py
```
**Output:** JSON serialized student data

---

### Interactive Scripts

#### 07_chatbot.py (Multi-turn Chatbot)
```bash
python 07_chatbot.py
```
**Usage:**
```
you: What is diabetes?
AI: Diabetes is a chronic metabolic disorder...
=============================================

you: What are its symptoms?
AI: Common symptoms include...
=============================================

you: exit
```

#### 09_message_placeholder.py (Customer Support)
```bash
python 09_message_placeholder.py
```
**Usage:**
```
You: I need help with my order
AI: I'll be happy to help you with your order...

You: What's the refund policy?
AI: Our refund policy allows returns within 30 days...

You: exit
```

---

### Web UI Script

#### 04_dynamic_prompt.py (Streamlit App)
```bash
streamlit run 04_dynamic_prompt.py
```
**Access:** http://localhost:8501
**Features:**
- Select research paper
- Choose explanation style
- Pick explanation length
- Click "Summarize" button
- View structured response

---

## 📊 Execution Order for Learning

**Recommended sequence:**

1. `01_chat_models_langchain.py` – Learn basic LLM interaction
2. `06_messages.py` – Understand message types
3. `08_chat_prompt_template.py` – Learn prompt templating
4. `07_chatbot.py` – See multi-turn conversation
5. `02_emb_models_langchain.py` – Learn embeddings
6. `03_document_similarity_langchain.py` – Apply embeddings to search
7. `05_prompt_generator.py` + `04_dynamic_prompt.py` – Build web UI
8. `09_message_placeholder.py` – Advanced templating
9. `11_pydantic_demo.py` – Data validation basics
10. `10_with_structured_output_pydantic.py` – Structured LLM outputs

---

## 🔍 Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution:**
1. Ensure `.env` file exists in project root
2. Add your API key:
   ```
   GROQ_API_KEY=gsk_xxxxx
   ```
3. Ensure `load_dotenv()` is called before using the key

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install langchain langchain-groq langchain-huggingface pydantic python-dotenv scikit-learn
```

### Issue: Model download hanging

**Solution (for embeddings):**
- First run downloads ~100MB from HuggingFace
- Requires internet connection
- Subsequent runs use cached model
- Check progress with verbose output:
  ```bash
  python -u 02_emb_models_langchain.py
  ```

### Issue: Email validation fails in 11_pydantic_demo.py

**Solution:**
```bash
pip install email-validator
```

### Issue: Streamlit not installing

**Solution:**
```bash
pip install streamlit
```

Then run:
```bash
streamlit run 04_dynamic_prompt.py
```

### Issue: Can't connect to Groq API

**Solutions:**
1. Check internet connection
2. Verify API key is valid: https://console.groq.com
3. Check API rate limits (free tier has limits)
4. Ensure `.env` has correct key name: `GROQ_API_KEY`

### Issue: Chat history file not found (09_message_placeholder.py)

**Solution:**
Create `chat_history.txt` in project root:
```bash
touch chat_history.txt
# or manually create with sample content
echo "Sample chat history" > chat_history.txt
```

---

## 📝 Project Structure Insights

### Conceptual Flow

```
LLM Basics (01) 
    ↓
Message Types (06) 
    ↓
Prompt Templates (08)
    ↓
Multi-turn Chat (07, 09)
    
Embeddings (02)
    ↓
Similarity Search (03)

Structured Outputs (10)
    ↓
Data Validation (11)

Advanced UI (04, 05)
```

### Key Design Patterns Used

1. **Chaining** (`prompt | model`)
2. **Message History** (maintaining context)
3. **Templates** (reusable prompts)
4. **Structured Outputs** (type-safe extraction)
5. **Embeddings** (semantic search)

---

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [Groq Console](https://console.groq.com)
- [HuggingFace Models](https://huggingface.co/sentence-transformers)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## 📄 License

This project is part of a learning journey in Generative AI. Adjust as needed for your purposes.

---

**Created:** January 2026 | **Repository:** [gen-ai-langchain-journey](https://github.com/DebjitDebjit2004/gen-ai-langchain-journey)
