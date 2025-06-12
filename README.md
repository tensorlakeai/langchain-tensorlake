# langchain-tensorlake

`langchain-tensorlake` provides seamless integration between [Tensorlake](https://tensorlake.ai) and [LangChain](https://langchain.com), enabling you to build sophisticated document processing agents with structured extraction workflows.

---

## Installation

```bash
pip install -U langchain-tensorlake
```

---

## Quick Start

### 1. Set up your environment

You should configure credentials for Tensorlake and OpenAI by setting the following environment variables:
```
export TENSORLAKE_API_KEY="your-tensorlake-api-key"
export OPENAI_API_KEY = "your-openai-api-key"
```

Get your Tensorlake API key from the [Tensorlake Cloud Console](https://cloud.tensorlake.ai/). New users get 100 free credits!

### 2. Do the necessary imports

```python
from langchain_tensorlake import DocumentParserOptions, document_markdown_tool
from langgraph.prebuilt import create_react_agent
import asyncio
import os
```

### 3. Build a Signature Detection Agent

```python
async def main(question):
    # Create the agent with the Tensorlake tool
    agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[document_markdown_tool],
            prompt=(
                """
                I have a document that needs to be parsed. \n\nPlease parse this document and answer the question about it.
                """
            ),
            name="real-estate-agent",
        )
    
    # Run the agent
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # Print the result
    print(result["messages"][-1].content)
```

### 4. Example Usage

```python
# Define the path to the document to be parsed
path = "path/to/your/document.pdf"

# Define the question to be asked and create the agent
question = f"What contextual information can you extract about the signatures in my document found at {path}?"

if __name__ == "__main__":
    asyncio.run(main(question))
```

---

## Customization

You can configure how documents are parsed using DocumentParserOptions, such as:
- `chunking_strategy`: fragment, page, or section
- `detect_tables`: enable or disable table extraction
- `detect_signatures`: flag signature pages 
- `extract_structured`: define a schema for structured output
