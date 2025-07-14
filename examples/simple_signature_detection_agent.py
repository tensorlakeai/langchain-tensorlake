# 1. Import the langchain-tensorlake tool
import os
from langgraph.prebuilt import create_react_agent
from langchain_tensorlake import document_markdown_tool

# 2. Load the environment variables 
os.environ["TENSORLAKE_API_KEY"] = "TENSORLAKE_API_KEY_PLACEHOLDER"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY_PLACEHOLDER"

# 3. Define the path to the document to be parsed
document_path = "https://pub-226479de18b2493f96b64c6674705dd8.r2.dev/real-estate-purchase-all-signed.pdf"

# 4. Define the question to be asked and create the agent
question = f"What contextual information can you extract about the signatures in my document found at {document_path}?"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[document_markdown_tool],
    prompt="I have a document that needs to be parsed. Please parse it and answer the question.",
    name="real-estate-agent",
)

result = agent.invoke({"messages": [{"role": "user", "content": question}]})
print(result["messages"][-1].content)