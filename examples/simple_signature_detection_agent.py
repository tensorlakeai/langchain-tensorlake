from langchain_tensorlake import document_markdown_tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

# This is using a sample real estate document with signatures
# You can replace this with any PDF document containing signatures
path = "https://pub-226479de18b2493f96b64c6674705dd8.r2.dev/real-estate-purchase-all-signed.pdf"

question = f"How many signatures are found in this whole documents the document found at {path}?"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[document_markdown_tool],
    prompt="I have a document that needs to be parsed. Please parse it and answer the question.",
    name="real-estate-agent",
)

result = agent.invoke({"messages": [{"role": "user", "content": question}]})
print(result["messages"][-1].content)
