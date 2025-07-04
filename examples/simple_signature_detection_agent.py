# 1. Import the langchain-tensorlake tool
import asyncio
import os

from langgraph.prebuilt import create_react_agent

from langchain_tensorlake import document_markdown_tool

# 2. Load the environment variables 
os.environ["TENSORLAKE_API_KEY"] = "TENSORLAKE_API_KEY_PLACEHOLDER"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY_PLACEHOLDER"

# 3. Define the path to the document to be parsed
document_path = os.path.join(os.path.dirname(__file__), "nova_juno_agreement.pdf")

# 4. Define the question to be asked and create the agent
question = f"What contextual information can you extract about the signatures in my document found at {document_path}?"

async def main():
    # 5. Create the agent with the Tensorlake tool
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
    
    # 6. Run the agent
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # 7. Print the result
    print(result["messages"][-1].content)   
    
if __name__ == "__main__":
    asyncio.run(main())
