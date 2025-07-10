# Helper packages
import asyncio
import os
from typing import Annotated, TypedDict, Union, List
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tensorlake import DocumentParserOptions, document_markdown_tool


load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TENSORLAKE_API_KEY = os.getenv("TENSORLAKE_API_KEY")


def build_document_analysis_prompt(parsed_result: str, questions: Union[str, List[str]]) -> str:
    # Normalize single question to list
    if isinstance(questions, str):
        questions = [questions]

    question_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

    system_prompt = f"""You are a helpful assistant that answers questions about documents with signature detection data.

Your responsibilities:
1. Answer questions based on that loaded data
2. Help users understand the signature analysis results

You can answer questions like:
- How many signatures were found?
- Which pages contain signatures?
- Who signed the document?
- What does the content say around signatures?
- What type of document is this?
- Who are the parties involved?
- What is the date of the signature?
- Did each party sign the document?
- Are there any missing signatures on any pages?
- Which property is missing signatures?
- Who is the agent for the properties missing signatures?

I've processed a document and got this result:
{parsed_result}

Please analyze the above parsed output and answer the following:
{question_block}
"""
    return system_prompt


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Agent node - decides whether to use tools
async def agent_node(state: AgentState):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    ).bind_tools([document_markdown_tool])

    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


# Conditional Logic for Tool Use
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# LangGraph Workflow Setup
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([document_markdown_tool]))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile()


# Document + Agent Pipeline
async def analyze_signatures_agents(
        path: str,
        questions: List[str]
) -> str:
    """Invoke the tool with parsing options, then use agent for analysis."""

    print("🔍 Processing document with signature detection...")

    # Pass parsing options and run the tool
    parsed_output = await document_markdown_tool.ainvoke({
        "path": path,
        "options": DocumentParserOptions(detect_signature=True)
    })

    # Build prompt
    prompt = build_document_analysis_prompt(parsed_output, questions)

    # Run agent on prompt
    final_state = await app.ainvoke({
        "messages": [HumanMessage(content=prompt)]
    })

    return final_state["messages"][-1].content


async def example_signature_detection_real_estate():

    # change to your own file path
    path = "path/to/your/document.pdf"

    analysis_questions = [
        "How many signatures were detected in this document and who are the parties involved?",
        "What contextual information can you extract about any signatures?",
        "Are there any missing signatures on any pages?"
    ]

    result = await analyze_signatures_agents(
        path=path,
        questions=analysis_questions
    )

    print("Analysis Result:\n\n", result)


if __name__ == "__main__":
    # run the example
    asyncio.run(example_signature_detection_real_estate())