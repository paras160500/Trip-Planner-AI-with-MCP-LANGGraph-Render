# -----------------------------------------------------------------------------------
#                               Import and Init statements
# -----------------------------------------------------------------------------------

import os 
import asyncio
import certifi
from dotenv import load_dotenv
load_dotenv()
from langchain_mcp_adapters.client import MultiServerMCPClient

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = MultiServerMCPClient(
    {
        "tavily" : {
            "transport" : "streamable_http",
            "url" : f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        }
    }
)

# -----------------------------------------------------------------------------------
#                               Import and Init statements
# -----------------------------------------------------------------------------------

async def get_all_tools():
    """
        Getting information of all available tools 
    """
    tools = await client.get_tools()
    print("Available Tools")
    for tool in tools:
        print(tool.name)

tavily_search_tool = None 

async def get_tavily_search_tool():
    """
        Among all the available tools only getting the 
        tavily search tool
    """
    global tavily_search_tool
    if tavily_search_tool is not None:
        return 

    tools = await client.get_tools()
    print("Available MCP Tools :- ")
    for tool in tools:
        print(tool.name)

    tavily_search_tool = next(
        tool for tool in tools if tool.name == "tavily_search"
    )


async def tavily_mcp_search(query : str):
    """
        Function will call the tavily_search tool with a query
        in the backend.py file
    """
    await get_tavily_search_tool()
    result = await tavily_search_tool.ainvoke(
        {
            "query" : query
        }
    )
    # print(result)
    return result