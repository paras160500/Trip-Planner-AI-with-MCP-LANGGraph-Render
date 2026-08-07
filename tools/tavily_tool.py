# -----------------------------------------------------------------------------------
#                               Import and Init statements
# -----------------------------------------------------------------------------------
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# -----------------------------------------------------------------------------------
#                               Function logic Statements
# -----------------------------------------------------------------------------------

def tavily_search(query : str):
    """
        Searching on the tavily and generate results array
        having title url and snippet(content with it)
        Args:
            query(str) : The question of user
        Returns:
            string of results.
    """
    # Get the response from the client
    response = client.search(query=query , max_results=3)
    # make empty array to store and parse the result
    results = [] 
    # Parsing the results in proper format
    for i, r in enumerate(response['results'] , 1): 
        title = r.get("title" , "Unknown")
        url = r.get("url" , "")
        snippet = r.get("content" , "").strip()
        # Keep only the first 300 char to avoid tons of text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" " , 1)[0] + "..."
        results.append(f"{i}. **{title}**\n {url}\n {snippet}")

    return "\n\n".join(results)