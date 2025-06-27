# tools.py
from langchain_core.tools import tool
from datetime import datetime
import requests
from pydantic import BaseModel
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client
import uuid
import os
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )

@tool
def generate_image(query: str) -> str:
    """ Generates an image based on a query and returns the output path"""
    # Initialize the client
    stability_api = client.StabilityInference(
        key="",  # Get from https://dreamstudio.ai/
        engine="stable-diffusion-xl-1024-v1-0",  # Engine for SDXL
    )

    responses = stability_api.generate(
        prompt= query,
        steps=30,  # Reduce for faster results (min: 20)
    )
    for resp in responses:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                return "Content violation detected!"
            elif artifact.type == generation.ARTIFACT_IMAGE:
                filename = f"output_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join("generated_images", filename)
                os.makedirs("generated_images", exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(artifact.binary)

                return f"http://localhost:8001/generated_images/{filename}"

                # return output_path  # 👈 returned to final_answer



@tool
def add(x: float, y: float) -> float:
    """Add numbers and return the result. Only use this tool for addition."""
    return x + y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract two numbers and return the result. Only use this tool for subtraction."""
    return y - x

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers and return the result. Only use this tool for multiplciation."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raises the base to the power of the exponent (i.e., base ** exponent). Use this tool for exponentiation or power calculations like '2 to the power of 3'."""
    return x ** y

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_location_from_ip():
    """Get the geographical location based on the IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            return (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {data.get('city', 'N/A')},\n"
                f"Country: {data.get('country', 'N/A')}"
            )
        return "Location could not be determined."
    except Exception as e:
        return f"Error occurred: {e}"



@tool
def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web. This tool can be used to find current events or latest news."""
    params = {
        "api_key": "2dc4dd1213236548179def254c3960f129f7b9ece98061cab8ebee4dbf950227",
        "engine": "google",
        "q": query,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json()
    return [Article.from_serpapi_result(result) for result in results.get("organic_results", [])]

@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
     The answer should be in natural language as this will be provided
     to the user directly. The tools_used must include a list of tool
     names that were used within the `scratchpad`. You MUST use this tool
      to conclude the interaction.
    """
    return {"answer": answer, "tools_used": tools_used}

# List of tools to import 
TOOLS = [generate_image,add, subtract, multiply, exponentiate, get_current_datetime, get_location_from_ip, final_answer, serpapi]
