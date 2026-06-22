import json
import asyncio

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_tracing_disabled


class GeneralAgent:
    def __init__(self) -> None:
        model = OpenAIChatCompletionsModel(
            model="qwen3:8b",
            openai_client=AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
        )
        set_tracing_disabled(True)
        self._agent = Agent(
            name="General Agent", instructions="Answer questions reliably", model=model
        )

    async def run(self, user_input: str) -> str:
        response = await Runner.run(
            self._agent,
            user_input,
        )
        return response.final_output


def _create_house_string() -> str:
    output = ""
    with open("module_1/house_data.json", "r") as file:
        data = json.load(file)

    for house in data["house_data"]:
        output += (
            f"House located at {house['address']}, {house['city']}, {house['state']} {house['zip']} with "
            f"{house['bedrooms']} bedrooms, {house['bathrooms']} bathrooms, "
            f"{house['square_feet']} sq ft area, priced at ${house['price']}, "
            f"built in {house['year_built']}.\n"
        )

    return output


def _generate_prompt(query):
    houses_layout = _create_house_string()

    prompt = f"""
    Use the following houses information to answer users queries.
    {houses_layout}
    Query: {query}    
                """
    return prompt


async def main():
    agent = GeneralAgent()
    prompt = _generate_prompt("What is the most expensive house?")

    print("calling agent...")
    print(await agent.run(prompt))


if __name__ == "__main__":
    asyncio.run(main())
