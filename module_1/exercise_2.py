from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    SQLiteSession,
    set_tracing_disabled,
)


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
            model=model,
            name="General Agent",
            instructions="Be helpful",
        )

    async def run(self, user_input: str, session: SQLiteSession) -> str:
        result = await Runner.run(self._agent, user_input, session=session)
        return result.final_output


async def main() -> None:
    agent = GeneralAgent()
    session = SQLiteSession("conversation_123")
    await session.add_items(
        [
            {"role": "user", "content": "Hello, who won the FIFA World Cup in 2018?"},
            {"role": "assistant", "content": "France won the 2018 FIFA World Cup."},
        ]
    )

    print("Calling model...")
    reply = await agent.run("Who was the captain?", session)
    print(reply)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
