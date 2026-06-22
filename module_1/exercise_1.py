from openai import AsyncOpenAI

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled


def main():
    set_tracing_disabled(True)
    model = OpenAIChatCompletionsModel(
        model="qwen3:8b",
        openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    )
    agent = Agent(name="Assistant", instructions="You are helpful.", model=model)
    print(Runner.run_sync(agent, "Tell me about your model").final_output)

if __name__ == "__main__":
    main()
