import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

# Get OpenAI key from env
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat model


def initialize_llm(deployment_model: str = "gpt-4"):
    return ChatOpenAI(
        model=deployment_model,
        api_key=SecretStr(api_key)  # type: ignore[call-arg]
    )


# Initialize model
model = initialize_llm("gpt-4")  # or "gpt-4o", "gpt-3.5-turbo", etc.

# Example use
prompt = "Explain what LangChain is in 1 sentence."
response = model.invoke(prompt)

# Print response content
print(response.content)
