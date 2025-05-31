import json
import re
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Function to load the configuration from a file


def load_config(config_file_path: str = 'azure_config.json'):
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)
    return config

# Function to initialize the LLM


def initialize_llm(deployment_model: str, config_file_path: str = 'azure_config.json') -> AzureChatOpenAI:
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    model_config = config.get(deployment_model)
    if not model_config:
        raise ValueError(f"Model '{deployment_model}' not found in config")

    return AzureChatOpenAI(
        model=deployment_model,
        api_key=model_config['api_key'],
        deployment_name=model_config['deployment'],  # type: ignore[call-arg]
        azure_endpoint=model_config['api_base'],
        api_version=model_config['api_version'],
    )

# Function to detect and filter out irrelevant content


def filter_text(text):
    # Strip extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove any text that is too short or irrelevant
    if len(text) < 80:  # Skip short content that might just be noise
        return None

    return text

# Function to process JSON descriptions and remove exact duplicates


def process_descriptions(data):
    seen_texts = set()  # Set to track duplicate texts by exact matching
    for record in data:
        innovation_id = record.get("innovation_id")
        core_summary = record.get("core_summary")
        participants = record.get("participants")

        filtered_descriptions = []
        for description in record.get("descriptions", []):
            source_text = description.get("text", "")
            filtered_text = filter_text(source_text)

            if filtered_text and filtered_text not in seen_texts:
                filtered_descriptions.append({
                    "source": description.get("source"),
                    "text": filtered_text,
                    "date": description.get("date", "unknown")
                })
                # Add to set to track future duplicates
                seen_texts.add(filtered_text)

        if filtered_descriptions:
            yield {
                "innovation_id": innovation_id,
                "core_summary": core_summary,
                "participants": participants,
                "descriptions": filtered_descriptions
            }

# function to process the extracted texts and remove duplicates


def filter_innovations_file(input_file='structured_innovations.json', output_file='filtered_innovations.json'):
    # Load the configuration (optional, but left here for context if used later)
    config = load_config()

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process and filter descriptions
    filtered_data = list(process_descriptions(data))

    # Save the filtered data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned data saved to {output_file}")
