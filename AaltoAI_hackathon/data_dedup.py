import json
import re
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Function to load the configuration from a file
def load_config(config_file_path: str = 'data/azure_config.json'):
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)
    return config

# Function to initialize the LLM
def initialize_llm(deployment_model: str, config_file_path: str = 'data/azure_config.json') -> AzureChatOpenAI:
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    model_config = config.get(deployment_model)
    if not model_config:
        raise ValueError(f"Model '{deployment_model}' not found in config")

    return AzureChatOpenAI(
        model=deployment_model,
        api_key=model_config['api_key'],
        deployment_name=model_config['deployment'],
        azure_endpoint=model_config['api_base'],
        api_version=model_config['api_version'],
    )

# Function to detect and filter out irrelevant content
def filter_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 80:  # Skip short content that might just be noise
        return None
    return text

# Function to check if two descriptions are duplicates using AI
def check_if_duplicate(description_1, description_2, model):
    try:
        prompt = f"Are the following two descriptions about the same innovation? If yes, explain why. If not, explain the differences.\n\nDescription 1: {description_1}\n\nDescription 2: {description_2}"
        response = model.invoke(prompt)
        
        response_text = response.content.strip().lower()
        if "same" in response_text or "similar" in response_text:
            return True
        return False
    except Exception as e:
        print(f"Error during AI comparison: {e}")
        return False

# Function to process descriptions in chunks and handle deduplication
def process_descriptions_in_chunks(data, model, chunk_size=20):
    filtered_data = []
    seen_texts = []  # List to track descriptions for de-duplication using AI
    innovation_map = {}  # A map to store potential duplicates and their unified innovation_id
    
    # Process data in chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1) // chunk_size}")
        
        # Process each record in the chunk
        for record in chunk:
            innovation_id = record.get("innovation_id")
            core_summary = record.get("core_summary")
            participants = record.get("participants")
            
            filtered_descriptions = []
            for description in record.get("descriptions", []):
                source_text = description.get("text", "")
                filtered_text = filter_text(source_text)
                
                if filtered_text:
                    is_duplicate = False
                    matched_innovation_id = innovation_id  # Default to current innovation_id

                    # Check if this description is a duplicate
                    for seen_text, seen_id in seen_texts:
                        if check_if_duplicate(filtered_text, seen_text, model):
                            is_duplicate = True
                            matched_innovation_id = seen_id  # Match the ID of the first seen description
                            break
                    
                    if not is_duplicate:
                        # If it's not a duplicate, add the description with its current innovation_id
                        seen_texts.append((filtered_text, innovation_id))
                    else:
                        # If it's a duplicate, we update the innovation_id to the matched one
                        innovation_id = matched_innovation_id

                    # Add the description under the innovation_id
                    filtered_descriptions.append({
                        "source": description.get("source"),
                        "text": filtered_text,
                        "date": description.get("date", "unknown")
                    })

            if filtered_descriptions:
                filtered_data.append({
                    "innovation_id": innovation_id,
                    "core_summary": core_summary,
                    "participants": participants,
                    "descriptions": filtered_descriptions
                })
    
    return filtered_data

# Function to merge descriptions for the same innovation_id
def merge_innovations(filtered_data):
    merged_data = []
    innovation_map = {}

    # Group entries by innovation_id
    for record in filtered_data:
        innovation_id = record["innovation_id"]

        if innovation_id not in innovation_map:
            innovation_map[innovation_id] = {
                "innovation_id": innovation_id,
                "core_summary": record["core_summary"],
                "participants": record["participants"],
                "descriptions": []
            }
        
        # Add descriptions under the same innovation_id
        for description in record["descriptions"]:
            innovation_map[innovation_id]["descriptions"].append(description)
    
    # Convert the innovation map to a list of merged records
    for innovation_id, merged_record in innovation_map.items():
        merged_data.append(merged_record)
    
    return merged_data

# Main function to process and de-duplicate innovations by context using AI
def main():
    # Load the configuration and model
    config = load_config()
    model = initialize_llm(deployment_model='gpt-4.1-mini', config_file_path='data/azure_config.json')
    
    # Read the filtered innovations file
    input_file = 'filtered_innovations.json'
    output_file_dedup = 'deduplicated_innovations.json'
    output_file_merged = 'merged_innovations.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process and de-duplicate descriptions in chunks
    filtered_data = process_descriptions_in_chunks(data, model, chunk_size=20)

    # Save the deduplicated data to a new JSON file
    with open(output_file_dedup, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    # Merge innovations by innovation_id
    merged_data = merge_innovations(filtered_data)

    # Save the merged data to a new JSON file
    with open(output_file_merged, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"✅ De-duplicated data saved to {output_file_dedup}")
    print(f"✅ Merged data saved to {output_file_merged}")

if __name__ == '__main__':
    main()