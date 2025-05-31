#!/usr/bin/env python
# coding: utf-8

# ### VTT challenge: Innovation Ambiguity
#
# #### Source Dataframe
# - Websites from finnish companies that mention 'VTT' on their website
# - `Orbis ID`, also `VAT id` is a unique identifier for organizations, later used to merge different alias of the same organization to one unique id

# 1. original source dataframe
import hashlib
import os
import sys
import pickle
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
import sys

# redirect stdout and stderr to a file
sys.stdout = open("output.log", "w")
sys.stderr = sys.stdout

# load environment variables
load_dotenv()

df = pd.read_csv('data/dataframes/vtt_mentions_comp_domain.csv')
df = df[df['Website'].str.startswith('www.')]
df['source_index'] = df.index

print(
    f"DF with content from {len(df)} websites of {len(df['Company name'].unique())} different companies ")
df.head(10)


# ##### End-to-End relationship extraction
# - Based on the above website content, entities of the type `Organization` and `Innovation` are extracted, as well as their type of relationship
# - `Collaboration` between Organization and `Developed_by` between Innovation and Organization
# - The relationships are stored in a custom object as displayed below:

# 2.1. example of custom python object of data


class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str  # unique identifier for node of type 'Organisation', else 'name provided by llm' for type: 'Innovation'
    type: str  # allowed node types: 'Organization', 'Innovation'
    properties: Dict[str, str] = Field(default_factory=dict)


class Relationship(BaseModel):
    """Represents a relationship between two nodes in the knowledge graph."""
    source: str
    source_type: str  # allowed node types: 'Organization', 'Innovation'
    target: str
    target_type: str  # allowed node types: 'Organization', 'Innovation'
    type: str  # allowed relationship types: 'DEVELOPED_BY', 'COLLABORATION'
    properties: Dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    page_content: str  # manually appended - source text of website
    # metadata including source URL and document ID
    metadata: Dict[str, str] = Field(default_factory=dict)


class GraphDocument(BaseModel):
    """Represents a complete knowledge graph extracted from a document."""
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    source_document: Optional[Document] = None


# 2.2 loading example of custom graph document
# example file naming convention
print(
    "The extracted graph documents are saved as f'{df['Company name'].replace(' ','_')}_{df['source_index'].pkl}.pkl, under data/graph_docs/ \n")

for i, row in df[:3].iterrows():
    print(
        f"{i}: 'data/graph_docs/' + {row['Company name'].replace(' ','_')}_{row['source_index']}.pkl")


# 2.3 loading example of custom graph document

path = 'data/graph_docs/'
index = 0

# load graph document
with open(os.path.join(path, os.listdir(path)[index]), 'rb') as doc:
    graph_doc = pickle.load(doc)

print(f"Example custom graph document:\n\n {graph_doc} \n\n ")

print("Example custom graph document nodes :\n")
for doc in graph_doc:
    for node in doc.nodes:
        print(
            f"- {node.id} ({node.type})    :   {node.properties['description']}")

print("\nExample custom graph document relationships:\n")
for doc in graph_doc:
    for relationship in doc.relationships:
        print(
            f"- {relationship.source} ({relationship.source_type}) - {relationship.type} -> {relationship.target} ({relationship.target_type})    :    description: {relationship.properties['description']}")


# #### Name ambiguity resolution
# - within the source text, variation/ alias of organization name lead to ambiguity
# - this ambiguity is partly solved by mapping organization to a unique identifier: `VAT ID`
# - the dict: `entity_glossary` stores Ids and Alias as key-value pairs


# 3. load entity glossary
entity_glossary = json.load(
    open('data/entity_glossary/entity_glossary.json', 'r', encoding='utf-8'))
print(entity_glossary.get('FI26473754'))


# 2.3 loading example of custom graph document

path = 'data/graph_docs_names_resolved/'
index = 0

# load graph document
with open(os.path.join(path, os.listdir(path)[index]), 'rb') as doc:
    graph_doc = pickle.load(doc)

print(f"Example custom graph document:\n\n {graph_doc} \n\n ")

print("Example custom graph document nodes :\n")
for doc in graph_doc:
    for node in doc.nodes[:3]:
        print(
            f"- {node.id} ({node.type})    :   {node.properties['description']}")

print("\nExample custom graph document relationships:\n")
for doc in graph_doc:
    for relationship in doc.relationships[:3]:
        print(
            f"- {relationship.source} ({relationship.source_type}) - {relationship.type} -> {relationship.target} ({relationship.target_type})    :    description: {relationship.properties['description']}")


# transform graph document into dataframe


df_relationships_comp_url = pd.DataFrame(index=None)

with tqdm(total=len(df), desc="Entities resolved") as pbar:
    for i, row in df.iterrows():
        try:
            Graph_Docs = pickle.load(open(os.path.join('data/graph_docs_names_resolved/',
                                     f"{row['Company name'].replace(' ','_')}_{i}.pkl"), 'rb'))[0]  # load graph doc

            node_description = {}  # unique identifier
            node_en_id = {}
            for node in Graph_Docs.nodes:
                node_description[node.id] = node.properties['description']
                node_en_id[node.id] = node.properties['english_id']

            # get relationship triplets
            relationship_rows = []
            for i in range(len(Graph_Docs.relationships)):

                relationship_rows.append({
                    "Document number": row['source_index'],
                    "Source Company": row["Company name"],
                    "relationship description": Graph_Docs.relationships[i].properties['description'],
                    "source id": Graph_Docs.relationships[i].source,
                    "source type": Graph_Docs.relationships[i].source_type,
                    "source english_id": node_en_id.get(Graph_Docs.relationships[i].source, None),
                    "source description": node_description.get(Graph_Docs.relationships[i].source, None),
                    "relationship type": Graph_Docs.relationships[i].type,
                    "target id": Graph_Docs.relationships[i].target,
                    "target type": Graph_Docs.relationships[i].target_type,
                    "target english_id": node_en_id.get(Graph_Docs.relationships[i].target, None),
                    "target description": node_description.get(Graph_Docs.relationships[i].target, None),
                    "Link Source Text": row["Link"],
                    "Source Text": row["text_content"],
                })

            df_relationships_comp_url = pd.concat([df_relationships_comp_url, pd.DataFrame(
                relationship_rows, index=None)], ignore_index=True)

        except:
            continue

        pbar.update(1)

df_relationships_comp_url.head(5)


# #### Innovation and Collaboration disclosure on VTT-domain
# - in addition to the discussion of VTT contribution on company websites, the second datasource includes websites under the vtt domain that discuss collaboration with other companies
# - the list of source urls is provided under `data/dataframes/comp_mentions_vtt_domain.vsc`
# - the extract relationships as custom objects are provided under `data/dataframes/graph_docs_vtt_domain`
# - the extract relationships with organization resolution under `data/dataframes/graph_docs_vtt_domain`


# transform graph document into dataframe

df_relationships_vtt_domain = pd.DataFrame(index=None)
df_vtt_domain = pd.read_csv('data/dataframes/comp_mentions_vtt_domain.csv')

with tqdm(total=len(df_vtt_domain), desc="Entities resolved") as pbar:
    for index_source, row in df_vtt_domain.iterrows():
        try:
            Graph_Docs = pickle.load(open(os.path.join('data/graph_docs_vtt_domain_names_resolved/',
                                     f"{row['Vat_id'].replace(' ','_')}_{index_source}.pkl"), 'rb'))[0]  # load graph doc

            node_description = {}  # unique identifier
            node_en_id = {}
            for node in Graph_Docs.nodes:
                node_description[node.id] = node.properties['description']
                node_en_id[node.id] = node.properties['english_id']

            # get relationship triplets
            relationship_rows = []
            for i in range(len(Graph_Docs.relationships)):

                relationship_rows.append({
                    "Document number": index_source,
                    "VAT id": row["Vat_id"],
                    "relationship description": Graph_Docs.relationships[i].properties['description'],
                    "source id": Graph_Docs.relationships[i].source,
                    "source type": Graph_Docs.relationships[i].source_type,
                    "source english_id": node_en_id.get(Graph_Docs.relationships[i].source, None),
                    "source description": node_description.get(Graph_Docs.relationships[i].source, None),
                    "relationship type": Graph_Docs.relationships[i].type,
                    "target id": Graph_Docs.relationships[i].target,
                    "target type": Graph_Docs.relationships[i].target_type,
                    "target english_id": node_en_id.get(Graph_Docs.relationships[i].target, None),
                    "target description": node_description.get(Graph_Docs.relationships[i].target, None),
                    "Link Source Text": row["source_url"],
                    "Source Text": row["main_body"],
                })

            df_relationships_vtt_domain = pd.concat([df_relationships_vtt_domain, pd.DataFrame(
                relationship_rows, index=None)], ignore_index=True)

        except:
            continue

        pbar.update(1)

df_relationships_vtt_domain.head(5)


# #### assess to OpenAI endpoint
# - for this challenge we want to provide you access to OpenAI models: 4o-mini, 4.1 or 4.1-mini
# - `ASK @ VTT-stand for key :)`

# 4. load api access credentials

config_file_path = './azure_config.json'


def initialize_llm(deployment_model: str, config_file_path: str = 'data/azure_config.json') -> AzureChatOpenAI:
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    # Set the environment variable that AzureOpenAI expects
    # os.environ["AZURE_OPENAI_API_KEY_4O"] = cfg["api_key"]
    print("Loaded config:", config)

    print("Setting keys and endpoints for Azure OpenAI models...")

    config["gpt-4o-mini"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_4O_M")
    config["gpt-4.1-mini"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_41_M")
    config["gpt-4.1"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_41")
    config["gpt-4o-mini"]["api_base"] = os.getenv("AZURE_OPENAI_BASE_URL_4O_M")
    config["gpt-4.1-mini"]["api_base"] = os.getenv(
        "AZURE_OPENAI_BASE_URL_41_M")
    config["gpt-4.1"]["api_base"] = os.getenv(
        "AZURE_OPENAI_BASE_URL_41")

    model_config = config.get(deployment_model)
    if not model_config:
        raise ValueError(f"Model '{deployment_model}' not found in config")

    print("Model config:", model_config)
    print(f"Using Azure OpenAI model: {deployment_model}")
    print(f"Using Azure OpenAI endpoint: {model_config['api_base']}")
    print(f"Using Azure OpenAI API version: {model_config['api_version']}")
    print(f"Using Azure OpenAI deployment name: {model_config['deployment']}")
    print(f"Using Azure OpenAI API key: {model_config['api_key'][:4]}...")

    # Use the correct nested config in the return
    return AzureChatOpenAI(
        model=deployment_model,
        api_key=model_config['api_key'],
        deployment_name=model_config['deployment'],  # type: ignore[call-arg]
        azure_endpoint=model_config['api_base'],
        api_version=model_config['api_version'],
    )


# initialize
model4om = initialize_llm(deployment_model='gpt-4o-mini',
                          config_file_path='data/keys/azure_config.json')
model41m = initialize_llm(deployment_model='gpt-4.1-mini',
                          config_file_path='data/keys/azure_config.json')
model41 = initialize_llm(deployment_model='gpt-4.1',
                         config_file_path='data/keys/azure_config.json')

# example use:
prompt = ''
model4om.invoke(prompt).content
model41m.invoke(prompt).content
model41.invoke(prompt).content


# exit()

# structured = []

# # Ensure df is your cleaned, combined dataframe
# for (vat_id, url), group in df.groupby(["vat_id", "source_url"]):
#     full_text = " ".join(group["text"].astype(str).tolist())

#     prompt = f"""Summarize the following article and list all organizations involved in the innovation.

# TEXT:
# {full_text[:4000]}"""  # truncate for safety

#     response = model41.invoke(prompt).content

#     # Very simple way to extract entities and summary
#     # In production, you'd parse this more safely
#     summary = response.split("\n")[0].strip()
#     participants = [line.strip("-• ").strip()
#                     for line in response.split("\n")[1:] if line.strip()]

#     innovation_id = hashlib.md5(f"{vat_id}{url}".encode()).hexdigest()[:8]

#     structured.append({
#         "innovation_id": innovation_id,
#         "core_summary": summary,
#         "participants": participants,
#         "descriptions": [
#             {
#                 "source": row["source_url"],
#                 "text": row["text"][:500],  # Optional truncation
#                 "date": row.get("date", "unknown")
#             }
#             for _, row in group.iterrows()
#         ]
#     })

# with open("vtt_structured_innovations.json", "w") as f:
#     json.dump(structured, f, indent=2)

# Extract innovation-centric entries from both DataFrames
all_df = pd.concat(
    [df_relationships_comp_url, df_relationships_vtt_domain], ignore_index=True)

# Filter relationships that involve Innovations
innovation_links = all_df[
    (all_df["source type"] == "Innovation") | (
        all_df["target type"] == "Innovation")
].copy()

# Normalize the data to have 'innovation_id', 'partner_id', etc.


def extract_innovation_group(row):
    if row["source type"] == "Innovation":
        return pd.Series({
            "innovation_id": row["source english_id"],
            "innovation_desc": row["source description"],
            "partner_id": row["target english_id"],
            "partner_desc": row["target description"],
            "partner_type": row["target type"],
            "relationship": row["relationship type"],
            "source_text": row["Source Text"][:500],
            "source_link": row["Link Source Text"],
        })
    else:
        return pd.Series({
            "innovation_id": row["target english_id"],
            "innovation_desc": row["target description"],
            "partner_id": row["source english_id"],
            "partner_desc": row["source description"],
            "partner_type": row["source type"],
            "relationship": row["relationship type"],
            "source_text": row["Source Text"][:500],
            "source_link": row["Link Source Text"],
        })


flat_df = innovation_links.apply(extract_innovation_group, axis=1)
# ensure innovations are valid
flat_df = flat_df.dropna(subset=["innovation_id"])

# Group into final structured list
grouped = flat_df.groupby("innovation_id")
output = []

for innovation_id, group in grouped:
    participants = list(group[group["partner_type"] ==
                        "Organization"]["partner_desc"].dropna().unique())
    descriptions = group[["source_link", "source_text"]].drop_duplicates().rename(
        columns={"source_link": "source", "source_text": "text"}
    ).to_dict(orient="records")

    output.append({
        "innovation_id": innovation_id,
        "core_summary": group["innovation_desc"].iloc[0],
        "participants": participants,
        "descriptions": descriptions,
    })

# Write to JSON
with open("structured_innovations.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(
    f"✅ Saved {len(output)} unique innovation objects to structured_innovations.json")
