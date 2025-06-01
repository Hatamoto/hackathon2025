#!/usr/bin/env python
# coding: utf-8

from utils import write_json_and_track, has_file_changed
from draw_graph import modify_all_functions_and_generate_html
from data_dedup import run_deduplication_pipeline
from data_filter import filter_innovations_file
from data_loop import dedup_until_converged
from data_valid import run_final_validation
from resolve_vat import resolve_temp_vat_ids
import re
import matplotlib.pyplot as plt
import networkx as nx
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
from pathlib import Path
import subprocess

outputToFile = False
if outputToFile:
    sys.stdout = open("output.log", "w")
    sys.stderr = sys.stdout

load_dotenv()

df = pd.read_csv('data/dataframes/vtt_mentions_comp_domain.csv')
df = df[df['Website'].str.startswith('www.')]
df['source_index'] = df.index

print(f"DF with content from {len(df)} websites of {len(df['Company name'].unique())} different companies ")

class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, str] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: str
    source_type: str
    target: str
    target_type: str
    type: str
    properties: Dict[str, str] = Field(default_factory=dict)

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, str] = Field(default_factory=dict)

class GraphDocument(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    source_document: Optional[Document] = None

path = 'data/graph_docs/'
index = 0
with open(os.path.join(path, os.listdir(path)[index]), 'rb') as doc:
    graph_doc = pickle.load(doc)

print(f"Example custom graph document:\n\n {graph_doc} \n\n ")

for doc in graph_doc:
    for node in doc.nodes:
        print(f"- {node.id} ({node.type})    :   {node.properties['description']}")

for doc in graph_doc:
    for relationship in doc.relationships:
        print(f"- {relationship.source} ({relationship.source_type}) - {relationship.type} -> {relationship.target} ({relationship.target_type})    :    description: {relationship.properties['description']}")

path = 'data/graph_docs_names_resolved/'
index = 0
with open(os.path.join(path, os.listdir(path)[index]), 'rb') as doc:
    graph_doc = pickle.load(doc)

for doc in graph_doc:
    for node in doc.nodes[:3]:
        print(f"- {node.id} ({node.type})    :   {node.properties['description']}")

for doc in graph_doc:
    for relationship in doc.relationships[:3]:
        print(f"- {relationship.source} ({relationship.source_type}) - {relationship.type} -> {relationship.target} ({relationship.target_type})    :    description: {relationship.properties['description']}")

df_relationships_comp_url = pd.DataFrame(index=None)
with tqdm(total=len(df), desc="Entities resolved") as pbar:
    for i, row in df.iterrows():
        try:
            Graph_Docs = pickle.load(open(os.path.join('data/graph_docs_names_resolved/', f"{row['Company name'].replace(' ','_')}_{i}.pkl"), 'rb'))[0]
            node_description = {}
            node_en_id = {}
            for node in Graph_Docs.nodes:
                node_description[node.id] = node.properties['description']
                node_en_id[node.id] = node.properties['english_id']

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

            df_relationships_comp_url = pd.concat([df_relationships_comp_url, pd.DataFrame(relationship_rows, index=None)], ignore_index=True)
        except:
            continue
        pbar.update(1)

df_relationships_vtt_domain = pd.DataFrame(index=None)
df_vtt_domain = pd.read_csv('data/dataframes/comp_mentions_vtt_domain.csv')
with tqdm(total=len(df_vtt_domain), desc="Entities resolved") as pbar:
    for index_source, row in df_vtt_domain.iterrows():
        try:
            Graph_Docs = pickle.load(open(os.path.join('data/graph_docs_vtt_domain_names_resolved/', f"{row['Vat_id'].replace(' ','_')}_{index_source}.pkl"), 'rb'))[0]
            node_description = {}
            node_en_id = {}
            for node in Graph_Docs.nodes:
                node_description[node.id] = node.properties['description']
                node_en_id[node.id] = node.properties['english_id']

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

            df_relationships_vtt_domain = pd.concat([df_relationships_vtt_domain, pd.DataFrame(relationship_rows, index=None)], ignore_index=True)
        except:
            continue
        pbar.update(1)

print("Loading Azure OpenAI API credentials...")
config_file_path = './azure_config.json'

def initialize_llm(deployment_model: str, config_file_path: str = 'data/azure_config.json') -> AzureChatOpenAI:
    with open(config_file_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    config["gpt-4o-mini"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_4O_M")
    config["gpt-4.1-mini"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_41_M")
    config["gpt-4.1"]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_41")
    config["gpt-4o-mini"]["api_base"] = os.getenv("AZURE_OPENAI_BASE_URL_4O_M")
    config["gpt-4.1-mini"]["api_base"] = os.getenv("AZURE_OPENAI_BASE_URL_41_M")
    config["gpt-4.1"]["api_base"] = os.getenv("AZURE_OPENAI_BASE_URL_41")

    model_config = config.get(deployment_model)
    return AzureChatOpenAI(
        model=deployment_model,
        api_key=model_config['api_key'],
        deployment_name=model_config['deployment'],
        azure_endpoint=model_config['api_base'],
        api_version=model_config['api_version'],
    )

model41m = initialize_llm(deployment_model='gpt-4.1-mini', config_file_path='data/keys/azure_config.json')
print(f"Successfully initialized Azure OpenAI model: {model41m.model_name}")

resolved_path = "resolved_entity_glossary.json"
changelog_path = "vat_resolution_log.json"
if os.path.exists(resolved_path) and os.path.exists(changelog_path) and not has_file_changed("data/entity_glossary/entity_glossary.json"):
    print("Using cached resolved glossary")
    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_glossary = json.load(f)
    with open(changelog_path, "r", encoding="utf-8") as f:
        changes = json.load(f)
else:
    print("Running VAT resolution via LLM")
    with open("data/entity_glossary/entity_glossary.json", "r", encoding="utf-8") as f:
        entity_glossary = json.load(f)
    resolved_glossary, changes = resolve_temp_vat_ids(entity_glossary, model41m)
    write_json_and_track(resolved_path, resolved_glossary)
    write_json_and_track(changelog_path, changes)

glossary_df = pd.DataFrame([
    {
        "vat_id": vat_id,
        "alias": alias.strip(),
        "inferred": info.get("inferred", 0)
    }
    for vat_id, info in resolved_glossary.items()
    for alias in info.get("alias", [])
])
glossary_df["alias_norm"] = glossary_df["alias"].str.lower().str.strip()

structured_path = "structured_innovations.json"
filtered_path = "filtered_innovations.json"

if not (os.path.exists(structured_path) and os.path.getsize(structured_path) > 0):
    print("Running innovation structuring pipeline...")
    all_df = pd.concat([df_relationships_comp_url, df_relationships_vtt_domain], ignore_index=True)
    innovation_links = all_df[(all_df["source type"] == "Innovation") | (all_df["target type"] == "Innovation")].copy()

    def extract_innovation_group(row):
        if row["source type"] == "Innovation":
            return pd.Series({
                "innovation_id": row["source english_id"],
                "innovation_desc": row["source description"],
                "partner_id": row["target english_id"],
                "partner_desc": row["target description"],
                "partner_type": row["target type"],
                "relationship": row["relationship type"],
                "source_text": row["Source Text"][:5000],
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
                "source_text": row["Source Text"][:5000],
                "source_link": row["Link Source Text"],
            })

    flat_df = innovation_links.apply(extract_innovation_group, axis=1)
    flat_df = flat_df.dropna(subset=["innovation_id"])
    grouped = flat_df.groupby("innovation_id")
    output = []

    for innovation_id, group in tqdm(grouped, desc="Processing innovations"):
        participants = []
        vat_ids_found = 0
        unknown_counter = 1
        for name in group[group["partner_type"] == "Organization"]["partner_id"].dropna().unique():
            name_norm = name.lower().strip()
            match = glossary_df[glossary_df["alias_norm"] == name_norm]
            vat_id = match["vat_id"].iloc[0] if not match.empty else f"UNKNOWN_{unknown_counter:06d}"
            if not match.empty:
                vat_ids_found += 1
            else:
                unknown_counter += 1
            participants.append([vat_id, name])

        descriptions = group[["source_link", "source_text"]].drop_duplicates().rename(columns={"source_link": "source", "source_text": "text"}).to_dict(orient="records")
        output.append({
            "innovation_id": innovation_id,
            "core_summary": group["innovation_desc"].iloc[0],
            "participants": participants,
            "descriptions": descriptions,
        })

    write_json_and_track("structured_innovations.json", output)
    print(f"Saved {len(output)} unique innovation objects to structured_innovations.json")
else:
    print("Skipping innovation structuring pipeline, structured_innovations.json already exists")

if not (os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0):
    print("Running innovation filtering pipeline...")
    filter_innovations_file(input_file="structured_innovations.json")
else:
    print("Skipping filtering pipeline, filtered_innovations.json already exists")

if not (os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0):
    print("Running deduplication pipeline...")
    dedup_until_converged(max_iter=10)
else:
    print("Skipping deduplication, filtered_innovations.json already exists")

if True:
    print("Drawing graphs...")
    modify_all_functions_and_generate_html()
    print("Graphs generated and saved to html.")
