import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, SecretStr
import json
from tqdm import tqdm
import pickle
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
model = ChatOpenAI(
    model="gpt-4o", api_key=SecretStr(openai_api_key), temperature=0.0)
#!/usr/bin/env python
# coding: utf-8

# ### VTT challenge: Innovation Ambiguity
#
# #### Source Dataframe
# - Websites from finnish companies that mention 'VTT' on their website
# - `Orbis ID`, also `VAT id` is a unique identifier for organizations, later used to merge different alias of the same organization to one unique id


# 1. original source dataframe
df = pd.read_csv('data/dataframes/vtt_mentions_comp_domain.csv')
df = df[df['Website'].str.startswith('www.')]
df['source_index'] = df.index

print(
    f"DF with content from {len(df)} websites of {len(df['Company name'].unique())} different companies ")
df.head(3)


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

# Initialize the OpenAI chat model
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


def initialize_llm(deployment_model: str = "gpt-4"):
    return ChatOpenAI(
        model=deployment_model,
        # ensures compatibility with Pydantic
        api_key=SecretStr(openai_api_key)  # type: ignore[call-arg]
    )


# Initialize a model (pick one model only)
model = initialize_llm("gpt-4")  # or "gpt-4o", "gpt-3.5-turbo", etc.

# Example usage
prompt = "Explain what LangChain is in one sentence."
response = model.invoke(prompt)
print(response.content)
