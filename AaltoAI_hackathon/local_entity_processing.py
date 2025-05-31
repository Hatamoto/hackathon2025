# 0. example of custom python object of data
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str # unique identifier for node of type 'Organisation', else 'name provided by llm' for type: 'Innovation'
    type: str # allowed node types: 'Organization', 'Innovation'
    properties: Dict[str, str] = Field(default_factory=dict)

class Relationship(BaseModel):
    """Represents a relationship between two nodes in the knowledge graph."""
    source: str
    source_type: str # allowed node types: 'Organization', 'Innovation'
    target: str 
    target_type: str # allowed node types: 'Organization', 'Innovation'
    type: str # allowed relationship types: 'DEVELOPED_BY', 'COLLABORATION'
    properties: Dict[str, str] = Field(default_factory=dict)

class Document(BaseModel):
    page_content:str # manually appended - source text of website
    metadata: Dict[str, str] = Field(default_factory=dict) # metadata including source URL and document ID

class GraphDocument(BaseModel):
    """Represents a complete knowledge graph extracted from a document."""
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    source_document: Optional[Document] = None
