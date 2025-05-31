import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


# This script visualizes the collaboration network of VTT and its innovations

def load_participants_from_resolved_json(path="resolved_entity_glossary.json"):
    with open(path, "r", encoding="utf-8") as f:
        glossary_dict = json.load(f)

    return [(vat_id, entry) for vat_id, entry in glossary_dict.items()]


def build_graph_from_innovations(json_path="structured_innovations.json", glossary_path="resolved_entity_glossary.json", output_path="vtt_network_graph.png"):

    print("Building graph from innovations...")

    # Load your structured JSON
    with open(json_path, encoding="utf-8") as f:
        innovations = json.load(f)

    participants = load_participants_from_resolved_json(glossary_path)

    G = nx.Graph()

    # Add central node for VTT
    VTT_ID = "FI01111693"  # assuming this is VTT's VAT ID
    G.add_node(VTT_ID, label="VTT", type="vtt")

    # Add nodes and edges
    # Add nodes and edges
    for item in tqdm(innovations[:50], desc="Building graph from innovations"):
        innovation_id = item["innovation_id"][:30]  # short label
        G.add_node(innovation_id, label=innovation_id, type="project")
        G.add_edge(VTT_ID, innovation_id)

        for vat_id, info in participants:
            if vat_id == VTT_ID:
                continue
            aliases = info
            # use first alias or VAT ID as label
            if isinstance(aliases, dict):
                alias_list = aliases.get("alias", [])
            else:
                alias_list = aliases

            label = alias_list[0] if alias_list else vat_id

            G.add_node(vat_id, label=label, type="org")
            G.add_edge(innovation_id, vat_id)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # or nx.kamada_kawai_layout(G)

    node_colors = []
    labels = {}

    for node, data in G.nodes(data=True):
        labels[node] = data["label"]
        if data["type"] == "vtt":
            node_colors.append("red")
        elif data["type"] == "project":
            node_colors.append("green")
        else:
            node_colors.append("skyblue")

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=False, node_color=node_colors,
            node_size=600, edge_color="#999")
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("🧠 VTT Innovation Collaboration Network")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
