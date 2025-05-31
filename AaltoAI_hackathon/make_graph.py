import os
import json


# Visualize the collaboration network of VTT and its innovations
# Load your structured JSON
with open("structured_innovations.json", encoding="utf-8") as f:
    innovations = json.load(f)

G = nx.Graph()

# Add central node for VTT
VTT_ID = "FI01111693"  # assuming this is VTT's VAT ID
G.add_node(VTT_ID, label="VTT", type="vtt")

# Add nodes and edges
for item in innovations:
    innovation_id = item["innovation_id"][:30]  # short label
    G.add_node(innovation_id, label=innovation_id, type="project")
    G.add_edge(VTT_ID, innovation_id)

    for vat in item["participants"]:
        if vat == VTT_ID:
            continue
        G.add_node(vat, label=vat, type="org")
        G.add_edge(innovation_id, vat)

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
plt.savefig("vtt_network_graph.png", dpi=300)
plt.show()
