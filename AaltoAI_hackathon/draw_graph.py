import textwrap
import numpy as np
from collections import Counter
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyvis.network import Network
from tqdm import tqdm
from pathlib import Path


def load_participants_from_resolved_json(path="resolved_entity_glossary.json"):
    with open(path, "r", encoding="utf-8") as f:
        glossary_dict = json.load(f)
    return [(vat_id, entry) for vat_id, entry in glossary_dict.items()]


def build_graph_from_innovations(
    json_path="filtered_innovations.json",
    glossary_path="resolved_entity_glossary.json",
    output_path="output/vtt_network_graph.png"
) -> str:

    print("🔧 Building networkx graph from innovations...")

    with open(json_path, encoding="utf-8") as f:
        innovations = json.load(f)
    with open(glossary_path, encoding="utf-8") as f:
        glossary = json.load(f)
    participants = [(vat_id, entry) for vat_id, entry in glossary.items()]

    G = nx.Graph()
    VTT_ID = "FI01111693"
    G.add_node(VTT_ID, label="VTT", type="vtt")

    for item in innovations:
        innovation_id = item["innovation_id"]
        G.add_node(innovation_id, label=innovation_id, type="project")
        G.add_edge(VTT_ID, innovation_id)

        for vat_id, name, *_ in item["participants"]:
            if vat_id == VTT_ID:
                continue
            G.add_node(vat_id, label=name, type="org")
            G.add_edge(innovation_id, vat_id)

    pos = nx.kamada_kawai_layout(G)
    node_colors = []
    labels = {}
    for node, data in G.nodes(data=True):
        labels[node] = data.get("label", node)
        node_type = data.get("type")
        color = {"vtt": "red", "project": "green"}.get(node_type, "skyblue")
        node_colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    nx.draw(G, pos, ax=ax, with_labels=False, node_color=node_colors,
            node_size=600, edge_color="#999")
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    ax.set_title("VTT Innovation Collaboration Network")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_pyvis_innovation_graph(input="filtered_innovations.json", output_html="output/innovation_network.html") -> str:

    print("🔧 Building pyvis interactive innovation network graph...")

    with open(input, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()
    for item in data:
        inn_id = item["innovation_id"]
        G.add_node(inn_id, label=inn_id, group="Innovation")
        for vat, name, *_ in item["participants"]:
            G.add_node(vat, label=name, group="Organization")
            G.add_edge(inn_id, vat, color="pink")

    net = Network(height="750px", width="100%",
                  bgcolor="#ffffff", font_color="black")  # type: ignore
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(output_html, notebook=False)
    return output_html


def build_barplot_graph_from_participants(input="filtered_innovations.json", output_path="output/barplot.png") -> str:

    print("🔧 Building barplot graph from innovations...")

    with open(input, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    multi_source = sum(
        len(set(p[0] for p in i["participants"])) > 1 for i in data)
    multi_dev = sum(len(set(p[1]
                    for p in i["participants"])) > 1 for i in data)

    plt.bar(["Total Innovations", "Multi-Source Innovations", "Multi-Developer Innovations"],
            [total, multi_source, multi_dev], color=["blue", "green", "red"])
    plt.title("Innovation Statistics")
    for i, v in enumerate([total, multi_source, multi_dev]):
        plt.text(i, v * 1.05, str(v), ha='center')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def build_top_organizations_graph(input="filtered_innovations.json", output_path="output/top_orgs.png", top_n=10) -> str:

    print("🔧 Building top organizations graph from innovations...")

    with open(input, encoding="utf-8") as f:
        data = json.load(f)

    org_counter = Counter()
    for i in data:
        for vat, name, *_ in i["participants"]:
            org_counter[name] += 1

    top_orgs = org_counter.most_common(top_n)
    names, counts = zip(*top_orgs)
    colors = cm.RdBu_r(np.linspace(0, 1, len(counts)))  # type: ignore

    plt.barh(names[::-1], counts[::-1], color=colors[::-1])
    plt.xscale("log")
    plt.title("Top Organizations by Innovation Count")
    for i, v in enumerate(counts[::-1]):
        plt.text(v, i, str(v), va='center')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def modify_all_functions_and_generate_html():
    img1 = build_graph_from_innovations()
    img2 = build_barplot_graph_from_participants()
    img3 = build_top_organizations_graph()
    pyvis_html = build_pyvis_innovation_graph()

    with open(pyvis_html, encoding="utf-8") as f:
        pyvis_content = f.read()

    pyvis_head = pyvis_content.split("<head>")[1].split("</head>")[0]
    pyvis_body = pyvis_content.split("<body>")[1].split("</body>")[0]

    html_content = textwrap.dedent(f"""\
        <html>
        <head><title>VTT Innovation Report</title>
                                   {pyvis_head}</head>
        <body>
            <h1>VTT Innovation Report</h1>
            <h2>1. Innovation Collaboration Graph</h2>
            <img src="{img1}" width="100%">
            <h2>2. Innovation Statistics</h2>
            <img src="{img2}" width="100%">
            <h2>3. Top Participating Organizations</h2>
            <img src="{img3}" width="100%">
            <h2>4. Interactive Innovation Network</h2>
            {pyvis_body}
        </body>
        </html>
    """)

    output_html_path = "combined_report.html"
    with open(output_html_path, "w", encoding="utf-8") as out:
        out.write(html_content)

    print(f"✅ Full HTML report saved to {output_html_path}")
