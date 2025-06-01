# hackathon2025

In project root:
Create venv

python3 -m venv .venv

Activate:

source .venv/bin/activate

Copy data folder from vtt zip into AaltoAI_Hackathon

When in venv (prompt)

pip install -r requirements.txt

python3 main.py

you get files:

structured_innovations.json

-   this is our json structured cleaned project database with collaborators and sources. vat ids are inferred from llm model
    for temp_xxx ids.

normalized_entity_glossary.json
resolved_entity_glossary.json

-   work files for saving inferred collaborator data.
    the code skips these steps if these files are present,
    so delete them to see the action.

filtered_innovations.json

deduplicated_innovations.json

merged_innovations.json

-   stages of deduplication where we first filter, then deduplicate and merge sources to generate deduplicated data.

vtt_network_graph.png

-   visualization of VTT's innovation network (lot of data)
