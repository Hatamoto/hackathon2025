import re
from pathlib import Path
from tqdm import tqdm


def resolve_temp_vat_ids(entity_glossary, model):
    change_log = []
    temp_keys = [k for k in entity_glossary if k.startswith("temp")]

    for temp_id in tqdm(temp_keys, desc="🔁 Resolving temp VAT IDs (domestic + international)"):
        aliases_dict = entity_glossary[temp_id]
        aliases = aliases_dict.get("alias", [])
        alias_list = ', '.join(aliases)

        prompt = (
            f"The following names are aliases or company names possibly from Finland or abroad: {alias_list}. "
            f"If you can identify the company, provide the official VAT ID or registration number, "
            f"preferably including country code (e.g., FI12345678, DE999999999, US 12-3456789, CHE-123.456.789, etc.). "
            f"Only respond with the best guess for the VAT or registration ID, nothing else."
        )

        try:
            response = model.invoke(prompt).content.strip()

            # Match broader international company/VAT registration formats
            match = re.search(
                r'\b('
                # e.g., IT02131140589, GB 08892708, NO 938 910 039 MVA
                r'[A-Z]{2}[-\s]?\d{7,12}([-\s]?\d{1,2})?([-\s]?[A-Z]{2,4})?|'
                # US EIN: 12-3456789
                r'\d{2}[-\s]?\d{7}|'
                # Swiss: CHE-123.456.789
                r'CHE[-\s]?\d{3}\.\d{3}\.\d{3}|'
                # UK/DE like: SC123456, DE999999999
                r'[A-Z]{2,3}\d{6,10}|'
                # fallback plain numeric
                r'\d{9,12}'
                r')\b',
                response,
                re.IGNORECASE
            )

            # Bypass regex for now
            match = response if response else None

            if match:
                # new_vat_id = match.group(0)
                new_vat_id = response.strip()
                if new_vat_id in entity_glossary:
                    existing_aliases = entity_glossary[new_vat_id].setdefault(
                        "alias", [])
                    entity_glossary[new_vat_id]["alias"] = list(
                        set(existing_aliases + aliases))
                else:
                    entity_glossary[new_vat_id] = {"alias": aliases}
                entity_glossary[new_vat_id]["inferred"] = 1
                del entity_glossary[temp_id]
                change_log.append({
                    "replaced": temp_id,
                    "new_vat_id": new_vat_id,
                    "aliases": aliases,
                    "response": response
                })
            else:
                unresolved_id = f"UNRESOLVED_{temp_id[4:].zfill(6)}"
                entity_glossary[unresolved_id] = {
                    "alias": aliases,
                    "inferred": 0
                }
                del entity_glossary[temp_id]
                change_log.append({
                    "replaced": temp_id,
                    "new_vat_id": unresolved_id,
                    "aliases": aliases,
                    "response": response,
                    "note": "⚠️ No valid ID — marked as unresolved"
                })

        except Exception as e:
            change_log.append({
                "replaced": temp_id,
                "new_vat_id": None,
                "aliases": aliases,
                "error": str(e)
            })

    return entity_glossary, change_log
