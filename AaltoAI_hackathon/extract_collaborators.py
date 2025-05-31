import pandas as pd
import os
import json

# Polku kansioon, jossa CSV-tiedostot sijaitsevat
data_folder = "../data/dataframes"

# Tulokset kerätään tähän
results = []
maininnat_per_file = {}

# Seurantalaskurit
kokonaisrivit = 0
vtt_maininnat = 0

# Mahdolliset sarakkeet eri tarkoituksiin
text_fields = ['main_body', 'source_text', 'text_content']
possible_vat_fields = ['vat_id', 'vatid', 'source_company', 'company_name']
possible_url_fields = ['source_url', 'link_source_text', 'link']
possible_date_fields = ['date_published', 'date_obtained', 'date_in_text', 'date_merged']
possible_project_fields = [
    'title', 'project_title', 'relationship_description',
    'source_description', 'target_description', 'project_name',
    'description'
]

# CSV-tiedostojen läpikäynti
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder, filename)
        print(f"Käsitellään: {filename}")
        try:
            df = pd.read_csv(file_path, dtype=str)
            alkuperainen_rivimaara = len(df)
            kokonaisrivit += alkuperainen_rivimaara

            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            text_field = next((col for col in text_fields if col in df.columns), None)
            if not text_field:
                print(f"⚠️ Ei löytynyt sisältökenttää tiedostosta: {filename}")
                continue

            vtt_mentions = df[df[text_field].str.contains("VTT", case=False, na=False)]
            löytyi = len(vtt_mentions)
            vtt_maininnat += löytyi
            maininnat_per_file[filename] = löytyi

            if vtt_mentions.empty:
                print(f"ℹ️ Ei VTT-mainintoja tiedostossa: {filename}")
                continue

            vat_field = next((col for col in possible_vat_fields if col in vtt_mentions.columns), None)
            url_field = next((col for col in possible_url_fields if col in vtt_mentions.columns), None)
            date_field = next((col for col in possible_date_fields if col in vtt_mentions.columns), None)
            project_field = next((col for col in possible_project_fields if col in vtt_mentions.columns), None)

            if not url_field:
                print(f"⚠️ Ei löytynyt linkkikenttää tiedostosta: {filename}")
                continue

            columns = [url_field]
            if vat_field: columns.append(vat_field)
            if date_field: columns.append(date_field)
            if project_field: columns.append(project_field)

            selected = vtt_mentions[columns].copy()

            rename_dict = {}
            if vat_field: rename_dict[vat_field] = "vat_id"
            rename_dict[url_field] = "source_url"
            if date_field: rename_dict[date_field] = "date_published"
            if project_field: rename_dict[project_field] = "project_info"

            selected = selected.rename(columns=rename_dict)

            for col in ["vat_id", "date_published", "project_info"]:
                if col not in selected.columns:
                    selected[col] = "" if col != "vat_id" else "UNKNOWN"

            selected["source_file"] = filename

            column_order = ["vat_id", "source_url", "date_published", "project_info", "source_file"]
            for col in column_order:
                if col not in selected.columns:
                    selected[col] = ""
            selected = selected[column_order]

            results.append(selected)

        except Exception as e:
            print(f"❌ Virhe tiedostossa {filename}: {e}")
        print("-" * 50)

if results:
    final_df = pd.concat(results).drop_duplicates()

    final_df = final_df[final_df["project_info"].str.strip().astype(bool)]

    keywords = ["collaborat", "project", "innovation", "develop", "cooperation", "research"]
    pattern = '|'.join(keywords)
    final_df = final_df[final_df["project_info"].str.contains(pattern, case=False, na=False)]

    # --- UUSI VAIHE: Muuta vat_id jos se ei ole FI-alkuinen ---
    try:
        with open("../data/entity_glossary/entity_glossary.json", encoding="utf-8") as f:
            glossary = json.load(f)

        alias_map = {}
        for vat_id, data in glossary.items():
            for alias in data.get("alias", []):
                alias_map[alias.upper()] = vat_id

        def resolve_vat(vat):
            if isinstance(vat, str) and not vat.startswith("FI"):
                return alias_map.get(vat.strip().upper(), vat)
            return vat

        final_df["vat_id"] = final_df["vat_id"].apply(resolve_vat)

    except Exception as e:
        print(f"❌ Virhe glossary-tiedoston käsittelyssä: {e}")

    final_df["dedup_key"] = (
        final_df["vat_id"].str.lower().fillna("") + "_" +
        final_df["project_info"].str.lower().fillna("").str[:50]
    )

    final_df.to_csv("vtt_collaborators.csv", index=False)

    print(f"\n📊 VTT-maininnat yrityksittäin:")
    print(final_df['vat_id'].value_counts().to_string())

    print(f"\n📂 VTT-maininnat tiedostoittain:")
    for fname, count in maininnat_per_file.items():
        print(f"  {fname}: {count} riviä")

    print(f"\n✅ Käytiin yhteensä {kokonaisrivit} riviä läpi.")
    print(f"🔎 Löydettiin {vtt_maininnat} riviä, joissa mainitaan VTT.")
    print(f"📁 Lopputiedostossa on {len(final_df)} suodatettua VTT-mainintaa.")
    print(f"📀 Tallennettu tiedostoon: vtt_collaborators.csv")

else:
    print(f"\n❌ Käytiin yhteensä {kokonaisrivit} riviä läpi.")
    print("🔍 Ei löytynyt yhtään VTT-mainintaa.")