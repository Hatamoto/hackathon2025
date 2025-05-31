import pandas as pd
import os

# Polku kansioon, jossa CSV-tiedostot sijaitsevat
data_folder = "../data/dataframes"

# Tulokset kerätään tähän
results = []

# Tiedostokohtaiset käsittelytavat
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder, filename)
        print(f"Käsitellään: {filename}")
        try:
            df = pd.read_csv(file_path, dtype=str)

            # --- 1. comp_mentions_vtt_domain.csv ---
            if filename == "comp_mentions_vtt_domain.csv":
                if 'main_body' in df.columns:
                    vtt_mentions = df[df['main_body'].str.contains("VTT", case=False, na=False)]
                    if not vtt_mentions.empty:
                        selected = vtt_mentions[['Vat_id', 'source_url']].drop_duplicates()
                        selected = selected.rename(columns={"Vat_id": "vat_id"})
                        selected['source_file'] = filename
                        results.append(selected)

            # --- 2. df_relationships_vtt_domain.csv ---
            elif filename == "df_relationships_vtt_domain.csv":
                if 'Source Text' in df.columns:
                    vtt_mentions = df[df['Source Text'].str.contains("VTT", case=False, na=False)]
                    if not vtt_mentions.empty:
                        selected = vtt_mentions[['VAT id', 'Link Source Text']].drop_duplicates()
                        selected = selected.rename(columns={"VAT id": "vat_id", "Link Source Text": "source_url"})
                        selected['source_file'] = filename
                        results.append(selected)

            # --- 3. df_relationships_comp_domain.csv ---
            elif filename == "df_relationships_comp_domain.csv":
                if 'Source Text' in df.columns:
                    vtt_mentions = df[df['Source Text'].str.contains("VTT", case=False, na=False)]
                    if not vtt_mentions.empty:
                        selected = vtt_mentions[['Source Company', 'Link Source Text']].drop_duplicates()
                        selected = selected.rename(columns={"Source Company": "vat_id", "Link Source Text": "source_url"})
                        selected['source_file'] = filename
                        results.append(selected)

            # --- 4. vtt_mentions_comp_domain.csv ---
            elif filename == "vtt_mentions_comp_domain.csv":
                if 'text_content' in df.columns:
                    vtt_mentions = df[df['text_content'].str.contains("VTT", case=False, na=False)]
                    if not vtt_mentions.empty:
                        selected = vtt_mentions[['Company name', 'Link']].drop_duplicates()
                        selected = selected.rename(columns={"Company name": "vat_id", "Link": "source_url"})
                        selected['source_file'] = filename
                        results.append(selected)

        except Exception as e:
            print(f"Virhe tiedostossa {filename}: {e}")
        print("-" * 50)

# Yhdistä ja tallenna lopputulos
if results:
    final_df = pd.concat(results).drop_duplicates()
    final_df.to_csv("vtt_collaborators.csv", index=False)
    print(f"\n✅ Löydettiin {len(final_df)} VTT-mainintaa. Tiedot tallennettu 'vtt_collaborators.csv'")
else:
    print("\n❌ Ei löytynyt yhtään VTT-mainintaa.")