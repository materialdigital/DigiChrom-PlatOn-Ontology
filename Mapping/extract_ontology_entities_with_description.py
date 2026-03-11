import pandas as pd
from rdflib import Graph, RDFS, SKOS, DCTERMS, URIRef, Literal, Namespace
import re

# === EINSTELLUNGEN ===
owl_file = "pmdco-full.ttl"
output_csv = "PMDco_rdf_summary_all_entities.csv"
export_invalid = True  # ungültige Entitäten separat speichern

# === NAMESPACES ===
OBO = Namespace("http://purl.obolibrary.org/obo/")

# === RDF LADEN ===
g = Graph()
g.parse(owl_file, format="ttl")
print(f"✅ Ontologie geladen: {owl_file}, Tripel: {len(g)}")

# === HILFSFUNKTIONEN ===
def extract_label_from_iri(uri_str: str) -> str:
    """Nimmt den letzten Teil der URI nach # oder / und macht CamelCase lesbarer"""
    if not isinstance(uri_str, str):
        return ""
    fragment = re.split(r"[#/]", uri_str)[-1]
    fragment = re.sub(r"([a-z])([A-Z])", r"\1 \2", fragment)  # CamelCase auftrennen
    fragment = fragment.replace("_", " ").strip()
    return fragment

def get_description(entity):
    """Sammelt Kommentare/Definitionen aus verschiedenen Prädikaten"""
    descs = []
    for p in [RDFS.comment, SKOS.definition, DCTERMS.description, OBO["IAO_0000115"]]:
        for _, _, o in g.triples((entity, p, None)):
            if isinstance(o, Literal):
                descs.append(str(o))
    if descs:
        return " | ".join(descs)
    return ""

# === ENTITÄTEN SAMMELN ===
data = []
invalid_entities = []

for s in set(g.subjects()):
    s_str = str(s)

    # nur "echte" URIs behalten
    if not (s_str.startswith("http://") or s_str.startswith("https://")):
        invalid_entities.append(s_str)
        continue

    # Typ bestimmen
    types = list(g.objects(s, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")))
    type_labels = [t.split("#")[-1] if "#" in t else t.split("/")[-1] for t in map(str, types)]
    ent_type = "|".join(type_labels) if type_labels else ""

    # Label suchen – englisch bevorzugt
    labels_en, labels_all = [], []
    for _, _, o in g.triples((s, RDFS.label, None)):
        if isinstance(o, Literal):
            labels_all.append(str(o))
            if o.language == "en":
                labels_en.append(str(o))

    label = labels_en[0] if labels_en else (labels_all[0] if labels_all else "")

    # Wenn kein Label vorhanden → aus IRI generieren
    if not label.strip():
        label = extract_label_from_iri(s_str)

    # Beschreibung sammeln
    desc = get_description(s)

    data.append({
        "Entity": s_str,
        "Type": ent_type,
        "Label": label,
        "Description": desc
    })

# === DATAFRAME SPEICHERN ===
df = pd.DataFrame(data)
df.to_csv(output_csv, sep=";", index=False)
print(f"💾 CSV-Datei gespeichert: {output_csv} ({len(df)} Entitäten)")

# === UNGÜLTIGE ENTITÄTEN SEPARAT ===
if export_invalid and invalid_entities:
    df_invalid = pd.DataFrame({"Invalid_Entities": invalid_entities})
    invalid_file = output_csv.replace(".csv", "_invalid.csv")
    df_invalid.to_csv(invalid_file, sep=";", index=False)
    print(f"⚠️ {len(df_invalid)} ungültige Einträge gespeichert unter: {invalid_file}")

