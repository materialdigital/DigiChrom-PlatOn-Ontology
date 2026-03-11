import pandas as pd
from rdflib import Graph, URIRef, RDFS, OWL

# === 1. Lade Excel-Datei ===
df = pd.read_excel("BWMD_PMDco_LLM_Mapping_Enhanced.xlsx")

# === 2. Neues RDF-Graph-Objekt ===
g = Graph()

# === 3. Fehlende oder ungültige Einträge zählen ===
skipped_rows = []

# === 4. Jede Zeile verarbeiten ===
for idx, row in df.iterrows():

    bwmd_uri = str(row.get("BWMD_Entity", "")).strip()
    pmdco_uri = str(row.get("PMDco_Entity", "")).strip()
    relation = str(row.get("Relation", "")).strip().lower()

    # Nur verarbeiten, wenn beides echte URIs sind
    if not (bwmd_uri.startswith("http://") or bwmd_uri.startswith("https://")):
        skipped_rows.append({"row": idx, "reason": f"BWMD ungültig: {bwmd_uri}"})
        continue

    if not (pmdco_uri.startswith("http://") or pmdco_uri.startswith("https://")):
        skipped_rows.append({"row": idx, "reason": f"PMDco ungültig: {pmdco_uri}"})
        continue

    bwmd = URIRef(bwmd_uri)
    pmdco = URIRef(pmdco_uri)

    # === CLASS RELATIONS ===
    if relation in ["equivalentclass", "equivalent_class", "equivalentclassof"]:
        g.add((bwmd, OWL.equivalentClass, pmdco))

    elif relation in ["subclassof", "subclass_of", "subclass"]:
        g.add((bwmd, RDFS.subClassOf, pmdco))

    # === PROPERTY RELATIONS ===
    elif relation in ["equivalentproperty", "equivalent_property"]:
        g.add((bwmd, OWL.equivalentProperty, pmdco))

    elif relation in ["subpropertyof", "subproperty_of"]:
        g.add((bwmd, RDFS.subPropertyOf, pmdco))

    # === UNKNOWN ===
    else:
        skipped_rows.append({"row": idx, "reason": f"Unbekannte Relation: {relation}"})

# === 5. RDF-Datei speichern ===
output_file = "BWMD_PMDco_mapping.ttl"
g.serialize(destination=output_file, format="turtle")

print(f"✅ Mapping-Datei erstellt: {output_file}")
print(f"📄 Enthaltene Tripel: {len(g)}")

# === 6. Bericht über übersprungene Zeilen ===
if skipped_rows:
    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df.to_csv("BWMD_PMDco_skipped_rows.csv", sep=";", index=False)
    print(f"⚠️ {len(skipped_rows)} Zeilen übersprungen – siehe BWMD_PMDco_skipped_rows.csv")
else:
    print("✅ Keine Zeilen übersprungen – alles gültig!")
