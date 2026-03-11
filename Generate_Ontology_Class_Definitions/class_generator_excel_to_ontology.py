import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, SKOS
from rdflib.namespace import DC, DCTERMS

# === Einstellungen ===
excel_path = "klassen.xlsx"
ontology_path = "BWMD_Ontologie+Platon_20251209.rdf"
output_path = "BWMD_Ontologie+Platon_20260204.rdf"
base_uri = "https://www.iwm.fraunhofer.de/ontologies/bwmd-ontology#"

# === RDF Graph laden ===
g = Graph()
g.parse(ontology_path, format="xml")

# === Namespaces extrahieren ===
namespaces = dict(g.namespaces())
namespaces["base"] = Namespace(base_uri)

g.bind("base", namespaces["base"])
g.bind("dc", DC)
g.bind("dcterms", DCTERMS)
g.bind("owl", OWL)
g.bind("skos", SKOS)

# === Excel einlesen ===
df = pd.read_excel(excel_path)
df.columns = [c.strip().lower() for c in df.columns]

# === Hilfsfunktion: sichere String-Reinigung (NaN-fest) ===
def clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

# === Hilfsfunktion: URI aus CURIE oder URI erstellen ===
def resolve_uri(prefixed_value):
    if prefixed_value.startswith("http://") or prefixed_value.startswith("https://"):
        return URIRef(prefixed_value)
    if ":" in prefixed_value:
        prefix, local = prefixed_value.split(":", 1)
        if prefix in namespaces:
            return namespaces[prefix][local]
    return namespaces["base"][prefixed_value]

# === Vorhandene Klassen sammeln ===
vorhandene_klassen = set(str(s) for s in g.subjects(RDF.type, OWL.Class))

# === Statistik-Zähler ===
stats = {
    "rows_total": 0,
    "classes_created": 0,
    "classes_existing": 0,
    "superclass_found": 0,
    "superclass_fallback": 0,
    "labels_de": 0,
    "labels_en": 0,
    "definitions": 0,
}

# === Verarbeitung ===
for index, row in df.iterrows():
    stats["rows_total"] += 1

    class_name = clean_cell(row.get("class_name"))
    superclass_entry = clean_cell(row.get("superclass_name"))
    creator = clean_cell(row.get("dc_creator"))
    label_de = clean_cell(row.get("label_de"))
    label_en = clean_cell(row.get("label_en"))
    definition = clean_cell(row.get("skos_definition"))
    defined_by = clean_cell(row.get("dc_definedby"))

    if not class_name:
        print(f"⚠️  Zeile {index + 2}: keine class_name angegeben – übersprungen.")
        continue

    class_uri = resolve_uri(class_name)

    if str(class_uri) in vorhandene_klassen:
        print(f"🔁 Klasse existiert bereits: {class_name} – wird übersprungen.")
        stats["classes_existing"] += 1
        continue

    print(f"✅ Neue Klasse: {class_name}")
    stats["classes_created"] += 1

    g.add((class_uri, RDF.type, OWL.Class))

    # === Oberklasse behandeln ===
    if superclass_entry:
        superclass_uri = resolve_uri(superclass_entry)

        if (superclass_uri, RDF.type, OWL.Class) in g or (superclass_uri, RDF.type, RDFS.Class) in g:
            g.add((class_uri, RDFS.subClassOf, superclass_uri))
            stats["superclass_found"] += 1
            print(f"  ↳ Oberklasse: {superclass_entry}")
        else:
            local_superclass_name = superclass_entry.split(":")[-1]
            fallback_uri = namespaces["base"][local_superclass_name]
            g.add((class_uri, RDFS.subClassOf, fallback_uri))
            stats["superclass_fallback"] += 1
            print(
                f"  ⚠️  Oberklasse {superclass_entry} nicht gefunden "
                f"- verwendet stattdessen: base:{local_superclass_name}"
            )

    # === Annotationen ===
    if label_de:
        g.add((class_uri, RDFS.label, Literal(label_de, lang="de")))
        stats["labels_de"] += 1

    if label_en:
        g.add((class_uri, RDFS.label, Literal(label_en, lang="en")))
        stats["labels_en"] += 1

    if creator:
        g.add((class_uri, DC.creator, Literal(creator)))

    if definition:
        g.add((class_uri, SKOS.definition, Literal(definition)))
        stats["definitions"] += 1

    if defined_by:
        g.add((class_uri, DCTERMS.definedBy, URIRef(defined_by)))

    vorhandene_klassen.add(str(class_uri))

# === Ausgabe speichern ===
g.serialize(destination=output_path, format="pretty-xml")
print(f"\n💾 Ontologie gespeichert unter: {output_path}")

# === Statistik ausgeben ===
print("\n📊 === Import-Statistik ===")
print(f"📄 Excel-Zeilen gesamt:        {stats['rows_total']}")
print(f"✅ Neue Klassen erzeugt:      {stats['classes_created']}")
print(f"🔁 Klassen bereits vorhanden: {stats['classes_existing']}")
print(f"⬆️  Oberklassen gefunden:     {stats['superclass_found']}")
print(f"⚠️  Fallback-Oberklassen:     {stats['superclass_fallback']}")
print(f"🏷️  Deutsche Labels:          {stats['labels_de']}")
print(f"🏷️  Englische Labels:         {stats['labels_en']}")
print(f"📘 SKOS-Definitionen:         {stats['definitions']}")
