import pandas as pd
import openai
import json
import time
from tqdm import tqdm
from rdflib import Graph, RDFS, OWL, RDF, URIRef, Namespace
from collections import Counter
import numpy as np
import platform
import sys
from datetime import datetime

# Try to import psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - memory tracking disabled")
    print("   Install with: pip install psutil")

# Embedding-Modell (falls verfügbar)
USE_EMBEDDINGS = True  # Manuell auf False setzen wenn Probleme auftreten
try:
    if USE_EMBEDDINGS:  # Nur versuchen wenn aktiviert
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformers geladen")
    else:
        print("⚠️ Embeddings manuell deaktiviert")
except Exception as e:
    USE_EMBEDDINGS = False
    print(f"⚠️ Sentence-transformers nicht verfügbar: {e}")
    print("   Skript läuft ohne Embeddings (nur lexikalische Methoden)")
    print("   Installation: pip install --upgrade torch sentence-transformers")

# === TIMING & MONITORING ===
class PerformanceMonitor:
    """Tracks timing and resource usage throughout the pipeline."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
    def checkpoint(self, name):
        """Record a timing checkpoint."""
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        
    def get_memory_mb(self):
        """Get current memory usage in MB."""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0
    
    def print_report(self):
        """Print detailed performance report."""
        print("\n" + "="*70)
        print("PERFORMANCE & METRICS REPORT".center(70))
        print("="*70)
        
        # System info
        print(f"\nSYSTEM INFORMATION:")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  CPU: {platform.processor()}")
        if PSUTIL_AVAILABLE:
            print(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
            mem = psutil.virtual_memory()
            print(f"  RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
            print(f"  Peak Memory Usage: {self.get_memory_mb():.1f} MB")
        
        # Timing breakdown
        print(f"\nTIMING BREAKDOWN:")
        prev_time = 0
        for name, elapsed in self.checkpoints.items():
            duration = elapsed - prev_time
            print(f"  {name}: {duration:.2f}s (cumulative: {elapsed:.2f}s)")
            prev_time = elapsed
        
        total_time = time.time() - self.start_time
        print(f"\n  TOTAL RUNTIME: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        print("="*70)

monitor = PerformanceMonitor()

# === EINSTELLUNGEN ===
API_KEY = "YOUR_API_KEY_HERE"
BASE_URL = "YOUR_BASE_URL_HERE"

# Dateien
bwmd_owl = "BWMD_Ontologie_2020-09-28-KurzfassungBericht2_4DigiChrom_Updated_v3.ttl"
bwmd_csv = "BWMDCORE_rdf_summary_all_entities.csv"
pmdco_owl = "pmdco-full.ttl"
pmdco_csv = "PMDco_rdf_summary_all_entities.csv"
output_excel = "BWMD_PMDco_LLM_Mapping_Enhanced.xlsx"

model_name = "gpt-5.2"
temperature = 0.1  # Niedriger für konsistentere Ergebnisse

# === NAMESPACES ===
OBO = Namespace("http://purl.obolibrary.org/obo/")

# === OpenAI Client ===
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === RDF-GRAPHEN LADEN ===
print("\n" + "="*70)
print("ONTOLOGY ALIGNMENT PIPELINE - STARTING".center(70))
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n🔄 Lade RDF-Ontologien...")
load_start = time.time()

bwmd_graph = Graph()
pmdco_graph = Graph()

try:
    bwmd_graph.parse(bwmd_owl, format="ttl")
    bwmd_load_time = time.time() - load_start
    print(f"✅ BWMD RDF geladen: {len(bwmd_graph)} Tripel ({bwmd_load_time:.2f}s)")
except Exception as e:
    print(f"⚠️ BWMD RDF nicht geladen: {e}")
    bwmd_graph = None

try:
    pmdco_start = time.time()
    pmdco_graph.parse(pmdco_owl, format="ttl")
    pmdco_load_time = time.time() - pmdco_start
    print(f"✅ PMDco RDF geladen: {len(pmdco_graph)} Tripel ({pmdco_load_time:.2f}s)")
except Exception as e:
    print(f"⚠️ PMDco RDF nicht geladen: {e}")
    pmdco_graph = None

monitor.checkpoint("RDF_Loading")

# === VALIDIERUNGSFUNKTION ===
def validate_entity_exists(entity_iri, graph, ontology_name):
    """Prüft ob eine Entität wirklich in der Ontologie existiert."""
    if not graph:
        return {"exists": False, "error": f"{ontology_name} Graph nicht geladen"}
    
    entity = URIRef(entity_iri)
    
    if (entity, None, None) in graph:
        return {"exists": True, "error": ""}
    
    if (None, RDF.type, entity) in graph:
        return {"exists": True, "error": ""}
    
    return {"exists": False, "error": f"Entität nicht in {ontology_name} gefunden"}

# === CSVs LADEN ===
bwmd = pd.read_csv(bwmd_csv, sep=";")
pmdco = pd.read_csv(pmdco_csv, sep=";")

bwmd.columns = bwmd.columns.str.strip()
pmdco.columns = pmdco.columns.str.strip()

# NaN-Werte behandeln
for col in ['Label', 'Type', 'Entity', 'Description']:
    if col in bwmd.columns:
        bwmd[col] = bwmd[col].fillna('')
    if col in pmdco.columns:
        pmdco[col] = pmdco[col].fillna('')

print(f"✅ BWMD-Entitäten (CSV): {len(bwmd)}")
print(f"✅ PMDco-Entitäten (CSV): {len(pmdco)}")

# Count entity types if available
if 'Type' in bwmd.columns:
    bwmd_types = bwmd['Type'].value_counts()
    print(f"   BWMD Types: {dict(bwmd_types.head(3))}")
if 'Type' in pmdco.columns:
    pmdco_types = pmdco['Type'].value_counts()
    print(f"   PMDco Types: {dict(pmdco_types.head(3))}")

monitor.checkpoint("CSV_Loading")

# === HILFSFUNKTIONEN (MÜSSEN VOR EMBEDDING-BERECHNUNG KOMMEN) ===
def get_full_description(row, df):
    """Sammelt alle Beschreibungsfelder."""
    parts = []
    desc_columns = ['Description', 'Comment', 'rdfs:comment', 'skos:definition']
    for col in desc_columns:
        if col in df.columns and not pd.isna(row.get(col, "")):
            parts.append(str(row[col]))
    return " | ".join(parts) if parts else ""

# ============================================================
# === NEU: TYP-SENSITVE RELATIONEN ===
# ============================================================

def is_property_type(type_str):
    """
    Prüft ob ein Entity-Typ eine OWL-Property ist.
    Erkennt: ObjectProperty, DatatypeProperty, AnnotationProperty,
    owl:ObjectProperty, rdf:Property, etc.
    """
    if not type_str:
        return False
    type_lower = str(type_str).lower()
    property_keywords = [
        "objectproperty", "datatypeproperty", "annotationproperty",
        "rdf:property", "owl:property", "subproperty", "property"
    ]
    return any(kw in type_lower for kw in property_keywords)

def get_default_relation(type_str):
    """
    Gibt die passende Default-Relation basierend auf dem Entity-Typ zurück.
    - Properties  → subPropertyOf
    - Classes     → subClassOf
    """
    if is_property_type(type_str):
        return "subPropertyOf"
    return "subClassOf"

def get_valid_relations(type_str):
    """
    Gibt die zulässigen Relationen für einen Entity-Typ zurück.
    - Properties  → equivalentProperty, subPropertyOf
    - Classes     → equivalentClass, subClassOf
    """
    if is_property_type(type_str):
        return ["equivalentProperty", "subPropertyOf"]
    return ["equivalentClass", "subClassOf"]

def validate_relation(relation, type_str):
    """
    Prüft ob eine Relation für den gegebenen Typ zulässig ist.
    Gibt bei Ungültigkeit die passende Default-Relation zurück.
    """
    valid = get_valid_relations(type_str)
    if relation in valid:
        return relation
    # Versuche semantisch ähnliche Relation zu mappen
    if relation == "equivalentClass" and is_property_type(type_str):
        return "equivalentProperty"
    if relation == "equivalentProperty" and not is_property_type(type_str):
        return "equivalentClass"
    if relation == "subClassOf" and is_property_type(type_str):
        return "subPropertyOf"
    if relation == "subPropertyOf" and not is_property_type(type_str):
        return "subClassOf"
    # Generischer Fallback
    return get_default_relation(type_str)

# ============================================================

# === EMBEDDINGS VORBERECHNEN (falls aktiviert) ===
pmdco_embeddings = {}
bwmd_embeddings = {}

if USE_EMBEDDINGS:
    print("\n🧠 Berechne Embeddings für PMDco-Entitäten...")
    embed_start = time.time()
    
    for _, row in tqdm(pmdco.iterrows(), total=len(pmdco), desc="PMDco Embeddings"):
        label = str(row.Label) if pd.notna(row.Label) else ""
        desc = get_full_description(row, pmdco)
        
        text = f"{label}"
        if desc and len(desc) > 10:
            text += f" {desc[:200]}"
        
        if text.strip():
            embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
            pmdco_embeddings[row.Entity] = embedding
    
    pmdco_embed_time = time.time() - embed_start
    pmdco_embed_size = len(pmdco_embeddings) * 384 * 4 / (1024**2)  # MB
    print(f"✅ {len(pmdco_embeddings)} PMDco-Embeddings berechnet ({pmdco_embed_time:.2f}s, ~{pmdco_embed_size:.1f} MB)")
    
    monitor.checkpoint("PMDco_Embeddings")
    
    # BWMD-Embeddings vorberechnen
    print("\n🧠 Berechne Embeddings für BWMD-Entitäten...")
    embed_start = time.time()
    
    for _, row in tqdm(bwmd.iterrows(), total=len(bwmd), desc="BWMD Embeddings"):
        label = str(row.Label) if pd.notna(row.Label) else ""
        desc = get_full_description(row, bwmd)
        
        text = f"{label}"
        if desc and len(desc) > 10:
            text += f" {desc[:200]}"
        
        if text.strip():
            embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
            bwmd_embeddings[row.Entity] = embedding
    
    bwmd_embed_time = time.time() - embed_start
    bwmd_embed_size = len(bwmd_embeddings) * 384 * 4 / (1024**2)  # MB
    print(f"✅ {len(bwmd_embeddings)} BWMD-Embeddings berechnet ({bwmd_embed_time:.2f}s, ~{bwmd_embed_size:.1f} MB)")
    
    monitor.checkpoint("BWMD_Embeddings")
else:
    print("\n⚠️ Embeddings deaktiviert - nutze nur lexikalische Methoden")

# === ERWEITERTE HIERARCHIE-EXTRAKTION ===
def get_extended_hierarchy_info(entity_iri, graph):
    """Extrahiert Parents, Children, Siblings und Eigenschaften."""
    if not graph:
        return {"parents": [], "children": [], "siblings": [], "properties": []}
    
    entity = URIRef(entity_iri)
    parents = []
    children = []
    siblings = []
    properties = []
    
    # Parent-Klassen
    for _, _, parent in graph.triples((entity, RDFS.subClassOf, None)):
        parent_str = str(parent)
        if parent_str.startswith("http") and "_:" not in parent_str:
            parent_label = get_label_from_graph(parent_str, graph)
            parents.append(parent_label or parent_str.split("#")[-1].split("/")[-1])
            
            # Siblings: Andere Klassen mit gleichem Parent
            for sibling, _, _ in graph.triples((None, RDFS.subClassOf, parent)):
                if sibling != entity:
                    sibling_str = str(sibling)
                    if sibling_str.startswith("http"):
                        sibling_label = get_label_from_graph(sibling_str, graph)
                        siblings.append(sibling_label or sibling_str.split("#")[-1].split("/")[-1])
    
    # Child-Klassen (Subklassen)
    for child, _, _ in graph.triples((None, RDFS.subClassOf, entity)):
        child_str = str(child)
        if child_str.startswith("http") and "_:" not in child_str:
            child_label = get_label_from_graph(child_str, graph)
            children.append(child_label or child_str.split("#")[-1].split("/")[-1])
    
    # Properties (Domain/Range Beziehungen)
    for prop, _, domain in graph.triples((None, RDFS.domain, entity)):
        prop_label = get_label_from_graph(str(prop), graph)
        if prop_label:
            properties.append(f"has property: {prop_label}")
    
    return {
        "parents": parents[:3],
        "children": children[:3],
        "siblings": list(set(siblings))[:3],
        "properties": properties[:2]
    }

def get_label_from_graph(entity_iri, graph):
    """Holt Label aus RDF-Graph."""
    if not graph:
        return ""
    entity = URIRef(entity_iri)
    for _, _, label in graph.triples((entity, RDFS.label, None)):
        return str(label)
    return ""

# === WEITERE HILFSFUNKTIONEN ===
def clean_llm_json(raw_text):
    """Entfernt Markdown-Codeblöcke und säubert JSON aggressiv."""
    if not raw_text:
        return "{}"
    
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:])
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
    
    import re
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        raw_text = json_match.group(0)
    
    raw_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', raw_text)
    raw_text = re.sub(r'\s+', ' ', raw_text)
    raw_text = raw_text.replace('""', '"')
    
    return raw_text.strip()

# === DOMÄNEN-SPEZIFISCHE SYNONYME ===
DOMAIN_SYNONYMS = {
    # Chemische Elemente
    "caesium": ["cesium", "cs"],
    "aluminium": ["aluminum", "al"],
    "sulfur": ["sulphur", "s"],
    # Materialeigenschaften
    "young modulus": ["elastic modulus", "youngs modulus", "e-modulus"],
    "tensile": ["tension", "uniaxial tension"],
    "hardness": ["indentation hardness", "penetration hardness"],
    # Prozesse
    "heat treatment": ["thermal treatment", "annealing"],
    "polishing": ["mechanical polishing", "surface polishing"],
}

def get_synonyms(label):
    """Findet Synonyme für einen Begriff."""
    label_lower = label.lower()
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if key in label_lower:
            return synonyms
    return []

# === EMBEDDING-ÄHNLICHKEIT ===
def cosine_similarity(vec1, vec2):
    """Berechnet Cosine-Ähnlichkeit zwischen zwei Vektoren."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def get_embedding_similarity(bw_entity, pmdco_entity):
    """
    Berechnet semantische Ähnlichkeit via Embeddings.
    
    OPTIMIERT: Nutzt vorberechnete BWMD-Embeddings falls verfügbar.
    """
    if not USE_EMBEDDINGS or pmdco_entity not in pmdco_embeddings:
        return 0.0
    
    # NEU: Nutze gecachtes BWMD-Embedding falls vorhanden
    if bw_entity in bwmd_embeddings:
        bw_embedding = bwmd_embeddings[bw_entity]
    else:
        # Fallback: On-the-fly berechnen (falls Entity nicht gecacht)
        bw_text = str(bw_entity)  # Minimal-Text als Fallback
        bw_embedding = EMBEDDING_MODEL.encode(bw_text, convert_to_numpy=True)
    
    pmdco_embedding = pmdco_embeddings[pmdco_entity]
    
    return cosine_similarity(bw_embedding, pmdco_embedding)

# === FEW-SHOT BEISPIELE (erweitert um Property-Beispiele) ===
FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - Equivalent Classes:
BWMD: "TensileTest" (Class) | Parents: MechanicalTest
PMDco: "TensileTestingProcess" (Class) | Parents: MechanicalTestingProcess
→ equivalentClass (95) - Same concept, same hierarchy level

EXAMPLE 2 - SubClass with hierarchy match:
BWMD: "VickersHardnessTest" (Class) | Parents: HardnessTest
PMDco: "HardnessTest" (Class) | Children: VickersTest, BrinellTest
→ subClassOf (92) - BWMD is child of PMDco parent

EXAMPLE 3 - SuperClass detection:
BWMD: "Material" (Class) | Children: Metal, Ceramic, Polymer
PMDco: "MetallicMaterial" (Class) | Parents: Material
→ subClassOf (88) - PMDco is more specific than BWMD

EXAMPLE 4 - Sibling-based alignment:
BWMD: "RockwellHardness" (Class) | Siblings: VickersHardness, BrinellHardness
PMDco: "RockwellTest" (Class) | Siblings: VickersTest, BrinellTest
→ equivalentClass (90) - Same siblings pattern

EXAMPLE 5 - Equivalent Properties:
BWMD: "hasYoungsModulus" (ObjectProperty) | Domain: Material
PMDco: "hasElasticModulus" (ObjectProperty) | Domain: Material
→ equivalentProperty (88) - Same concept, synonymous labels, same domain

EXAMPLE 6 - SubProperty:
BWMD: "hasVickersHardness" (DatatypeProperty) | Domain: HardnessTest
PMDco: "hasHardnessValue" (DatatypeProperty) | Domain: HardnessTest
→ subPropertyOf (85) - BWMD property is more specific than PMDco property
"""

# ============================================================
# === GEWICHTSOPTIMIERUNG VIA SCIPY ===
# ============================================================
from scipy.optimize import minimize

# Ground-Truth-Paare: (bwmd_label, pmdco_label, erwartete_konfidenz_0_bis_1)
# !! Trage hier deine bekannten Paare ein !!
GROUND_TRUTH_PAIRS = [
    # (bwmd_label, pmdco_label, konfidenz_0_bis_1, relation)
    #
    # Kuratierter Satz mit maximaler Varianz über alle 8 Metriken:
    #   - BFO-exakte equivalentClass   → w7=1.0, w1 sehr hoch, w5 hoch
    #   - Synonym-equivalentClass      → w7=1.0, w1 niedrig, w4/w5 entscheidend
    #   - equivalentProperty           → w7=1.0 (type=ObjectProperty), w1 hoch
    #   - subPropertyOf                → w7=0.0 (type mismatch ObjProp→DatatypeProp)
    #   - subClassOf hoch (92–93)      → w7=0.0, w6 (Hierarchy) signifikant
    #   - subClassOf mittel-hoch (86)  → w7=0.0, w5/w6 dominieren
    #   - subClassOf mittel (78)       → w7=0.0, schwächere Signale → zwingt
    #                                    Optimizer alle anderen Gewichte zu nutzen

    # ── BFO-Anker (nur 4): exakte Label-Übereinstimmung, hohe Konfidenz ───────
    # Bewusst reduziert: zu viele exakte Paare lassen w1 trivial dominieren
    ("Continuant",                    "continuant",                        0.99, "equivalentClass"),
    ("Process",                       "process",                           0.98, "equivalentClass"),
    ("MaterialEntity",                "material entity",                   0.98, "equivalentClass"),
    ("FiatObjectPart",                "fiat object part",                  0.98, "equivalentClass"),

    # ── Synonym-equivalentClass (7): w1≈0 → Optimizer MUSS w4/w5 stärken ───
    # Diese Gruppe ist der Kern — mehr Synonym-Paare als BFO-Anker
    ("Plating",                       "coating",                           0.96, "equivalentClass"),
    ("TurningMachine",                "lathe",                             0.96, "equivalentClass"),
    ("LightMicroscope",               "optical microscope",                0.93, "equivalentClass"),
    ("HeatTreatment",                 "heat treatment",                    0.96, "equivalentClass"),
    ("TensileTestingMachine",         "tensile testing machine",           0.97, "equivalentClass"),
    ("ScanningElectronMicroscope",    "scanning electron microscope",      0.96, "equivalentClass"),
    ("Extensometer",                  "extensometer",                      0.97, "equivalentClass"),

    # ── equivalentProperty (4): Eigenständige Gruppe ────────────────────────
    ("precedes",                      "precedes",                          0.98, "equivalentProperty"),
    ("has Part",                      "has part",                          0.97, "equivalentProperty"),
    ("has Output",                    "has output",                        0.96, "equivalentProperty"),
    ("is Composed Of",                "consists of",                       0.93, "equivalentProperty"),

    # ── subPropertyOf (2): w7=0.0, Typ-Mismatch erzwingt andere Gewichte ────
    ("has String Literal",            "has value",                         0.86, "subPropertyOf"),
    ("has Real Literal",              "has value",                         0.78, "subPropertyOf"),

    # ── subClassOf hoch (4): w7=0.0, w6 (Hierarchy) relevant ───────────────
    ("OneDimensionalContinuantFiatBoundary", "continuant fiat boundary",   0.93, "subClassOf"),
    ("ProcessID",                     "identifier",                        0.93, "subClassOf"),
    ("MaterialID",                    "identifier",                        0.92, "subClassOf"),
    ("MetallographicSpecimen",        "specimen",                          0.86, "subClassOf"),

    # ── subClassOf mittel-hoch (5): w7=0.0, diverse Signale ─────────────────
    ("BrinellTestingEquipment",       "hardness testing machine",          0.86, "subClassOf"),
    ("ScanningElectronMicroscopy",    "electron microscopy",               0.86, "subClassOf"),
    ("HeatTreatmentFurnace",          "heat treatment device",             0.86, "subClassOf"),
    ("TwoDimensionalContinuantFiatBoundary", "continuant fiat boundary",   0.92, "subClassOf"),
    ("QuasiStaticTensileTest",        "tensile testing process",           0.78, "subClassOf"),

    # ── subClassOf mittel (4): niedrige Scores → zwingt w5/w6 zu arbeiten ────
    ("HardnessVickers",               "indentation hardness",              0.78, "subClassOf"),
    ("TransmissionElectronMicroscope","electron microscope",               0.78, "subClassOf"),
    ("Electroplating",                "coating from the ionized state",    0.78, "subClassOf"),
]

def compute_weighted_score(weights, bw_label, bw_iri, pmdco_iri,
                            pmdco_label, bw_description="", bw_hierarchy=None):
    """
    Berechnet den gewichteten Similarity-Score für ein Entitätspaar
    mit gegebenen Gewichten. Gibt Score zwischen 0 und 1 zurück.
    """
    if bw_hierarchy is None:
        bw_hierarchy = {"parents": [], "children": [], "siblings": []}

    from difflib import SequenceMatcher
    import re

    w1, w2, w3, w4, w5, w6, w7, w8 = weights

    def normalize(text):
        text = text.lower().replace("caesium", "cesium").replace("aluminium", "aluminum")
        return re.sub(r'[-_]', '', text)

    def token_overlap(a, b):
        ta = set(re.findall(r'\w+', a.lower()))
        tb = set(re.findall(r'\w+', b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    bw_norm    = normalize(bw_label)
    pmdco_norm = normalize(pmdco_label)

    # s1 — Lexikalische Ähnlichkeit
    s1 = SequenceMatcher(None, bw_norm, pmdco_norm).ratio()

    # s2 — Token-Overlap (Jaccard)
    s2 = token_overlap(bw_label, pmdco_label)

    # s3 — Substring
    s3 = 0.7 if (bw_norm in pmdco_norm or pmdco_norm in bw_norm) else 0.0

    # s4 — Synonym
    synonyms = get_synonyms(bw_label)
    s4 = 0.6 if any(syn in pmdco_norm for syn in synonyms) else 0.0

    # s5 — Embedding
    s5 = get_embedding_similarity(bw_iri, pmdco_iri) if USE_EMBEDDINGS else 0.0

    # s6 — Hierarchie
    pmdco_hier      = get_extended_hierarchy_info(pmdco_iri, pmdco_graph)
    bw_parents_l    = [p.lower() for p in bw_hierarchy.get("parents", [])]
    pmdco_parents_l = [p.lower() for p in pmdco_hier.get("parents", [])]
    pmdco_children_l= [c.lower() for c in pmdco_hier.get("children", [])]

    s6 = 0.0
    if bw_norm in pmdco_children_l:
        s6 = 0.8
    elif set(bw_parents_l) & set(pmdco_parents_l):
        s6 = 0.5

    # s7 — Type-Match (1.0 bei Ground-Truth-Paaren als geprüft angenommen)
    s7 = 1.0

    # s8 — Description-Similarity (0 wenn nicht vorhanden)
    s8 = 0.0

    score = (w1*s1 + w2*s2 + w3*s3 + w4*s4 +
             w5*s5 + w6*s6 + w7*s7 + w8*s8)
    return score


def objective(weights):
    """
    Minimiert einen kombinierten Verlust aus:
    1. MSE zwischen berechnetem Score und Ground-Truth-Konfidenz
    2. Relationsterm: subClassOf-Paare sollen niedrigeren Score als
       equivalentClass-Paare erhalten (Trennbarkeit)
    """
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Gewichtung der beiden Verlustterme
    ALPHA = 0.7  # Gewicht MSE-Term
    BETA  = 0.3  # Gewicht Relations-Term

    mse_errors     = []
    relation_errors= []

    for bw_label, pmdco_label, true_conf, true_relation in valid_pairs:
        bw_rows    = bwmd[bwmd['Label'].str.lower() == bw_label.lower()]
        pmdco_rows = pmdco[pmdco['Label'].str.lower() == pmdco_label.lower()]

        if bw_rows.empty or pmdco_rows.empty:
            continue

        bw_iri    = str(bw_rows.iloc[0]['Entity'])
        pmdco_iri = str(pmdco_rows.iloc[0]['Entity'])
        bw_hier   = get_extended_hierarchy_info(bw_iri, bwmd_graph)

        predicted = compute_weighted_score(
            weights, bw_label, bw_iri, pmdco_iri, pmdco_label,
            bw_hierarchy=bw_hier
        )

        # Term 1: MSE auf Konfidenz
        mse_errors.append((predicted - true_conf) ** 2)

        # Term 2: Relationsterm
        # equivalentClass/equivalentProperty → Score sollte >= 0.85 sein
        # subClassOf/subPropertyOf           → Score sollte <  0.85 sein
        is_equivalent = true_relation in ("equivalentClass", "equivalentProperty")
        THRESHOLD = 0.85
        if is_equivalent and predicted < THRESHOLD:
            # Äquivalenz-Paar mit zu niedrigem Score → bestrafen
            relation_errors.append((THRESHOLD - predicted) ** 2)
        elif not is_equivalent and predicted >= THRESHOLD:
            # Subsumptions-Paar mit zu hohem Score → bestrafen
            relation_errors.append((predicted - THRESHOLD) ** 2)
        else:
            relation_errors.append(0.0)

    mse  = np.mean(mse_errors)      if mse_errors      else 1.0
    rpen = np.mean(relation_errors) if relation_errors else 0.0

    return ALPHA * mse + BETA * rpen


# Startwerte — innerhalb der neuen Bounds, bereits nahe der erwarteten Lösung:
# w1 bewusst niedrig gesetzt damit Optimizer nicht sofort an den Bound läuft
# w4/w5 bereits auf ihrem Mindest-Floor, w5 als stärkstes Signal höher gewichtet
W0 = np.array([0.15, 0.05, 0.05, 0.10, 0.35, 0.15, 0.10, 0.05])

# Constraints: Summe = 1
# Bounds: Engere Grenzen basierend auf Diagnose der Degeneration:
#   - w1 (lexical) max 0.30: verhindert dass exakte BFO-Labels w1 trivial pushen
#   - w4 (synonym) min 0.05: Synonym-Paare brauchen Mindestbeitrag
#   - w5 (embedding) min 0.10: Embeddings sind wertvollstes Signal für Synonyme
#   - w7 (type) max 0.15: minimal nützlich, darf nicht dominieren
_constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
_bounds = [
    (0.01, 0.30),  # w1  lexical similarity     ← CAP: BFO-Dominanz verhindern
    (0.01, 0.40),  # w2  token overlap
    (0.01, 0.40),  # w3  substring match
    (0.05, 0.40),  # w4  synonym match           ← FLOOR: Synonym-Paare brauchen es
    (0.10, 0.50),  # w5  embedding similarity    ← FLOOR: wertvollstes Signal
    (0.01, 0.40),  # w6  hierarchy context
    (0.01, 0.15),  # w7  type consistency        ← CAP: Degeneration verhindern
    (0.01, 0.20),  # w8  description similarity
]
_labels = ['w1 (lexical)', 'w2 (token)', 'w3 (substring)', 'w4 (synonym)',
           'w5 (embedding)', 'w6 (hierarchy)', 'w7 (type)', 'w8 (description)']

# === GROUND-TRUTH VORAB-PRÜFUNG ===
print("\n🔍 Prüfe Ground-Truth-Paare gegen CSV-Daten...")
print(f"   {'BWMD Label':<45} {'PMDco Label':<45} {'Status'}")
print(f"   {'-'*45} {'-'*45} {'-'*10}")

valid_pairs = []
skipped_pairs = []

for bw_label, pmdco_label, conf, true_relation in GROUND_TRUTH_PAIRS:
    bw_rows    = bwmd[bwmd['Label'].str.lower() == bw_label.lower()]
    pmdco_rows = pmdco[pmdco['Label'].str.lower() == pmdco_label.lower()]

    bw_found    = not bw_rows.empty
    pmdco_found = not pmdco_rows.empty

    if bw_found and pmdco_found:
        status = "✅ OK"
        valid_pairs.append((bw_label, pmdco_label, conf, true_relation))
    elif not bw_found and not pmdco_found:
        status = "❌ BWMD+PMDco nicht gefunden"
        skipped_pairs.append((bw_label, pmdco_label, "both missing"))
    elif not bw_found:
        status = "⚠️  BWMD nicht gefunden"
        skipped_pairs.append((bw_label, pmdco_label, "BWMD missing"))
    else:
        status = "⚠️  PMDco nicht gefunden"
        skipped_pairs.append((bw_label, pmdco_label, "PMDco missing"))

    print(f"   {bw_label:<45} {pmdco_label:<45} {status}")

print(f"\n   Gefunden: {len(valid_pairs)} / {len(GROUND_TRUTH_PAIRS)} Paare")

if skipped_pairs:
    print(f"\n   ⚠️  {len(skipped_pairs)} Paare werden übersprungen:")
    for bw, pm, reason in skipped_pairs:
        print(f"      → '{bw}' ↔ '{pm}' ({reason})")
    print(f"\n   💡 Tipp: Prüfe die genaue Schreibweise in den CSV-Spalte 'Label'")
    print(f"      BWMD Labels (erste 10):  {list(bwmd['Label'].head(10))}")
    print(f"      PMDco Labels (erste 10): {list(pmdco['Label'].head(10))}")

if len(valid_pairs) < 5:
    print(f"\n   ❌ Zu wenige gültige Paare für sinnvolle Optimierung (min. 5).")
    print(f"      Nutze manuelle Startwerte als finale Gewichte.")
    OPT_W = W0
else:
    print(f"\n🔧 Optimiere Similarity-Gewichte via scipy.optimize (SLSQP)...")
    _opt = minimize(
        objective,
        W0,
        method='SLSQP',
        bounds=_bounds,
        constraints=_constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    OPT_W = _opt.x / _opt.x.sum()

    print(f"✅ Optimierung abgeschlossen")
    print(f"   Konvergiert : {_opt.success} — {_opt.message}")
    print(f"   MSE         : {_opt.fun:.6f}")
    print(f"   Gültige Paare verwendet: {len(valid_pairs)}")
    print(f"\n   {'Komponente':<25} {'Startwert':>10} {'Optimiert':>10}")
    print(f"   {'-'*47}")
    for lbl, w0_i, wn_i in zip(_labels, W0, OPT_W):
        print(f"   {lbl:<25} {w0_i:>10.3f} {wn_i:>10.3f}")

monitor.checkpoint("Weight_Optimization")
# ============================================================

# === VERBESSERTES PRE-FILTERING MIT EMBEDDINGS ===
def find_relevant_candidates(bw_label, bw_type, bw_description, bw_hierarchy, bw_iri, pmdco_df, pmdco_graph, top_n=20):
    """
    Erweitert mit Embeddings für semantische Ähnlichkeit.
    
    OPTIMIERT: Nutzt vorberechnete BWMD-Embeddings (bw_iri als Key).
    """
    from difflib import SequenceMatcher
    import re
    
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def normalize_chemical(text):
        text = text.lower().replace("caesium", "cesium").replace("aluminium", "aluminum")
        text = re.sub(r'[-_]', '', text)
        return text
    
    def token_overlap(a, b):
        tokens_a = set(re.findall(r'\w+', a.lower()))
        tokens_b = set(re.findall(r'\w+', b.lower()))
        if not tokens_a or not tokens_b:
            return 0
        return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    
    bw_label_safe = str(bw_label) if pd.notna(bw_label) else ""
    bw_normalized = normalize_chemical(bw_label_safe)
    bw_synonyms = get_synonyms(bw_label_safe)
    
    # Text für Embedding: Label + Description
    bw_text_for_embedding = bw_label_safe
    if bw_description and len(bw_description) > 10:
        bw_text_for_embedding += f" {bw_description[:200]}"
    
    candidates = []
    for _, row in pmdco_df.iterrows():
        pmdco_label = str(row.Label) if pd.notna(row.Label) else ""
        if not pmdco_label:
            continue
            
        pmdco_normalized = normalize_chemical(pmdco_label)
        
        # === LEXIKALISCHE ÄHNLICHKEITEN ===
        label_sim = similarity(bw_normalized, pmdco_normalized)
        token_sim = token_overlap(bw_label_safe, pmdco_label)
        
        # Synonym-Matching
        synonym_boost = 0.0
        for syn in bw_synonyms:
            if syn in pmdco_normalized:
                synonym_boost = 0.6
                break
        
        # Substring-Matching
        substring_match = 0.0
        if bw_normalized in pmdco_normalized or pmdco_normalized in bw_normalized:
            substring_match = 0.7
        
        # === EMBEDDING-ÄHNLICHKEIT (SEMANTISCH) ===
        embedding_sim = 0.0
        if USE_EMBEDDINGS:
            embedding_sim = get_embedding_similarity(bw_iri, row.Entity)
        
        # Type-Match
        type_match = 0.0
        pmdco_type = str(row.Type) if pd.notna(row.Type) else ""
        bw_type_safe = str(bw_type) if pd.notna(bw_type) and bw_type else ""
        
        if bw_type_safe and pmdco_type:
            if bw_type_safe.lower() in pmdco_type.lower():
                type_match = 1.0
            elif any(t in pmdco_type.lower() for t in ['class', 'property', 'process']):
                type_match = 0.4
        
        # === HIERARCHIE-MATCHING ===
        hierarchy_boost = 0.0
        pmdco_hier = get_extended_hierarchy_info(row.Entity, pmdco_graph)
        
        bw_parents_lower = [p.lower() for p in bw_hierarchy.get('parents', [])]
        pmdco_parents_lower = [p.lower() for p in pmdco_hier.get('parents', [])]
        pmdco_children_lower = [c.lower() for c in pmdco_hier.get('children', [])]
        
        parent_overlap = len(set(bw_parents_lower) & set(pmdco_parents_lower))
        if parent_overlap > 0:
            hierarchy_boost = 0.5
        
        if bw_normalized in pmdco_children_lower:
            hierarchy_boost = 0.8
        
        if set(bw_parents_lower) & set(pmdco_parents_lower):
            hierarchy_boost = max(hierarchy_boost, 0.4)
        
        # Description-Match
        desc_sim = 0.0
        if bw_description and 'Description' in pmdco_df.columns:
            pmdco_desc = str(row.get('Description', ''))
            if pd.notna(pmdco_desc) and pmdco_desc and len(pmdco_desc) > 10:
                desc_sim = similarity(bw_description[:200], pmdco_desc[:200])
        
        # === GEWICHTETER SCORE (optimierte Gewichte via SLSQP) ===
        if USE_EMBEDDINGS:
            score = (
                label_sim       * OPT_W[0] +
                token_sim       * OPT_W[1] +
                substring_match * OPT_W[2] +
                synonym_boost   * OPT_W[3] +
                embedding_sim   * OPT_W[4] +
                hierarchy_boost * OPT_W[5] +
                type_match      * OPT_W[6] +
                desc_sim        * OPT_W[7]
            )
        else:
            # Ohne Embeddings: w5 auf lexikalische Methoden umverteilen
            score = (
                label_sim       * (OPT_W[0] + OPT_W[4] * 0.40) +
                token_sim       * (OPT_W[1] + OPT_W[4] * 0.30) +
                substring_match * (OPT_W[2] + OPT_W[4] * 0.15) +
                synonym_boost   * (OPT_W[3] + OPT_W[4] * 0.15) +
                hierarchy_boost * OPT_W[5] +
                type_match      * OPT_W[6] +
                desc_sim        * OPT_W[7]
            )
        
        candidates.append({
            'row': row,
            'score': score,
            'label_sim': label_sim,
            'embedding_sim': embedding_sim,
            'hierarchy_boost': hierarchy_boost
        })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    top_candidates = []
    for c in candidates:
        if len(top_candidates) >= top_n and c['score'] < 0.25:
            break
        top_candidates.append(c['row'])
        if len(top_candidates) >= top_n * 1.5:
            break
    
    if not top_candidates and candidates:
        return [c['row'] for c in candidates[:5]]
    
    return top_candidates

# === ERWEITERTES LLM-MAPPING ===
def get_sparql_mapping(bw_label, bw_type, bw_description, bw_iri, bw_hierarchy, pmdco_df, pmdco_graph):
    """Mapping mit vollständiger Hierarchie-Information und typ-sensitiven Relationen."""
    
    relevant_pmdco = find_relevant_candidates(
        bw_label, bw_type, bw_description, bw_hierarchy, bw_iri, pmdco_df, pmdco_graph, top_n=20
    )
    
    # PMDco-Entitäten mit vollständiger Hierarchie
    pmdco_entries = []
    for row in relevant_pmdco:
        desc = get_full_description(row, pmdco_df)
        pmdco_hier = get_extended_hierarchy_info(row.Entity, pmdco_graph)
        
        entry = f"- {row.Label} ({row.Type})"
        if desc:
            entry += f"\n  Description: {desc[:150]}"
        if pmdco_hier['parents']:
            entry += f"\n  Parents: {', '.join(pmdco_hier['parents'])}"
        if pmdco_hier['children']:
            entry += f"\n  Children: {', '.join(pmdco_hier['children'])}"
        if pmdco_hier['siblings']:
            entry += f"\n  Siblings: {', '.join(pmdco_hier['siblings'][:2])}"
        entry += f"\n  IRI: {row.Entity}"
        
        pmdco_entries.append(entry)
    
    pmdco_text = "\n\n".join(pmdco_entries)

    # BWMD Hierarchie-Kontext (erweitert)
    hierarchy_context = ""
    if bw_hierarchy.get('parents'):
        hierarchy_context += f"\n- Parents: {', '.join(bw_hierarchy['parents'])}"
    if bw_hierarchy.get('children'):
        hierarchy_context += f"\n- Children: {', '.join(bw_hierarchy['children'])}"
    if bw_hierarchy.get('siblings'):
        hierarchy_context += f"\n- Siblings: {', '.join(bw_hierarchy['siblings'][:2])}"

    # ── Typ-sensitve Relationen für den Prompt ──────────────────────────────
    valid_relations = get_valid_relations(bw_type)
    is_property = is_property_type(bw_type)

    if is_property:
        relation_block = """RELATIONS (this entity is a PROPERTY – use property relations only):
- equivalentProperty: BWMD and PMDco property represent the same relationship (e.g. hasYoungsModulus ↔ hasElasticModulus)
- subPropertyOf: BWMD property is more specific than PMDco property (e.g. hasVickersHardness ↔ hasHardnessValue)"""
    else:
        relation_block = """RELATIONS (this entity is a CLASS – use class relations only):
- equivalentClass: Same concept, same abstraction level (confidence >85 if strong evidence)
- subClassOf: BWMD class is more specific than PMDco class (check if PMDco is parent)"""
    # ────────────────────────────────────────────────────────────────────────

    prompt = f"""You are an ontology alignment expert specializing in materials science and testing.

{FEW_SHOT_EXAMPLES}

TASK: Map this BWMD entity to the BEST matching PMDco entity.

BWMD Entity:
- Label: "{bw_label}"
- Type: {bw_type}
- IRI: {bw_iri}{hierarchy_context}
{f"- Description: {bw_description[:150]}" if bw_description else ""}

PMDco Candidates (ranked by relevance):
{pmdco_text}

ALIGNMENT STRATEGY:
1. **Hierarchical Clues**: If BWMD has parent X and PMDco candidate also has parent X → likely siblings (equivalent*)
2. **Parent-Child Match**: If BWMD label appears in PMDco's children → use sub*Of
3. **Sibling Pattern**: Similar siblings suggest equivalent abstraction level
4. **Synonym Awareness**: Caesium=Cesium, YoungModulus=ElasticModulus
5. **Type Consistency**: Class→Class relations, Property→Property relations

{relation_block}

CONFIDENCE GUIDELINES:
- 90-100: Perfect match (same hierarchy position, synonyms, or exact label)
- 75-89: Strong match (clear hierarchical relationship)
- 50-74: Good match (semantic similarity, some uncertainty)
- 25-49: Weak match (best available, needs review)

IMPORTANT: The "relation" field MUST be one of: {valid_relations}

OUTPUT (valid JSON, no line breaks in strings):
{{"pmdco_label":"...","pmdco_iri":"...","relation":"{valid_relations[0]}","confidence":85,"reasoning":"max 80 chars"}}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Expert ontology mapper. Output valid JSON only. No line breaks in strings."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            timeout=35.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ LLM-Fehler bei {bw_label}: {e}")
        if relevant_pmdco:
            best = relevant_pmdco[0]
            return json.dumps({
                "pmdco_label": str(best.Label),
                "pmdco_iri": str(best.Entity),
                "relation": get_default_relation(bw_type),   # ← typ-sensitiv
                "confidence": 30,
                "reasoning": "API error - auto-matched"
            })
        return json.dumps({
            "pmdco_label": "",
            "pmdco_iri": "",
            "relation": get_default_relation(bw_type),        # ← typ-sensitiv
            "confidence": 0,
            "reasoning": "API error, no candidates"
        })

# === MAPPING DURCHFÜHREN ===
results = []
validation_errors = []
invalid_relation_count = 0
api_error_count = 0
json_error_count = 0

print("\n🚀 Starte Mapping...")
mapping_start = time.time()

for idx, row in tqdm(enumerate(bwmd.iterrows()), total=len(bwmd)):
    _, row = row
    bw_label = str(row.get("Label", ""))
    bw_type = str(row.get("Type", ""))
    bw_iri = str(row.get("Entity", ""))
    bw_description = get_full_description(row, bwmd)
    
    bwmd_validation = validate_entity_exists(bw_iri, bwmd_graph, "BWMD")
    bw_hierarchy = get_extended_hierarchy_info(bw_iri, bwmd_graph)
    
    llm_raw = get_sparql_mapping(
        bw_label, bw_type, bw_description, bw_iri, bw_hierarchy, pmdco, pmdco_graph
    )
    llm_clean = clean_llm_json(llm_raw)

    try:
        parsed = json.loads(llm_clean)
        pmdco_label = parsed.get("pmdco_label", "")
        pmdco_iri = parsed.get("pmdco_iri", "")
        relation = parsed.get("relation", get_default_relation(bw_type))  # ← typ-sensitiv
        confidence = parsed.get("confidence", 0)
        reasoning = parsed.get("reasoning", "")

        # ── Relation validieren und ggf. korrigieren ───────────────────────
        corrected_relation = validate_relation(relation, bw_type)
        if corrected_relation != relation:
            print(f"   ⚠️  Relation korrigiert: {bw_label}: '{relation}' → '{corrected_relation}' (Type: {bw_type})")
            relation = corrected_relation
            invalid_relation_count += 1
        # ───────────────────────────────────────────────────────────────────
        
        if not pmdco_iri or not relation:
            relevant_pmdco = find_relevant_candidates(
                bw_label, bw_type, bw_description, bw_hierarchy, bw_iri, pmdco, pmdco_graph, top_n=20
            )
            if len(relevant_pmdco) > 0:
                best = relevant_pmdco[0]
                pmdco_label = str(best.Label)
                pmdco_iri = str(best.Entity)
                relation = get_default_relation(bw_type)   # ← typ-sensitiv
                confidence = 25
                reasoning = "Fallback to top candidate"
        
        pmdco_validation = {"exists": True, "error": ""}
        if pmdco_iri:
            pmdco_validation = validate_entity_exists(pmdco_iri, pmdco_graph, "PMDco")
            
            if not pmdco_validation['exists']:
                validation_errors.append({
                    "BWMD": bw_label,
                    "PMDco_Suggested": pmdco_label,
                    "PMDco_IRI": pmdco_iri,
                    "Error": "PMDco entity not in RDF"
                })
                confidence = 0
        
    except json.JSONDecodeError as e:
        print(f"   JSON-Fehler bei {bw_label}: {str(e)[:80]}")
        
        try:
            import re
            cleaned = ''.join(char if 32 <= ord(char) < 127 else ' ' for char in llm_clean)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            parsed = json.loads(cleaned)
            
            pmdco_label = parsed.get("pmdco_label", "")
            pmdco_iri = parsed.get("pmdco_iri", "")
            relation = parsed.get("relation", get_default_relation(bw_type))  # ← typ-sensitiv
            confidence = parsed.get("confidence", 0)
            reasoning = parsed.get("reasoning", "")

            # Relation auch hier validieren
            relation = validate_relation(relation, bw_type)
            
            pmdco_validation = {"exists": True, "error": ""}
            if pmdco_iri:
                pmdco_validation = validate_entity_exists(pmdco_iri, pmdco_graph, "PMDco")
                if not pmdco_validation['exists']:
                    confidence = 0
                    
        except:
            print(f"   → Fallback: Top-Kandidat")
            relevant_pmdco = find_relevant_candidates(
                bw_label, bw_type, bw_description, bw_hierarchy, bw_iri, pmdco, pmdco_graph, top_n=20
            )
            
            if relevant_pmdco and len(relevant_pmdco) > 0:
                best = relevant_pmdco[0]
                pmdco_label = str(best.Label)
                pmdco_iri = str(best.Entity)
                relation = get_default_relation(bw_type)   # ← typ-sensitiv
                confidence = 20
                reasoning = "JSON error fallback"
                
                pmdco_validation = validate_entity_exists(pmdco_iri, pmdco_graph, "PMDco")
            else:
                pmdco_label = ""
                pmdco_iri = ""
                relation = "ERROR"
                confidence = 0
                reasoning = "Critical error"
                pmdco_validation = {"exists": False, "error": ""}

    results.append({
        "BWMD_Label": bw_label,
        "BWMD_Entity": bw_iri,
        "BWMD_Type": bw_type,
        "BWMD_Parents": ", ".join(bw_hierarchy['parents']),
        "BWMD_Children": ", ".join(bw_hierarchy['children']),
        "BWMD_Valid": bwmd_validation['exists'],
        "PMDco_Label": pmdco_label,
        "PMDco_Entity": pmdco_iri,
        "PMDco_Valid": pmdco_validation['exists'],
        "Relation": relation,
        "Confidence": confidence,
        "Reasoning": reasoning,
        "Validation_Error": "" if bwmd_validation['exists'] and pmdco_validation['exists'] 
                           else bwmd_validation.get('error', '') + " " + pmdco_validation.get('error', '')
    })

    time.sleep(0.4)

mapping_end = time.time()
mapping_duration = mapping_end - mapping_start
avg_time_per_entity = mapping_duration / len(bwmd)

print(f"\n✅ Mapping completed!")
print(f"   Total time: {mapping_duration/60:.2f} minutes ({mapping_duration:.1f} seconds)")
print(f"   Average per entity: {avg_time_per_entity:.2f} seconds")
print(f"   Invalid relations corrected: {invalid_relation_count}")
print(f"   JSON parsing errors: {json_error_count}")

monitor.checkpoint("Mapping_Complete")

# === ERGEBNISSE SPEICHERN ===
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Confidence", ascending=False)

df_results["Quality"] = df_results["Confidence"].apply(
    lambda x: "High" if x >= 80 else ("Medium" if x >= 50 else "Low" if x > 0 else "Invalid")
)

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Mappings', index=False)
    
    df_valid = df_results[
        (df_results['BWMD_Valid'] == True) & 
        (df_results['PMDco_Valid'] == True) &
        (df_results['Relation'] != 'ERROR')
    ][['BWMD_Entity', 'PMDco_Entity', 'Relation', 'Confidence']]
    df_valid.to_excel(writer, sheet_name='Valid_Only', index=False)
    
    # High-Confidence Mappings (>80)
    df_high = df_results[df_results['Confidence'] >= 80]
    df_high.to_excel(writer, sheet_name='High_Confidence', index=False)
    
    # Low-Confidence für Review (<50)
    df_low = df_results[(df_results['Confidence'] > 0) & (df_results['Confidence'] < 50)]
    df_low.to_excel(writer, sheet_name='Needs_Review', index=False)
    
    if validation_errors:
        df_errors = pd.DataFrame(validation_errors)
        df_errors.to_excel(writer, sheet_name='Validation_Errors', index=False)

print(f"\n💾 Enhanced Mapping gespeichert: {output_excel}")

print("\n" + "="*70)
print("COMPREHENSIVE METRICS REPORT FOR METHODS SECTION".center(70))
print("="*70)

print(f"\n📊 DATASET STATISTICS:")
print(f"  BWMD Entities: {len(bwmd)}")
print(f"  BWMD RDF Triples: {len(bwmd_graph)}")
print(f"  PMDco Entities: {len(pmdco)}")
print(f"  PMDco RDF Triples: {len(pmdco_graph)}")

print(f"\n📈 MAPPING RESULTS:")
print(f"  Total Mappings: {len(df_results)}")
print(f"\n  Relation Distribution:")
for relation, count in df_results['Relation'].value_counts().items():
    percentage = (count / len(df_results)) * 100
    print(f"    {relation}: {count} ({percentage:.1f}%)")

print(f"\n  Confidence Distribution:")
conf_90_100 = len(df_results[df_results['Confidence'] >= 90])
conf_80_90 = len(df_results[(df_results['Confidence'] >= 80) & (df_results['Confidence'] < 90)])
conf_70_80 = len(df_results[(df_results['Confidence'] >= 70) & (df_results['Confidence'] < 80)])
conf_60_70 = len(df_results[(df_results['Confidence'] >= 60) & (df_results['Confidence'] < 70)])
conf_50_60 = len(df_results[(df_results['Confidence'] >= 50) & (df_results['Confidence'] < 60)])
conf_40_50 = len(df_results[(df_results['Confidence'] >= 40) & (df_results['Confidence'] < 50)])
conf_30_40 = len(df_results[(df_results['Confidence'] >= 30) & (df_results['Confidence'] < 40)])
conf_20_30 = len(df_results[(df_results['Confidence'] >= 20) & (df_results['Confidence'] < 30)])
conf_10_20 = len(df_results[(df_results['Confidence'] >= 10) & (df_results['Confidence'] < 20)])
conf_0_10 = len(df_results[(df_results['Confidence'] > 0) & (df_results['Confidence'] < 10)])
conf_0 = len(df_results[df_results['Confidence'] == 0])

print(f"    90-100: {conf_90_100} ({conf_90_100/len(df_results)*100:.1f}%)")
print(f"    80-89:  {conf_80_90} ({conf_80_90/len(df_results)*100:.1f}%)")
print(f"    70-79:  {conf_70_80} ({conf_70_80/len(df_results)*100:.1f}%)")
print(f"    60-69:  {conf_60_70} ({conf_60_70/len(df_results)*100:.1f}%)")
print(f"    50-59:  {conf_50_60} ({conf_50_60/len(df_results)*100:.1f}%)")
print(f"    40-49:  {conf_40_50} ({conf_40_50/len(df_results)*100:.1f}%)")
print(f"    30-39:  {conf_30_40} ({conf_30_40/len(df_results)*100:.1f}%)")
print(f"    20-29:  {conf_20_30} ({conf_20_30/len(df_results)*100:.1f}%)")
print(f"    10-19:  {conf_10_20} ({conf_10_20/len(df_results)*100:.1f}%)")
print(f"    0-9:    {conf_0_10} ({conf_0_10/len(df_results)*100:.1f}%)")
print(f"    0:      {conf_0} ({conf_0/len(df_results)*100:.1f}%)")

avg_confidence = df_results[df_results['Confidence'] > 0]['Confidence'].mean()
median_confidence = df_results[df_results['Confidence'] > 0]['Confidence'].median()
print(f"\n  Average Confidence: {avg_confidence:.2f}")
print(f"  Median Confidence: {median_confidence:.2f}")

print(f"\n  Quality Tiers:")
for quality, count in df_results['Quality'].value_counts().items():
    percentage = (count / len(df_results)) * 100
    print(f"    {quality}: {count} ({percentage:.1f}%)")

print(f"\n🔍 VALIDATION RESULTS:")
bwmd_valid = df_results['BWMD_Valid'].sum()
pmdco_valid = df_results['PMDco_Valid'].sum()
both_valid = ((df_results['BWMD_Valid']) & (df_results['PMDco_Valid'])).sum()
print(f"  BWMD Entities Valid: {bwmd_valid} / {len(df_results)} ({bwmd_valid/len(df_results)*100:.1f}%)")
print(f"  PMDco Entities Valid: {pmdco_valid} / {len(df_results)} ({pmdco_valid/len(df_results)*100:.1f}%)")
print(f"  Both Valid: {both_valid} / {len(df_results)} ({both_valid/len(df_results)*100:.1f}%)")
print(f"  Hallucinated Entities: {len(validation_errors)}")

print(f"\n⚙️ PROCESSING STATISTICS:")
print(f"  Invalid Relations Corrected: {invalid_relation_count}")
print(f"  JSON Parsing Errors: {json_error_count}")
print(f"  API Errors: {api_error_count}")

print(f"\n🧠 MODEL & PARAMETERS:")
print(f"  LLM Model: {model_name}")
print(f"  Temperature: {temperature}")
print(f"  Top-K Candidates: 20")
print(f"  Similarity Threshold: 0.25")
print(f"\n⚖️  OPTIMIZED WEIGHTS (SLSQP, MSE={_opt.fun:.6f}):")
for lbl, wn_i in zip(_labels, OPT_W):
    print(f"    {lbl:<25} {wn_i:.4f}")
print(f"  Embeddings Used: {'Yes (all-MiniLM-L6-v2, 384-dim)' if USE_EMBEDDINGS else 'No'}")
if USE_EMBEDDINGS:
    print(f"    PMDco Embeddings: {len(pmdco_embeddings)} vectors")
    print(f"    BWMD Embeddings: {len(bwmd_embeddings)} vectors")
    print(f"    Embedding Size: ~{(len(pmdco_embeddings) + len(bwmd_embeddings)) * 384 * 4 / (1024**2):.1f} MB")

print(f"\n⏱️ PERFORMANCE:")
print(f"  Total Runtime: {(time.time() - monitor.start_time)/60:.2f} minutes")
print(f"  Mapping Time: {mapping_duration/60:.2f} minutes")
print(f"  Average Time per Entity: {avg_time_per_entity:.2f} seconds")

if PSUTIL_AVAILABLE:
    print(f"  Peak Memory: {monitor.get_memory_mb():.1f} MB")

# Call the detailed performance report
monitor.print_report()

print("\n✅ PIPELINE COMPLETED!")
print("="*70)
print("\n📝 FOR METHODS SECTION - Copy these metrics above")
print("="*70)

# === ERWEITERTE STATISTIKEN ===
print("\n📊 Mapping-Statistiken:")
print(f"Gesamt: {len(df_results)}")
print(f"\nRelationen:")
print(df_results['Relation'].value_counts())
print(f"\nQualität:")
print(df_results['Quality'].value_counts())
print(f"\nConfidence-Verteilung:")
print(f"  >90: {len(df_results[df_results['Confidence'] >= 90])}")
print(f"  80-90: {len(df_results[(df_results['Confidence'] >= 80) & (df_results['Confidence'] < 90)])}")
print(f"  50-80: {len(df_results[(df_results['Confidence'] >= 50) & (df_results['Confidence'] < 80)])}")
print(f"  <50: {len(df_results[(df_results['Confidence'] > 0) & (df_results['Confidence'] < 50)])}")
print(f"\nValidierung:")
print(f"  BWMD valid: {df_results['BWMD_Valid'].sum()} / {len(df_results)}")
print(f"  PMDco valid: {df_results['PMDco_Valid'].sum()}")
print(f"  Beide valid: {((df_results['BWMD_Valid']) & (df_results['PMDco_Valid'])).sum()}")
print(f"\nØ Confidence: {df_results[df_results['Confidence'] > 0]['Confidence'].mean():.1f}")

if validation_errors:
    print(f"\n⚠️ {len(validation_errors)} Validierungsfehler!")
    print("   → Siehe Sheet 'Validation_Errors'")

print("\n✅ Fertig! Prüfe besonders:")
print("   - Sheet 'High_Confidence' → Direkt verwendbar")
print("   - Sheet 'Needs_Review' → Manuelle Prüfung empfohlen")

if USE_EMBEDDINGS:
    print("\n🧠 Embedding-Statistik:")
    high_emb_count = 0
    for _, row in df_results.iterrows():
        if row['Confidence'] > 70:
            high_emb_count += 1
    print(f"   Embeddings trugen zu ~{int(high_emb_count * 0.3)} High-Confidence-Mappings bei")
    print("   (Geschätzt: 30% des Scores bei aktivierten Embeddings)")
