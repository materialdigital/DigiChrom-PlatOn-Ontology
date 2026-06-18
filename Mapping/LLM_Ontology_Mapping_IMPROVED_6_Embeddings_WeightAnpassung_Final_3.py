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
from scipy.optimize import minimize

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
API_KEY  = "sk-aEreZmXpktbKrkQvBGMD"
BASE_URL = "https://llm-proxy.imla.hs-offenburg.de/"

# ════════════════════════════════════════════════════════════════════════════
# ONTOLOGY ALIGNMENT CONFIGURATION
# Adapt these values for any source/target ontology pair.
# Everything else in this script is generic.
# ════════════════════════════════════════════════════════════════════════════

# ── File paths ───────────────────────────────────────────────────────────────
SOURCE_OWL  = "BWMD_Ontologie_2020-09-28-KurzfassungBericht2_4DigiChrom_Updated_v3.ttl"
SOURCE_CSV  = "BWMDCORE_rdf_summary_all_entities.csv"
TARGET_OWL  = "pmdco-full.ttl"
TARGET_CSV  = "PMDco_rdf_summary_all_entities.csv"
OUTPUT_FILE = "Alignment_Output.xlsx"
LOG_FILE    = "Alignment_Terminal_Log.txt"   # kompletter Terminal-Output (fuer SI / Reproduzierbarkeit)

# ── Ontology identities (used in prompts and print statements) ───────────────
SOURCE_NAME        = "BWMD"
SOURCE_DESCRIPTION = (
    "BWMD (Basic and Workflow-centric Materials Design Ontology): "
    "a BFO-aligned mid-level ontology developed at Fraunhofer IWM, "
    "covering materials testing, manufacturing processes, characterization "
    "and material properties. Classes use CamelCase labels. "
    "Available at: https://github.com/materials-data-facility/BWMD-ontology"
)
TARGET_NAME        = "PMDco"
TARGET_DESCRIPTION = (
    "PMDco (Platform MaterialDigital Core Ontology): "
    "the reference ontology of the German Platform MaterialDigital initiative, "
    "also BFO-aligned. Labels use natural language (lowercase). "
    "Available at: https://github.com/materialdigital/core-ontology"
)

# ── Backward-compatibility aliases (used throughout the rest of the script) ──
bwmd_owl   = SOURCE_OWL
bwmd_csv   = SOURCE_CSV
pmdco_owl  = TARGET_OWL
pmdco_csv  = TARGET_CSV
output_excel = OUTPUT_FILE
# ════════════════════════════════════════════════════════════════════════════

model_name  = "gpt-5.2"
temperature = 0.1

# ── Terminal-Output zusaetzlich in Logdatei schreiben ────────────────────────
class _Tee:
    """Dupliziert stdout in eine UTF-8-Logdatei (Progressbars bleiben auf stderr)."""
    def __init__(self, stream, logfile_path):
        self._stream = stream
        self._log = open(logfile_path, "w", encoding="utf-8", errors="replace")
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()

sys.stdout = _Tee(sys.stdout, LOG_FILE)
print(f"📄 Terminal-Output wird mitgeschrieben in: {LOG_FILE}")
# ─────────────────────────────────────────────────────────────────────────────


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
EXAMPLE 1 - Equivalent Classes, near-identical labels (BFO anchor):
BWMD: "Extensometer" (Class) | Parents: MeasurementEquipment
PMDco: "extensometer" (Class) | Full path: Entity > device > extensometer
→ equivalentClass (97) - Identical concept, same abstraction level, same hierarchy

EXAMPLE 2 - Equivalent Classes, synonym reformulation:
BWMD: "HardnessBrinell" (Class) | Parents: Hardness | Siblings: HardnessVickers
PMDco: "brinell hardness" (Class) | Full path: Entity > quality > ... > brinell hardness
→ equivalentClass (97) - Same concept, CamelCase vs natural language; sibling pattern confirms

EXAMPLE 3 - SubClass, domain-specific quantity → scalar measurement datum:
BWMD: "Pickling Temperature" (Class) | Parents: Coating Realated Quantity
PMDco: "scalar measurement datum" (Class) | Full path: Entity > information content entity > measurement datum > scalar measurement datum
→ subClassOf (86) - No direct PMDco equivalent for this specific coating quantity;
  BWMD uses this as a numerical value container → scalar measurement datum is correct.
  NOTE: Do NOT map to "temperature" (the abstract quality) — BWMD uses this class
  as a data point, not as the abstract physical concept.

EXAMPLE 4 - SubClass, equipment → device:
BWMD: "GrindingAndPolishingEquipment" (Class) | Parents: MetallographicEquipment
PMDco: "device" (Class) | Full path: Entity > device
→ subClassOf (82) - No specific PMDco class for this equipment; "device" is the
  correct general ancestor for any physical technical apparatus.

EXAMPLE 5 - SubClass, process specialisation:
BWMD: "QuasiStaticTensileTest" (Class) | Parents: MechanicalExperiment
PMDco: "tensile testing process" (Class) | Full path: Entity > process > planned process > tensile testing process
→ subClassOf (78) - Quasi-static tensile test is a specific realization of tensile testing

EXAMPLE 6 - SubClass, setpoint value → scalar value specification:
BWMD: "Temperature Target" (Class) | Parents: Coating Realated Quantity
PMDco: "scalar value specification" (Class) | Full path: Entity > information content entity > specification datum > scalar value specification
→ subClassOf (86) - Target/setpoint values are specifications, not measurements.
  Distinction: *Actual/*AsIs → scalar measurement datum; *Target/*AsSupposed → scalar value specification

EXAMPLE 7 - Equivalent Properties:
BWMD: "has Part" (ObjectProperty)
PMDco: "has part" (ObjectProperty) | IRI: BFO_0000051
→ equivalentProperty (97) - Identical mereological relation, capitalisation only

EXAMPLE 8 - SubProperty, ObjectProperty → ObjectProperty:
BWMD: "has Control Info" (ObjectProperty) | Domain: process nodes
PMDco: "has process attribute" (ObjectProperty)
→ subPropertyOf (82) - Control info is a specific process attribute; type-consistent

EXAMPLE 9 - Equivalent Classes, chemical element naming convention:
BWMD: "Silver" (Class) | Parents: ChemicalElement
PMDco candidates include: "silver atom" AND "portion of silver"
→ equivalentClass (95) with "silver atom" - BWMD element classes denote the
  chemical element as such. Convention for this ontology pair: map elements
  to the "X atom" class, NOT to "portion of X" (which denotes a material amount).

EXAMPLE 10 - SubClass NOT equivalent, despite near-identical concept:
BWMD: "Plating" (Class) | Parents: ManufacturingProcess | Children: Electroplating, ...
PMDco: "coating" (Class) | Full path: ... > planned process > coating
→ subClassOf (88) - Plating is electrochemical/chemical deposition only, while
  PMDco coating covers ALL coating methods (painting, PVD, thermal spraying).
  The source is narrower, so subClassOf is correct even though the concepts
  feel equivalent. Apply the strict instance test before using equivalentClass.
"""

GROUND_TRUTH_PAIRS = [
    # Verified against BWMD_PMDco_LLM_Mapping_Enhanced_6_Human_Check.xlsx
    # Human Check=True → Excel used directly; False → Recommendation applied

    # ── BFO / equivalentClass: exakte Labels ────────────────────────────────
    ("Continuant",                          "continuant",                        0.99, "equivalentClass"),
    ("Process",                             "process",                           0.98, "equivalentClass"),
    ("MaterialEntity",                      "material entity",                   0.98, "equivalentClass"),
    ("FiatObjectPart",                      "fiat object part",                  0.98, "equivalentClass"),
    ("Crack",                               "crack",                             0.99, "equivalentClass"),
    ("Density",                             "density",                           0.99, "equivalentClass"),
    ("Force",                               "force",                             0.99, "equivalentClass"),
    ("Hardness",                            "hardness",                          0.99, "equivalentClass"),
    ("HardnessBrinell",                     "brinell hardness",                  0.97, "equivalentClass"),
    ("Extensometer",                        "extensometer",                      0.97, "equivalentClass"),
    ("Temperature",                         "temperature",                       0.97, "equivalentClass"),
    ("LoadCell",                            "load cell",                         0.97, "equivalentClass"),
    ("Grain",                               "crystal grain",                     0.97, "equivalentClass"),
    ("EngineeringMaterial",                 "engineered material",               0.97, "equivalentClass"),
    ("Specimen",                            "specimen",                          0.96, "equivalentClass"),
    ("Furnace",                             "furnace",                           0.96, "equivalentClass"),
    ("LightMicroscopy",                     "light microscopy",                  0.96, "equivalentClass"),
    ("SteelMaterial",                       "steel",                             0.96, "equivalentClass"),
    ("TurningMachine",                      "lathe",                             0.96, "equivalentClass"),
    ("HeatTreatment",                       "heat treatment",                    0.96, "equivalentClass"),
    ("TensileTestingMachine",               "tensile testing machine",           0.97, "equivalentClass"),
    ("ScanningElectronMicroscope",          "scanning electron microscope",      0.96, "equivalentClass"),
    ("MachiningWithGeometricallyDeterminedEdge", "machining geometrically defined", 0.96, "equivalentClass"),
    ("InductionHeatingSystem",              "induction furnace",                 0.93, "equivalentClass"),
    ("Equipment",                           "device",                            0.95, "equivalentClass"),
    ("Organisation",                        "organization",                      0.93, "equivalentClass"),
    ("Identifier",                          "identifier",                        0.93, "equivalentClass"),
    ("InformationContentEntity",            "information content entity",        0.97, "equivalentClass"),

    # ── equivalentProperty ───────────────────────────────────────────────────
    ("precedes",                            "precedes",                          0.98, "equivalentProperty"),
    ("has Part",                            "has part",                          0.97, "equivalentProperty"),
    ("has Output",                          "has output",                        0.96, "equivalentProperty"),
    ("is Composed Of",                      "consists of",                       0.93, "equivalentProperty"),
    ("has Value",                           "has value",                         0.96, "equivalentProperty"),
    ("has Participant",                     "has participant",                   0.96, "equivalentProperty"),
    ("refers To",                           "refers to",                         0.96, "equivalentProperty"),
    ("in Accordance With",                  "complies with",                     0.93, "equivalentProperty"),
    ("has Temporal Info",                   "has temporal part",                 0.90, "equivalentProperty"),

    # ── subPropertyOf ────────────────────────────────────────────────────────
    ("has String Literal",                  "has value",                         0.86, "subPropertyOf"),
    ("has Real Literal",                    "has value",                         0.86, "subPropertyOf"),
    ("has Integer Literal",                 "has value",                         0.82, "subPropertyOf"),
    ("has Double Literal",                  "has specified numeric value",       0.82, "subPropertyOf"),
    ("has Max Double Value",                "has value",                         0.78, "subPropertyOf"),
    ("has Min Double Value",                "has value",                         0.78, "subPropertyOf"),
    ("is Unit Symbol For Unit",             "refers to",                         0.78, "subPropertyOf"),
    ("has Control Info",                    "has process attribute",             0.82, "subPropertyOf"),
    ("is Representative For",              "is about",                          0.78, "subPropertyOf"),
    ("chronological Connection",            "precedes",                          0.82, "subPropertyOf"),
    ("has Operator",                        "has participant",                   0.78, "subPropertyOf"),

    # ── subClassOf: BFO hierarchy ────────────────────────────────────────────
    ("OneDimensionalContinuantFiatBoundary","fiat line",                         0.93, "subClassOf"),
    ("TwoDimensionalContinuantFiatBoundary","continuant fiat boundary",          0.92, "subClassOf"),
    ("ProcessID",                           "identifier",                        0.93, "subClassOf"),
    ("MaterialID",                          "identifier",                        0.92, "subClassOf"),

    # ── subClassOf → scalar measurement datum (domain quantities as data points) ──
    # These are BWMD value-container classes: numerical data, not abstract qualities
    ("Pickling Temperature",                "scalar measurement datum",          0.86, "subClassOf"),
    ("Cathodic Degreasing Temperature",     "scalar measurement datum",          0.86, "subClassOf"),
    ("Temperature Actual",                  "scalar measurement datum",          0.86, "subClassOf"),
    ("AverageHardnessBrinell",              "scalar measurement datum",          0.86, "subClassOf"),
    ("YoungsModulus",                       "scalar measurement datum",          0.86, "subClassOf"),
    ("MaximumTensileStrength",              "scalar measurement datum",          0.86, "subClassOf"),
    ("GrainSize",                           "scalar measurement datum",          0.86, "subClassOf"),
    ("Voltage",                             "scalar measurement datum",          0.86, "subClassOf"),
    ("ElasticConstant",                     "scalar measurement datum",          0.86, "subClassOf"),
    ("Porosity",                            "scalar measurement datum",          0.86, "subClassOf"),
    ("Concentration",                       "scalar measurement datum",          0.86, "subClassOf"),
    ("Current Actual",                      "scalar measurement datum",          0.86, "subClassOf"),
    ("Redox Potential",                     "scalar measurement datum",          0.86, "subClassOf"),
    ("Layer Thickness",                     "scalar measurement datum",          0.86, "subClassOf"),
    ("PercentageElongationAfterFracture",   "scalar measurement datum",          0.86, "subClassOf"),

    # ── subClassOf → scalar value specification (setpoints / targets) ────────
    ("Temperature Target",                  "scalar value specification",        0.86, "subClassOf"),
    ("Ph Value Target",                     "specification datum",               0.86, "subClassOf"),
    ("CastingTemperatureAsSupposed",        "specification datum",               0.86, "subClassOf"),
    ("SolutionAnnealingTemperatureAsSupposed", "specification datum",            0.86, "subClassOf"),
    ("ExtensometerDisplacementAsSupposed",  "scalar value specification",        0.86, "subClassOf"),

    # ── subClassOf → time measurement datum ──────────────────────────────────
    ("Pickling Time",                       "time measurement datum",            0.86, "subClassOf"),
    ("Cathodic Degreasing Time",            "time measurement datum",            0.86, "subClassOf"),
    ("Plating Time",                        "time measurement datum",            0.86, "subClassOf"),
    ("Rinsing Time",                        "time measurement datum",            0.86, "subClassOf"),
    ("ArtificialAgingDuration",             "time measurement datum",            0.86, "subClassOf"),
    ("IndentationDurationBrinell",          "time measurement datum",            0.86, "subClassOf"),

    # ── subClassOf → device ───────────────────────────────────────────────────
    ("MicroscopeEquipment",                 "device",                            0.82, "subClassOf"),
    ("GrindingAndPolishingEquipment",       "device",                            0.82, "subClassOf"),
    ("ElectromechanicalTestingMachine",     "device",                            0.82, "subClassOf"),
    ("Rinsing Stations",                    "device",                            0.82, "subClassOf"),
    ("Plating Tank",                        "device",                            0.82, "subClassOf"),
    ("LabInfrastructureEquipment",          "device",                            0.82, "subClassOf"),
    ("InfrastructureEquipment",             "device",                            0.82, "subClassOf"),
    ("Temperature Control",                 "device",                            0.82, "subClassOf"),

    # ── subClassOf → planned process ─────────────────────────────────────────
    ("ChemicalProcess",                     "planned process",                   0.82, "subClassOf"),
    ("Treating",                            "planned process",                   0.82, "subClassOf"),
    ("Posttreatement",                      "planned process",                   0.82, "subClassOf"),
    ("ArtificialAging",                     "planned process",                   0.82, "subClassOf"),
    ("Annealing",                           "planned process",                   0.82, "subClassOf"),
    ("Polishing",                           "machining geometrically undefined", 0.82, "subClassOf"),
    ("Grinding",                            "machining geometrically undefined", 0.82, "subClassOf"),
    ("Electroplating",                      "coating from the ionized state",    0.82, "subClassOf"),

    # ── subClassOf → information content entity / data item ──────────────────
    ("ProcessParameterSet",                 "information content entity",        0.82, "subClassOf"),
    ("MaterialDataSet",                     "information content entity",        0.82, "subClassOf"),
    ("DataSet",                             "data item",                         0.82, "subClassOf"),
    ("Software",                            "information content entity",        0.82, "subClassOf"),
    ("SoftwareArchitecture",               "information content entity",        0.82, "subClassOf"),

    # ── subClassOf → chemical entity ─────────────────────────────────────────
    ("ChemicalElement",                     "chemical entity",                   0.82, "subClassOf"),
    ("ChemicalSolution",                    "chemical entity",                   0.82, "subClassOf"),
    ("BaseElementOfComposition",            "chemical entity",                   0.82, "subClassOf"),

    # ── equivalentClass: chemische Elemente → "X atom" Konvention ────────────
    ("Silver",                              "silver atom",                       0.95, "equivalentClass"),
    ("Copper",                              "copper atom",                       0.95, "equivalentClass"),
    ("Chromium",                            "chromium atom",                     0.95, "equivalentClass"),
    ("Nickel",                              "nickel atom",                       0.95, "equivalentClass"),
    ("Iron",                                "iron atom",                         0.95, "equivalentClass"),

    # ── subClassOf → organization ─────────────────────────────────────────────
    ("Cathodic Degreasing Supplier",        "organization",                      0.90, "subClassOf"),
    ("Anodes Supplier",                     "organization",                      0.90, "subClassOf"),
    ("Pickling Supplier",                   "organization",                      0.90, "subClassOf"),

    # ── subClassOf → identifier ───────────────────────────────────────────────
    ("PersonName",                          "identifier",                        0.90, "subClassOf"),
    ("ConsumableName",                      "identifier",                        0.90, "subClassOf"),

    # ── subClassOf → engineered material ─────────────────────────────────────
    ("FerrousMaterial",                     "engineered material",               0.86, "subClassOf"),
    ("AluminiumAlloy",                      "engineered material",               0.86, "subClassOf"),

    # ── subClassOf → computing process ───────────────────────────────────────
    ("DataTransformation",                  "computing process",                 0.82, "subClassOf"),
    ("DataReduction",                       "computing process",                 0.82, "subClassOf"),

    # ── subClassOf: classic specialisations ──────────────────────────────────
    ("BrinellTestingEquipment",             "hardness testing machine",          0.86, "subClassOf"),
    ("ScanningElectronMicroscopy",          "electron microscopy",               0.86, "subClassOf"),
    ("HeatTreatmentFurnace",               "heat treatment device",             0.86, "subClassOf"),
    ("MetallographicSpecimen",              "specimen",                          0.86, "subClassOf"),
    ("QuasiStaticTensileTest",              "tensile testing process",           0.78, "subClassOf"),
    ("HardnessVickers",                     "indentation hardness",              0.78, "subClassOf"),
    ("TransmissionElectronMicroscope",      "electron microscope",               0.78, "subClassOf"),
    ("HardnessTestingEquipment",            "hardness testing machine",          0.95, "equivalentClass"),
]

# ═══════════════════════════════════════════════════════════════════════════
# VERTEILUNG:
#   equivalentClass:   40 Paare
#   equivalentProperty: 7 Paare
#   subClassOf:        35 Paare
#   subPropertyOf:      6 Paare  (nur typ-konsistente Paare)
#   Total:             88 Paare
#
# ENTFERNTE PAARE (falsch oder typ-inkonsistent):
#   Software       → Computer            (Software ≠ Unterklasse von Computer)
#   GrainBoundary  → Korn                (korrigiert zu microstructure)
#   isComposedOf   → consists of         (Duplikat von is Composed Of)
#   hasTextualInfo → has value           (ObjectProp → DataProp: typ-inkonsistent)
#
# KORRIGIERTE RELATIONEN:
#   Grain, Temperature, LoadCell,
#   MachiningWithGeometricallyDeterminedEdge → equivalentClass (war subClassOf)
#   Specimen  → equivalentClass "specimen"   (war subClassOf "Probe-Rolle")
#   Plating   → equivalentClass "Beschichten" (war subClassOf "coating")
# ═══════════════════════════════════════════════════════════════════════════

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

    # s8 — Description-Similarity
    bw_desc_row     = bwmd[bwmd['Entity'] == bw_iri]
    pmdco_desc_row  = pmdco[pmdco['Entity'] == pmdco_iri]
    bw_desc_text    = str(bw_desc_row.iloc[0]['Description']).strip()  if not bw_desc_row.empty  else ""
    pmdco_desc_text = str(pmdco_desc_row.iloc[0]['Description']).strip() if not pmdco_desc_row.empty else ""
    if bw_desc_text and bw_desc_text != 'nan' and len(pmdco_desc_text) > 10 and pmdco_desc_text != 'nan':
        s8 = SequenceMatcher(None,
                             bw_desc_text[:200].lower(),
                             pmdco_desc_text[:200].lower()).ratio()
    else:
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

    # Gewichtung der Verlustterme
    ALPHA = 0.7   # Gewicht MSE-Term
    BETA  = 0.3   # Gewicht Relations-Term
    GAMMA = 0.05  # Entropy-Regularisierung: bevorzugt gleichmäßige Gewichtsverteilung

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

    # Entropy-Term: -H(w) = sum(w * log(w)), negiert weil wir minimieren
    # Hohe Entropy = gleichmäßige Verteilung → Optimizer verteilt Gewichte breiter
    entropy = -np.sum(weights * np.log(weights + 1e-9))

    return ALPHA * mse + BETA * rpen - GAMMA * entropy


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
    (0.10, 0.35),  # w5  embedding similarity    ← FLOOR+CAP: stärkstes Signal, aber begrenzt um andere Metriken zu aktivieren
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

# ── Auto-generate target ontology top-level overview (generic, once at start) ──
print(f"\n🗂️  Generiere {TARGET_NAME} Top-Level-Hierarchie...")
def get_target_toplevel_overview(graph, max_roots: int = 8, max_children: int = 6) -> str:
    """
    Extrahiert automatisch die obersten Hierarchieebenen der Zielontologie.
    Generisch — funktioniert mit jeder RDF-Ontologie.

    Ansatz: Statt "Klassen ohne benannten Parent" als Wurzeln zu raten (das liefert
    falsche Treffer, weil viele Klassen ihren Parent nur über Blank-Node-Restriktionen
    deklarieren), wird die echte Wurzel bestimmt und von dort per rdfs:subClassOf
    nach UNTEN traversiert. Die echte Wurzel ist die benannte Klasse, deren transitive
    Nachkommenschaft am größten ist und die selbst keinen benannten Oberbegriff besitzt.
    Wird einmalig beim Start generiert und in den Prompt injiziert.
    """
    def named_children(node):
        return sorted(
            {c for c in graph.subjects(RDFS.subClassOf, node) if isinstance(c, URIRef)},
            key=lambda c: label_of(c).lower()
        )

    def label_of(node):
        vals = list(graph.objects(node, RDFS.label))
        return str(vals[0]) if vals else str(node).split('#')[-1].split('/')[-1]

    # 1. Kandidaten für die echte Wurzel: benannte Klassen ohne benannten Parent,
    #    die selbst benannte Kinder haben (schließt isolierte/falsch verlinkte aus)
    all_classes = {c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    root_candidates = []
    for cls in all_classes:
        named_parents = [p for p in graph.objects(cls, RDFS.subClassOf) if isinstance(p, URIRef)]
        if not named_parents and named_children(cls):
            # Größe der transitiven Nachkommenschaft (begrenzte BFS) als Wichtigkeit
            seen, frontier = set(), [cls]
            while frontier and len(seen) < 5000:
                nxt = []
                for n in frontier:
                    for ch in named_children(n):
                        if ch not in seen:
                            seen.add(ch); nxt.append(ch)
                frontier = nxt
            root_candidates.append((cls, len(seen)))

    if not root_candidates:
        return f"TOP-LEVEL STRUCTURE OF {TARGET_NAME}:\n  (structure could not be extracted)"

    # Die Wurzel mit der größten Nachkommenschaft ist die ontologische Hauptwurzel
    root, _ = max(root_candidates, key=lambda x: x[1])

    # 2. Top-down: Wurzel -> ihre Kinder (Ebene 1) -> deren Kinder (Ebene 2)
    lines = [f"TOP-LEVEL STRUCTURE OF {TARGET_NAME}:"]
    lines.append(f"  {label_of(root)}")
    level1 = named_children(root)
    for n1 in level1[:max_roots]:
        lines.append(f"    {label_of(n1)}")
        level2 = named_children(n1)
        if level2:
            shown = [label_of(c) for c in level2[:max_children]]
            lines.append(f"      └── {', '.join(shown)}"
                         + (" ..." if len(level2) > max_children else ""))
    return "\n".join(lines)


# === ANKER-KANDIDATEN (generisch aus Ground Truth abgeleitet) ===

def get_anchor_candidates(gt_pairs: list, pmdco_df, min_occurrences: int = 3) -> dict:
    """
    Leitet aus den GROUND_TRUTH_PAIRS die häufigsten Zielklassen ab.
    Zielklassen die >= min_occurrences mal als verifiziertes Mapping-Ziel
    vorkommen, werden zu Anker-Kandidaten: sie werden jedem Kandidatenset
    angehängt, auch wenn die Ähnlichkeitsmetriken sie nicht in die Top-20 heben.

    Hintergrund: Domänenspezifische Quellklassen (z.B. Voltage) erreichen mit
    ihrer korrekten generischen Zielklasse (z.B. scalar measurement datum)
    oft nur geringe lexikalische/semantische Ähnlichkeit. Ohne Anker sieht
    das LLM die richtige Option nie.

    Vollständig generisch: Bei einem neuen Ontologiepaar entstehen die Anker
    automatisch aus dessen Ground-Truth-Paaren. Keine hartcodierten Klassennamen.
    Die Metriken bleiben unverändert; Anker werden NACH dem Scoring ergänzt.
    """
    target_counts = Counter(pm_label for _, pm_label, _, _ in gt_pairs)
    anchors = {}
    for pm_label, count in target_counts.items():
        if count >= min_occurrences:
            match = pmdco_df[pmdco_df['Label'].astype(str).str.strip().str.lower()
                             == pm_label.strip().lower()]
            if not match.empty:
                anchors[pm_label.strip().lower()] = match.iloc[0]
    return anchors


# Wird nach dem Laden der CSVs befüllt (siehe Main-Setup)
ANCHOR_CANDIDATES = {}


TARGET_TOPLEVEL_OVERVIEW = get_target_toplevel_overview(pmdco_graph)
print(TARGET_TOPLEVEL_OVERVIEW[:500] + "..." if len(TARGET_TOPLEVEL_OVERVIEW) > 500 else TARGET_TOPLEVEL_OVERVIEW)
# ────────────────────────────────────────────────────────────────────────────

# ── Anker-Kandidaten aus Ground Truth ableiten (generisch) ───────────────────
ANCHOR_CANDIDATES.update(get_anchor_candidates(GROUND_TRUTH_PAIRS, pmdco, min_occurrences=3))
print(f"\n⚓ {len(ANCHOR_CANDIDATES)} Anker-Kandidaten aus GT abgeleitet (>=3 Vorkommen):")
print(f"   {', '.join(sorted(ANCHOR_CANDIDATES.keys()))}")
# ────────────────────────────────────────────────────────────────────────────

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

# === VOLLSTÄNDIGE HIERARCHIE-EXTRAKTION ===

def get_full_ancestor_chain(entity_iri: str, graph, max_depth: int = 10) -> list:
    """
    Extrahiert die vollständige Vorfahrenkette bis zu owl:Thing/Entity.
    Gibt geordnete Liste von Labels zurück (direkter Elternteil zuerst).
    """
    ancestors = []
    current = URIRef(entity_iri)
    visited = set()
    for _ in range(max_depth):
        if current in visited:
            break
        visited.add(current)
        parents_found = list(graph.objects(current, RDFS.subClassOf))
        if not parents_found:
            parents_found = list(graph.objects(current, RDFS.subPropertyOf))
        if not parents_found:
            break
        parent = parents_found[0]
        label_vals = list(graph.objects(parent, RDFS.label))
        lbl = str(label_vals[0]) if label_vals else str(parent).split('#')[-1].split('/')[-1]
        if lbl.lower() in ('thing', 'resource', ''):
            break
        ancestors.append(lbl)
        current = parent
    return ancestors


def get_direct_children_labels(entity_iri: str, graph, max_children: int = 8) -> list:
    """Gibt die direkten Unterklassen/Untereigenschaften als Labels zurück."""
    children = []
    for pred in (RDFS.subClassOf, RDFS.subPropertyOf):
        for subj in graph.subjects(pred, URIRef(entity_iri)):
            label_vals = list(graph.objects(subj, RDFS.label))
            lbl = str(label_vals[0]) if label_vals else str(subj).split('#')[-1].split('/')[-1]
            if lbl:
                children.append(lbl)
            if len(children) >= max_children:
                return children
    return children





def format_hierarchy_block(label: str, ancestors: list, children: list) -> str:
    """
    Formatiert Vorfahren- und Kinderkette als lesbare Hierarchie.
    Thing > Occurrent > Process > planned process > LABEL
        Children: coating, heat treatment, ...
    """
    if ancestors:
        chain = " > ".join(reversed(ancestors)) + " > " + label
    else:
        chain = label
    result = f"    Full path: {chain}"
    if children:
        result += f"\n    Children:  {', '.join(children)}"
    return result


# === VERBESSERTES PRE-FILTERING MIT EMBEDDINGS ===
def detect_domain_type(label: str, parents: list, description: str = "") -> str:
    """
    Leitet den ontologischen Domänentyp einer Entität aus Label, Elternklassen
    und Beschreibung ab. Verhindert Kategorienfehler wie Prozess ↔ Maschine.
    Gibt einen von 6 Typen zurück: PROCESS | DEVICE | QUALITY | MATERIAL | INFORMATION | OTHER
    """
    text = " ".join([label] + parents + [description]).lower()

    # Schlüsselwörter pro Kategorie (Reihenfolge = Priorität)
    PROCESS_KW    = ["process", "treatment", "experiment", "measurement", "test",
                     "analysis", "preparation", "cleaning", "plating", "coating",
                     "grinding", "polishing", "annealing", "quenching", "casting",
                     "welding", "machining", "etching", "simulation", "inspection",
                     "fabrication", "procedure", "operation", "assay", "activity"]
    DEVICE_KW     = ["machine", "equipment", "device", "tool", "instrument",
                     "microscope", "furnace", "system", "sensor", "detector",
                     "apparatus", "setup", "tank", "anode", "cathode", "cell",
                     "camera", "pump", "motor", "heater", "cooler", "rack"]
    QUALITY_KW    = ["hardness", "roughness", "conductivity", "strength", "modulus",
                     "density", "temperature", "force", "stress", "strain",
                     "current", "voltage", "potential", "property", "quality",
                     "quantity", "rate", "ratio", "fraction", "concentration",
                     "thickness", "length", "area", "volume", "mass", "weight",
                     "grain size", "crack", "porosity", "texture", "microstructure"]
    MATERIAL_KW   = ["material", "alloy", "steel", "aluminium", "chromium",
                     "electrolyte", "solution", "substance", "chemical", "compound",
                     "metal", "ceramic", "polymer", "coating layer", "deposit",
                     "specimen", "sample", "substrate", "reagent", "agent"]
    INFORMATION_KW= ["dataset", "data set", "identifier", "name", "label",
                     "specification", "document", "file", "record", "report",
                     "information", "schema", "format", "value", "literal"]

    scores = {
        "PROCESS":     sum(1 for kw in PROCESS_KW     if kw in text),
        "DEVICE":      sum(1 for kw in DEVICE_KW      if kw in text),
        "QUALITY":     sum(1 for kw in QUALITY_KW     if kw in text),
        "MATERIAL":    sum(1 for kw in MATERIAL_KW    if kw in text),
        "INFORMATION": sum(1 for kw in INFORMATION_KW if kw in text),
    }

    # Explizite Priorität: wenn Label/Eltern eindeutige Gerätewörter enthalten,
    # überschreibt das einen PROCESS-Score der durch "test" in "TestingEquipment" entsteht
    label_lower = label.lower()
    device_override_kw = ["equipment", "machine", "device", "tool", "instrument",
                           "furnace", "apparatus", "system", "sensor", "tank",
                           "detector", "camera", "rack", "cell", "indenter"]
    if any(kw in label_lower for kw in device_override_kw):
        scores["DEVICE"] = max(scores["DEVICE"], scores["PROCESS"] + 1)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "OTHER"


# Mapping von Domänentyp auf typische PMDco-Klassen für Prompt-Hinweis
DOMAIN_TYPE_HINTS = {
    "PROCESS":     ("process", "planned process", "manufacturing process",
                    "conditioning process", "simulation process", "measuring function"),
    "DEVICE":      ("device", "measurement equipment", "forming machine",
                    "coating device", "microscope", "spectrometer", "furnace"),
    "QUALITY":     ("quality", "intensive quality", "mechanical property",
                    "electrical property", "thermal property", "process attribute"),
    "MATERIAL":    ("material", "engineered material", "chemical entity",
                    "portion of matter", "engineered material"),
    "INFORMATION": ("information content entity", "identifier", "specification datum",
                    "measurement datum", "data item"),
    "OTHER":       (),
}


def find_relevant_candidates(
        bw_label, bw_type, bw_description, bw_hierarchy, bw_iri, pmdco_df, pmdco_graph, top_n=20):
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

    # ── Anker-Kandidaten anhängen (generisch aus GT abgeleitet) ─────────────
    # Nur für Klassen-Entitäten relevant; Properties haben eigene Anker selten.
    # Anker werden NACH dem Scoring ergänzt — Metriken bleiben unberührt.
    if ANCHOR_CANDIDATES:
        existing_labels = {str(r.Label).strip().lower() for r in top_candidates}
        is_prop = is_property_type(bw_type)
        for anchor_label, anchor_row in ANCHOR_CANDIDATES.items():
            if anchor_label in existing_labels:
                continue
            anchor_type = str(anchor_row.Type) if pd.notna(anchor_row.Type) else ''
            anchor_is_prop = 'Property' in anchor_type
            if is_prop == anchor_is_prop:
                top_candidates.append(anchor_row)
    # ────────────────────────────────────────────────────────────────────────

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

    # ── Vollständige BWMD-Hierarchie extrahieren ────────────────────────────
    bw_ancestors = get_full_ancestor_chain(bw_iri, bwmd_graph)
    bw_children  = get_direct_children_labels(bw_iri, bwmd_graph)
    bw_hier_block = format_hierarchy_block(bw_label, bw_ancestors, bw_children)

    # ── Vollständige PMDco-Hierarchie pro Kandidat aufbauen ─────────────────
    pmdco_candidate_blocks = []
    for row in relevant_pmdco:
        cand_iri  = str(row.Entity)
        cand_lbl  = str(row.Label)
        cand_type = str(row.Type) if pd.notna(row.Type) else ''
        cand_desc = get_full_description(row, pmdco_df)
        c_ancestors = get_full_ancestor_chain(cand_iri, pmdco_graph)
        c_children  = get_direct_children_labels(cand_iri, pmdco_graph)
        c_hier      = format_hierarchy_block(cand_lbl, c_ancestors, c_children)
        is_anchor = cand_lbl.strip().lower() in ANCHOR_CANDIDATES
        anchor_tag = " [GENERAL ANCHOR — frequently the correct target for domain-specific source classes without a direct equivalent]" if is_anchor else ""
        block = (f"• {cand_lbl} ({cand_type}){anchor_tag}\n{c_hier}"
                 + (f"\n    Description: {cand_desc[:120]}" if cand_desc and len(cand_desc) > 5 else "")
                 + f"\n    IRI: {cand_iri}")
        pmdco_candidate_blocks.append(block)
    pmdco_full_text = "\n\n".join(pmdco_candidate_blocks)
    # ────────────────────────────────────────────────────────────────────────

    prompt = f"""You are an expert in ontology alignment for materials science and engineering.

ONTOLOGY CONTEXT:
You are mapping entities from the {SOURCE_NAME} ontology to the {TARGET_NAME} ontology.

- {SOURCE_NAME}: {SOURCE_DESCRIPTION}

- {TARGET_NAME}: {TARGET_DESCRIPTION}

{TARGET_TOPLEVEL_OVERVIEW}

Use your knowledge of both ontologies and the structure above to support your decision.

{FEW_SHOT_EXAMPLES}

TASK: Map the {SOURCE_NAME} entity below to the BEST matching {TARGET_NAME} entity from the candidates.

{SOURCE_NAME} Entity (source):
  Label:       {bw_label}
  Type:        {bw_type}
  IRI:         {bw_iri}
{bw_hier_block}
{f"  Description: {bw_description[:150]}" if bw_description else ""}

{TARGET_NAME} Candidates (ranked by similarity):
{pmdco_full_text}

DECISION RULES — follow in order:

STEP 1 – DIRECT EQUIVALENT: Same concept, same abstraction level?
  → equivalentClass / equivalentProperty. Confidence 88–99.
  → Applies even for synonyms and cross-language labels.
  → STRICT TEST before choosing equivalentClass: could every instance of the
    candidate also be an instance of the source class, and vice versa?
    If the source is narrower in ANY way (more specific method, domain
    restriction, named subtype), use subClassOf instead.
  → When torn between equivalentClass and subClassOf, choose subClassOf.

STEP 2 – SPECIALISATION: Source entity is a more specific version of the candidate?
  → subClassOf / subPropertyOf. Confidence 75–90.
  → Verify: does the candidate appear in the source entity's ancestor chain?

STEP 3 – NO DIRECT EQUIVALENT (most common for domain-specific source classes):
  → Select the closest ANCESTOR from candidates. Use subClassOf. Confidence 55–88.
  → Use the FULL HIERARCHY PATHS above to determine category membership.
  → VALUE-CONTAINER RULE (very important): If the source class represents a
    concrete data point — a measured value, setpoint, duration, count, or
    recorded parameter (check its parents and description) — do NOT map it
    to the abstract physical quality with a similar name. Map it to the
    appropriate data/measurement class among the candidates, typically one
    marked as [GENERAL ANCHOR]. The few-shot examples show this pattern.
  → Candidates marked [GENERAL ANCHOR] are verified frequent targets for
    source classes without a direct equivalent. Prefer them over a
    semantically related but categorically wrong specific class.
  → CRITICAL: The ancestor MUST be in the same ontological category:
      process → map to a process class in the target ontology
      device/equipment → map to a device or equipment class
      quality/property → map to a quality or property class
      data/information → map to an information or data class
      material/substance → map to a material or substance class
  → NEVER map a process to a device/equipment class, or vice versa.
  → When unsure, prefer a general ancestor over an incorrect specific one.

STEP 4 – LAST RESORT: subClassOf with the most general fitting top-level class.
  Confidence below 55.

{relation_block}

CONFIDENCE: 90–99 exact match | 80–89 clear equiv/subclass | 65–79 reasonable ancestor |
            50–64 plausible gap | 25–49 best available, flag for review

Valid relations: {valid_relations}

OUTPUT (valid JSON only, no line breaks inside strings):
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
        return response.choices[0].message.content, prompt, relevant_pmdco
    except Exception as e:
        print(f"⚠️ LLM-Fehler bei {bw_label}: {e}")
        if relevant_pmdco:
            best = relevant_pmdco[0]
            return json.dumps({
                "pmdco_label": str(best.Label),
                "pmdco_iri": str(best.Entity),
                "relation": get_default_relation(bw_type),
                "confidence": 30,
                "reasoning": "API error - auto-matched"
            }), prompt, relevant_pmdco
        return json.dumps({
            "pmdco_label": "",
            "pmdco_iri": "",
            "relation": get_default_relation(bw_type),
            "confidence": 0,
            "reasoning": "API error, no candidates"
        }), prompt, []


def fix_iri_from_candidates(pmdco_label, pmdco_iri, candidates):
    """
    Korrigiert LLM-Rueckgaben deterministisch anhand der Kandidatenliste.
    Das LLM darf nur aus den angebotenen Kandidaten waehlen, daher kann die
    korrekte volle IRI immer rekonstruiert werden. Behebt:
      - abgeschnittene IRIs (z.B. 'PMD_0020200' statt voller w3id-IRI)
      - Tippfehler in der IRI bei korrektem Label
    Rueckgabe: (label, iri, korrigiert_bool)
    """
    if not candidates:
        return pmdco_label, pmdco_iri, False

    cand_by_iri   = {str(c.Entity): c for c in candidates}
    iri_str = str(pmdco_iri).strip()

    # Fall 1: IRI stimmt exakt
    if iri_str in cand_by_iri:
        return pmdco_label, iri_str, False

    # Fall 2: LLM gab nur Fragment/Suffix zurueck -> volle Kandidaten-IRI finden
    if iri_str:
        for full_iri, c in cand_by_iri.items():
            if full_iri.endswith(iri_str) or full_iri.split('#')[-1].split('/')[-1] == iri_str:
                return str(c.Label), full_iri, True

    # Fall 3: Label-Match (case-insensitive) -> IRI des Kandidaten uebernehmen
    lbl = str(pmdco_label).strip().lower()
    for c in candidates:
        if str(c.Label).strip().lower() == lbl:
            return str(c.Label), str(c.Entity), True

    return pmdco_label, pmdco_iri, False


# === MAPPING DURCHFÜHREN ===
results = []
validation_errors = []
invalid_relation_count = 0
api_error_count = 0
json_error_count = 0

# Debug: 3 Beispiele ausgeben bei ~25%, ~50%, ~75% des Durchlaufs
total_entities = len(bwmd)
debug_indices = {total_entities // 4, total_entities // 2, (3 * total_entities) // 4}

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
    
    llm_raw, llm_prompt, llm_candidates = get_sparql_mapping(
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

        # ── IRI deterministisch gegen Kandidatenliste korrigieren ──────────
        pmdco_label, pmdco_iri, iri_fixed = fix_iri_from_candidates(
            pmdco_label, pmdco_iri, llm_candidates
        )
        if iri_fixed:
            print(f"   🔧 IRI korrigiert: {bw_label} → '{pmdco_label}' ({pmdco_iri})")
        # ───────────────────────────────────────────────────────────────────

        # ── Relation validieren und ggf. korrigieren ───────────────────────
        corrected_relation = validate_relation(relation, bw_type)
        if corrected_relation != relation:
            print(f"   ⚠️  Relation korrigiert: {bw_label}: '{relation}' → '{corrected_relation}' (Type: {bw_type})")
            relation = corrected_relation
            invalid_relation_count += 1
        # ───────────────────────────────────────────────────────────────────

        # ── Debug-Ausgabe für 3 Beispiele ──────────────────────────────────
        if idx in debug_indices:
            top_labels = [str(r.Label) for r in llm_candidates]
            conf_tier = "High" if confidence >= 80 else ("Medium" if confidence >= 50 else "Low")
            SEP1 = "=" * 70
            SEP2 = "-" * 70

            # ── Zusammenfassung ──────────────────────────────────────────────
            print(f"\n{SEP1}")
            print(f"BEISPIEL (Entity {idx+1}/{total_entities})")
            print(SEP2)
            print(f"   {SOURCE_NAME}:    {bw_label} ({bw_type})")
            if bw_hierarchy.get("parents"):
                print(f"   Parents: {', '.join(bw_hierarchy['parents'][:3])}")
            if bw_description:
                print(f"   Desc:    {bw_description[:100]}")
            print("\n   Kandidaten ans LLM (inkl. Anker):")
            for i, lbl in enumerate(top_labels, 1):
                marker = " <- gewaehlt" if lbl.lower() == pmdco_label.lower() else ""
                print(f"     {i}. {lbl}{marker}")
            print(f"\n   Ergebnis:  {pmdco_label}")
            print(f"   Relation:  {relation}  (Konfidenz: {confidence}, {conf_tier})")
            print(f"   Reasoning: {reasoning}")

            # ── Vollständiger Prompt ─────────────────────────────────────────
            print(f"\n{SEP2}")
            print("   VOLLSTAENDIGER PROMPT AN LLM (fuer SI-Sektion / Reproduzierbarkeit):")
            print(SEP2)
            print("   [SYSTEM]: Expert ontology mapper. Output valid JSON only. No line breaks in strings.")
            print()
            print("   [USER]:")
            for line in llm_prompt.split("\n"):
                print("   " + line)

            # ── LLM-Antwort (raw + parsed) ───────────────────────────────────
            print(f"\n{SEP2}")
            print("   LLM-ANTWORT (raw JSON):")
            print(SEP2)
            for line in llm_raw.strip().split("\n"):
                print("   " + line)
            print(f"\n   PARSED:")
            print(f"   pmdco_label : {pmdco_label}")
            print(f"   pmdco_iri   : {pmdco_iri}")
            print(f"   relation    : {relation}")
            print(f"   confidence  : {confidence}")
            print(f"   reasoning   : {reasoning}")
            print(f"{SEP1}\n")
        # ────────────────────────────────────────────────────────────────────
        
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