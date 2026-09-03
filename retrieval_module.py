import chromadb
from sentence_transformers import SentenceTransformer

# ---- Sample SOP documents ----
SAMPLE_DOCS = [
    """Valve Inspection Procedure: Before inspecting any valve, ensure the line 
    is depressurized and locked out per plant safety protocol. Check the valve 
    body for external corrosion, cracks, or leaks around the packing gland. 
    Operate the valve through its full range of motion to confirm smooth actuation 
    without excessive resistance. Inspect the seal quarterly for wear, and replace 
    if any deformation is visible. Record the valve's position (open/closed) and 
    any abnormal noise during operation. Flag any valve that fails to seat fully 
    for immediate maintenance review.""",

    """Pressure Gauge Calibration: Pressure gauges must be checked against a 
    calibrated reference gauge every six months. Isolate the gauge from the 
    process line before testing. A deviation of more than 2% from the reference 
    reading requires recalibration or replacement. Visually inspect the gauge 
    face for fogging, cracked glass, or a stuck needle, all of which indicate 
    the gauge should be taken out of service immediately. Document calibration 
    date, technician name, and deviation percentage in the equipment log.""",

    """Equipment Photo Documentation Guidelines: When photographing equipment 
    for inspection records, capture the full unit along with a close-up of the 
    nameplate showing model and serial number. Include any visible corrosion, 
    leaks, or damage in a separate close-up shot. Photos should be taken in 
    adequate lighting and timestamped automatically. Store all inspection photos 
    alongside the corresponding written inspection report for audit purposes."""
]

# ---- Model + persistent DB setup ----
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="sop_docs")


def chunk_text(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Split a document into smaller chunks of N sentences each."""
    # Clean up whitespace from the triple-quoted strings first
    clean_text = " ".join(text.split())
    # Naive sentence split on ". " — good enough for this dataset
    sentences = [s.strip() for s in clean_text.split(". ") if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i + sentences_per_chunk])
        if not chunk.endswith("."):
            chunk += "."
        chunks.append(chunk)
    return chunks


def build_index():
    # Clear old data first so re-running doesn't duplicate chunks
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunk_id = 0
    for doc in SAMPLE_DOCS:
        chunks = chunk_text(doc, sentences_per_chunk=2)
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk]
            )
            chunk_id += 1


def search_sop(query: str) -> list[str]:
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    return results['documents'][0]


if __name__ == "__main__":
    build_index()
    print("Index built successfully with chunked documents.")