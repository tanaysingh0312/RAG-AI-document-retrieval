# RAG AI Document Retrieval System

Production-grade Retrieval-Augmented Generation (RAG) system for enterprise knowledge management and document-grounded question answering.

The system transforms heterogeneous documents (PDFs, images, tables, manuals) into a structured semantic knowledge layer using OCR, layout detection, embeddings, and hybrid retrieval. A locally deployed LLM generates context-grounded responses with minimal hallucination, enabling reliable access to large-scale document repositories.

---

## Architecture

```
Document Sources → OCR & Layout Detection → Semantic Chunking → Embeddings → Vector Storage → Hybrid Retrieval → Reranking → LLM → Response
```

---

## Pipeline Overview

| Stage | Component | Description |
|------|----------|------------|
| Ingestion | Document Loader | Handles PDFs, images, manuals |
| OCR | PaddleOCR | Extracts text from scanned documents |
| Layout Detection | Unstructured | Identifies structure (tables, headings, sections) |
| Chunking | Semantic Chunker | Splits text with metadata |
| Embeddings | nomic-embed-text (768D) | Generates semantic vectors |
| Storage | ChromaDB | Stores vector representations |
| Retrieval | Hybrid (Vector + BM25) | Combines semantic and keyword search |
| Reranking | MMR + Cross-Encoder | Improves relevance and diversity |
| Generation | Mistral / Phi-3 (Ollama) | Produces grounded responses |

---

## Capabilities

### Document Processing
- Supports PDFs, scanned images, and structured documents  
- OCR-based text extraction  
- Layout-aware parsing of tables and sections  

### Semantic Indexing
- Metadata-enriched chunking  
- High-dimensional embeddings for accurate retrieval  

### Hybrid Retrieval
- Vector similarity search  
- BM25 keyword-based search  
- Fuzzy matching using RapidFuzz  

### Relevance Optimization
- Multi-stage reranking pipeline  
- Maximal Marginal Relevance (MMR)  
- Cross-encoder scoring  

### Response Generation
- Local LLM inference via Ollama  
- Context-restricted prompting  
- Reduced hallucination  

---

## Tech Stack

**AI / NLP**
- Ollama  
- Mistral / Phi-3  
- nomic-embed-text  

**Document Processing**
- PaddleOCR  
- Unstructured  

**Search & Retrieval**
- ChromaDB  
- BM25  
- RapidFuzz  

**Backend**
- Python  
- FastAPI  

**Infrastructure**
- Docker  
- Local LLM deployment  

---

## Repository Structure

| Path | Description |
|------|------------|
| docs_raw/ | Raw input documents |
| docs_processed/ | OCR and parsed outputs |
| rag_ingest.py | Document ingestion pipeline |
| rag.py | Retrieval and reranking logic |
| rag_pipeline.py | End-to-end workflow |
| ocr_service.py | OCR service |
| layout_detector_service.py | Layout detection |
| semantic_chunker.py | Chunking and metadata |
| main.py | FastAPI application |
| auth.py | Authentication layer |

---

## Running the System

### Install dependencies
```
pip install -r requirements.txt
```

### Start API
```
uvicorn main:app --reload
```

### Run pipeline
```
python rag_pipeline.py
```

---

## Design Principles

- Grounded responses only (no external hallucination)  
- Local inference for privacy and control  
- Hybrid retrieval for accuracy  
- Metadata-aware filtering  
- Modular and extensible architecture  

---

## Contributors

Tanay Singh — AI / Backend Development  
https://github.com/tanaysingh0312  

Jahnavi Dave — Research / Backend Development  
https://github.com/jahnavikdave1834  

---

## License

Apache License 2.0
