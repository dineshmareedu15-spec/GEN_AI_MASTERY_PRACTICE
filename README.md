# GEN_AI_MASTERY_PRACTICE

Practice scripts for Word Embeddings and the RAG pipeline (chunking -> ChromaDB -> retrieval -> LLM).
Each folder uses `Practice_1.py`, `Practice_2.py`, ... numbered from 1 inside the folder.

| Folder | Files | Topic |
|---|---|---|
| `Word_Embeddings` | Practice_1-3 | Word2Vec: min_count, save/load, PCA plot |
| `RAG_Document_Chunking_Step1` | Practice_1-3 | overlap sweep, paragraph packer, comparing chunkers |
| `RAG_Step2_Storing_Chunks_in_ChromaDB` | Practice_1-3 | metadata + checks, upsert/delete, PersistentClient |
| `RAG_Step3_Retrieving_Relevant_Context` | Practice_1-3 | similarity bars, metadata filters, threshold calibration |
| `RAG_Step4_Passing_Context_to_Gemini_Groq_LLM` | Practice_1-5 | shared setup, refusal checks, temperature, JSON output, with/without context |
| `Improving_RAG_Hybrid_Search_Empty_Retrievals (1)` | Practice_1-3 | RRF k value, fallback retrieval, stop-word retry |

## Setup
```
pip install -r requirements.txt
```
Create a `.env` file next to the Step 4 scripts (never commit it):
```
GROQ_API_KEY=your_key_here
# or
GEMINI_API_KEY=your_key_here
```

## Run
Run each script from inside its own folder, e.g.
```
cd RAG_Step3_Retrieving_Relevant_Context
python Practice_1.py
```
In `RAG_Step4_...`, Practice_2 to Practice_5 import `Practice_1.py`, so keep them in the same folder.
