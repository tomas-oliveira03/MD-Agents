from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import re

def load_papers(paper_paths):
    papers = []
    for path in paper_paths:
        with open(path, "r", encoding="utf-8") as f:
            papers.append(f.read())
    return papers

def chunk_text(text, max_words=150, overlap=50):
    # Improved chunking with overlap for better context preservation
    words = text.split()
    chunks = []
    
    # Clean and normalize text before chunking
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Create overlapping chunks
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) > 20:  # Only include chunks with substantial content
            chunks.append(chunk)
    
    return chunks

def create_embeddings(papers):
    # Using a more powerful embedding model for better semantic understanding
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    chunks = []
    
    for idx, paper in enumerate(papers):
        # Split paper into paragraphs first
        paragraphs = re.split(r'\n\s*\n', paper)
        
        for para_idx, para in enumerate(paragraphs):
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            for c in chunk_text(para):
                # Add more metadata to help with retrieval
                chunks.append({
                    "paper_id": idx+1,
                    "paragraph_id": para_idx,
                    "content": c,
                    "embedding": embedder.encode(c, normalize_embeddings=True)
                })
    
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=5, threshold=0.25):
    # Using the same embedding model as in create_embeddings
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Normalize and encode the query
    query = re.sub(r'\s+', ' ', query).strip()
    query_emb = embedder.encode(query, normalize_embeddings=True)
    
    # Calculate similarities
    similarities = []
    for chunk in chunks:
        sim_score = cosine_similarity([query_emb], [chunk['embedding']])[0][0]
        # Only add chunks that meet the threshold criteria
        if sim_score >= threshold:
            similarities.append((chunk, sim_score))
    
    # Sort by similarity
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # If we don't have enough results above threshold, adjust the message
    if len(ranked) == 0:
        print(f"Warning: No chunks found with similarity above threshold {threshold}")
        # Fall back to top results but with a higher threshold to avoid completely irrelevant content
        for chunk in chunks:
            sim_score = cosine_similarity([query_emb], [chunk['embedding']])[0][0]
            if sim_score >= threshold * 0.8:  # Use slightly lower threshold as fallback
                similarities.append((chunk, sim_score))
        ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Limit to top_k results
    ranked = ranked[:top_k]
    
    # Add score to each chunk and store global scores for main.py
    global ranked_scores
    ranked_scores = [r[1] for r in ranked]
    
    # Add score to each chunk
    result_chunks = []
    for chunk, score in ranked:
        chunk_copy = chunk.copy()  # Create a copy to avoid modifying the original
        chunk_copy['score'] = score
        result_chunks.append(chunk_copy)
    
    # Print some debug info about similarity scores
    print(f"Top similarity scores: {[round(r[1], 2) for r in ranked]}")
    print(f"Number of chunks retrieved: {len(result_chunks)}")
    
    return result_chunks
