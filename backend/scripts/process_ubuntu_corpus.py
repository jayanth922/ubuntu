#!/usr/bin/env python3
"""
Process Ubuntu Dialogue Corpus for training and RAG
"""
import os
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

def download_datasets():
    """Download datasets from Kaggle using API (requires kaggle.json credentials)"""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    print("Downloading Ubuntu Dialogue Corpus...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('rtatman/ubuntu-dialogue-corpus', 
                                          path=RAW_DIR, unzip=True)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Please download manually from: https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus")

def process_dialogues():
    """Process dialogues into QA pairs"""
    print("Processing dialogue data...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Path to the dialogue file - adjust if needed
    dialogue_file = os.path.join(RAW_DIR, 'dialogueText.csv')
    
    if not os.path.exists(dialogue_file):
        print(f"File not found: {dialogue_file}")
        return
    
    # Load dialogues (using only a subset for memory constraints)
    df = pd.read_csv(dialogue_file, nrows=100000)
    
    # Process dialogues into question-answer pairs
    qa_pairs = []
    current_dialogue = []
    current_id = None
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if current_id != row['dialogueID']:
            # Process previous dialogue
            if len(current_dialogue) >= 2:
                for i in range(0, len(current_dialogue) - 1):
                    qa_pairs.append({
                        "id": f"{current_id}_{i}",
                        "content": current_dialogue[i],
                        "response": current_dialogue[i+1],
                        "source": "Ubuntu Dialogue Corpus"
                    })
            
            # Start new dialogue
            current_dialogue = [row['text']]
            current_id = row['dialogueID']
        else:
            current_dialogue.append(row['text'])
    
    # Process final dialogue
    if current_id is not None and len(current_dialogue) >= 2:
        for i in range(0, len(current_dialogue) - 1):
            qa_pairs.append({
                "id": f"{current_id}_{i}",
                "content": current_dialogue[i],
                "response": current_dialogue[i+1],
                "source": "Ubuntu Dialogue Corpus"
            })
    
    # Save processed data
    print(f"Saving {len(qa_pairs)} QA pairs...")
    with open(os.path.join(PROCESSED_DIR, 'ubuntu_qa_pairs.json'), 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    return qa_pairs

def build_faiss_index(qa_pairs):
    """Build FAISS index for vector search"""
    print("Building FAISS index...")
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    contents = [pair['content'] for pair in qa_pairs]
    embeddings = model.encode(contents, show_progress_bar=True)
    
    # Normalize embeddings
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, os.path.join(PROCESSED_DIR, 'ubuntu_faiss_index.bin'))
    
    print("FAISS index built and saved!")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_datasets()
    qa_pairs = process_dialogues()
    if qa_pairs:
        build_faiss_index(qa_pairs)
    print("Processing complete!")