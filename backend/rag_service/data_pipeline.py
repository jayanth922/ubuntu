#!/usr/bin/env python3
import os
import logging
import pandas as pd
import json
import gzip
import hashlib
import shutil
import requests
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataPipeline")

class UbuntuCorpusProcessor:
    """
    Comprehensive processor for the Ubuntu Dialogue Corpus.
    Handles data acquisition, processing, cleaning, and preparation for RAG.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "/data/raw",
        processed_data_dir: str = "/data/processed",
        index_data_dir: str = "/data/index",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the Ubuntu Corpus Processor.
        
        Args:
            raw_data_dir: Directory for raw data files
            processed_data_dir: Directory for processed data files
            index_data_dir: Directory for index files
            chunk_size: Size of text chunks for RAG
            chunk_overlap: Overlap between chunks
            max_samples: Maximum number of samples to process (None for all)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.index_data_dir = Path(index_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_samples = max_samples
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.index_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.raw_file = self.raw_data_dir / "ubuntu_dialogs.csv"
        self.processed_file = self.processed_data_dir / "ubuntu_qa_pairs.json"
        self.chunked_file = self.processed_data_dir / "ubuntu_chunked.json"
        self.metadata_file = self.processed_data_dir / "metadata.json"
        
        # Stats
        self.stats = {
            "raw_dialogs": 0,
            "processed_qa_pairs": 0,
            "chunks": 0,
            "processing_time_seconds": 0,
            "last_processed": None,
            "version": "1.0.0"
        }
    
    def download_corpus(self, force: bool = False) -> bool:
        """
        Download the Ubuntu Dialogue Corpus if not already present.
        
        Args:
            force: Force download even if file exists
            
        Returns:
            bool: True if download was successful or file exists
        """
        if self.raw_file.exists() and not force:
            logger.info(f"Using existing file: {self.raw_file}")
            return True
        
        # URLs for the corpus - adjust as needed based on actual source
        urls = [
            "https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_dialogs.csv?dl=1",
            "https://github.com/rkadlec/ubuntu-ranking-dataset-creator/raw/master/src/dialogs/ubuntu_dialogs.csv"
        ]
        
        for url in urls:
            try:
                logger.info(f"Downloading Ubuntu Dialogue Corpus from {url}")
                response = requests.get(url, stream=True)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
                    
                    with open(self.raw_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    
                    progress_bar.close()
                    logger.info(f"Download complete: {self.raw_file}")
                    return True
                else:
                    logger.warning(f"Failed to download from {url}, status code: {response.status_code}")
            
            except Exception as e:
                logger.error(f"Error downloading from {url}: {e}")
        
        # If we reach here, all download attempts failed
        logger.error("All download attempts failed")
        
        # Create instructions for manual download
        manual_instructions = """
        Unable to download Ubuntu Dialogue Corpus automatically.
        
        Please download it manually:
        1. Go to: https://github.com/rkadlec/ubuntu-ranking-dataset-creator
        2. Download the ubuntu_dialogs.csv file
        3. Place it in the raw data directory: {}
        """.format(self.raw_data_dir)
        
        with open(self.raw_data_dir / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
            f.write(manual_instructions)
        
        logger.info("Created manual download instructions")
        return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text from the corpus.
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Replace common artifacts
        replacements = {
            "__eou__": ".",  # End of utterance
            "__eot__": ".",  # End of turn
            "\r": " ",
            "\n": " ",
            "  ": " ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Replace multiple spaces with single space
        while "  " in text:
            text = text.replace("  ", " ")
        
        # Strip whitespace
        return text.strip()
    
    def is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """
        Determine if a QA pair is valid and useful.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            bool: True if the pair is valid
        """
        # Minimum length requirements
        if len(question) < 10 or len(answer) < 10:
            return False
        
        # Maximum length check (avoid extremely long texts)
        if len(question) > 1000 or len(answer) > 5000:
            return False
        
        # Check for meaningful content
        low_value_phrases = [
            "i don't know", 
            "no idea", 
            "can't help", 
            "google it",
            "what?",
            "idk",
            "search",
            "try again"
        ]
        
        # Check if answer contains any low-value phrases
        if any(phrase in answer.lower() for phrase in low_value_phrases):
            return False
        
        # Check if answer is too similar to question (likely not informative)
        if answer.lower() in question.lower() or question.lower() in answer.lower():
            return False
            
        # Check if likely Ubuntu-related
        ubuntu_indicators = [
            "ubuntu", "linux", "apt", "sudo", "command", "terminal", 
            "package", "install", "dpkg", "system", "kernel", "bash",
            "file", "directory", "mount", "server", "config", "desktop",
            "gnome", "unity", "xorg", "driver", "update", "repository"
        ]
        
        # Check if either question or answer contains Ubuntu terminology
        is_ubuntu_related = any(
            term in question.lower() or term in answer.lower() 
            for term in ubuntu_indicators
        )
        
        return is_ubuntu_related
    
    def process_dialogue_corpus(self) -> int:
        """
        Process the Ubuntu Dialogue Corpus into QA pairs.
        
        Returns:
            int: Number of QA pairs processed
        """
        start_time = datetime.now()
        
        # Check if raw file exists
        if not self.raw_file.exists():
            success = self.download_corpus()
            if not success:
                logger.error(f"Raw data file not found: {self.raw_file}")
                return self._create_sample_data()
        
        try:
            # Try to read the CSV file with different parameters
            try:
                df = pd.read_csv(self.raw_file)
            except Exception as e:
                logger.warning(f"Error reading CSV with default settings: {e}")
                try:
                    # Try with tab delimiter
                    df = pd.read_csv(self.raw_file, sep='\t')
                except Exception as e2:
                    logger.error(f"Failed to read CSV file: {e2}")
                    return self._create_sample_data()
            
            # Check required columns
            required_columns = ['DialogID', 'EpisodeID', 'Utterance', 'From', 'To']
            alt_columns = ['Dialog ID', 'Episode ID', 'Utterance', 'From', 'To']
            
            # Check if columns exist or need renaming
            if all(col in df.columns for col in required_columns):
                pass  # Columns are correctly named
            elif all(col in df.columns for col in alt_columns):
                # Rename columns to standard format
                df = df.rename(columns={
                    'Dialog ID': 'DialogID',
                    'Episode ID': 'EpisodeID',
                    'Utterance': 'Utterance',
                    'From': 'From',
                    'To': 'To'
                })
            else:
                logger.error(f"CSV does not have required columns. Found: {df.columns.tolist()}")
                return self._create_sample_data()
            
            logger.info(f"Processing dialogue corpus with {len(df)} utterances")
            self.stats['raw_dialogs'] = len(df['DialogID'].unique())
            
            # Group by dialog ID and extract QA pairs
            qa_pairs = []
            
            # Track dialogs processed for max_samples limit
            dialogs_processed = 0
            
            for dialog_id, dialog_df in tqdm(df.groupby('DialogID'), desc="Processing dialogs"):
                # Sort by EpisodeID to get the right order
                dialog_df = dialog_df.sort_values('EpisodeID')
                
                # Process each turn in the conversation
                for i in range(len(dialog_df) - 1):
                    # Get current and next utterance
                    current = dialog_df.iloc[i]
                    next_utterance = dialog_df.iloc[i+1]
                    
                    # Clean the text
                    question = self.clean_text(current['Utterance'])
                    answer = self.clean_text(next_utterance['Utterance'])
                    
                    # Check if this is a valid QA pair
                    if self.is_valid_qa_pair(question, answer):
                        pair_id = f"{dialog_id}_{i}"
                        qa_pairs.append({
                            "id": pair_id,
                            "content": question,
                            "response": answer,
                            "source": "Ubuntu Dialogue Corpus",
                            "metadata": {
                                "dialog_id": str(dialog_id),
                                "turn": i,
                                "from_user": str(current['From']),
                                "to_user": str(current['To'])
                            }
                        })
                
                # Increment dialogs processed
                dialogs_processed += 1
                
                # Check if we've hit the max_samples limit
                if self.max_samples and dialogs_processed >= self.max_samples:
                    break
            
            # Save the processed data
            if qa_pairs:
                with open(self.processed_file, 'w') as f:
                    json.dump(qa_pairs, f, indent=2)
                
                self.stats['processed_qa_pairs'] = len(qa_pairs)
                logger.info(f"Saved {len(qa_pairs)} QA pairs to {self.processed_file}")
            else:
                logger.warning("No valid QA pairs extracted, creating sample data")
                return self._create_sample_data()
            
            # Update metadata
            self.stats['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
            self.stats['last_processed'] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            return len(qa_pairs)
            
        except Exception as e:
            logger.error(f"Error processing corpus: {e}", exc_info=True)
            return self._create_sample_data()
    
    def chunk_documents(self) -> int:
        """
        Chunk the processed QA pairs for better retrieval.
        
        Returns:
            int: Number of chunks created
        """
        # Check if processed file exists
        if not self.processed_file.exists():
            count = self.process_dialogue_corpus()
            if count == 0:
                logger.error("No processed data available for chunking")
                return 0
        
        try:
            # Load the processed QA pairs
            with open(self.processed_file, 'r') as f:
                documents = json.load(f)
            
            logger.info(f"Chunking {len(documents)} documents")
            
            chunked_documents = []
            
            for doc in tqdm(documents, desc="Chunking documents"):
                # Process question
                question = doc['content']
                answer = doc['response']
                doc_id = doc['id']
                
                # If content is short, keep as is
                if len(question) <= self.chunk_size and len(answer) <= self.chunk_size:
                    chunked_documents.append(doc)
                    continue
                
                # Chunk long answers
                if len(answer) > self.chunk_size:
                    # Split by paragraphs first if possible
                    paragraphs = answer.split('\n\n')
                    
                    if len(paragraphs) > 1:
                        # Process paragraph chunks
                        current_chunk = ""
                        chunk_index = 0
                        
                        for para in paragraphs:
                            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                                # Save current chunk
                                if current_chunk:
                                    chunk_doc = doc.copy()
                                    chunk_doc['response'] = current_chunk.strip()
                                    chunk_doc['id'] = f"{doc_id}_chunk_{chunk_index}"
                                    chunk_doc['is_chunk'] = True
                                    chunk_doc['parent_id'] = doc_id
                                    chunked_documents.append(chunk_doc)
                                    chunk_index += 1
                                    
                                    # Start new chunk with overlap from end of previous
                                    overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                                    current_chunk = current_chunk[overlap_point:] + "\n\n"
                                
                                # Add current paragraph
                                current_chunk += para + "\n\n"
                            else:
                                # Add paragraph to current chunk
                                current_chunk += para + "\n\n"
                        
                        # Add final chunk if not empty
                        if current_chunk.strip():
                            chunk_doc = doc.copy()
                            chunk_doc['response'] = current_chunk.strip()
                            chunk_doc['id'] = f"{doc_id}_chunk_{chunk_index}"
                            chunk_doc['is_chunk'] = True
                            chunk_doc['parent_id'] = doc_id
                            chunked_documents.append(chunk_doc)
                    else:
                        # Character-based chunking as fallback
                        for i in range(0, len(answer), self.chunk_size - self.chunk_overlap):
                            chunk_text = answer[i:i + self.chunk_size]
                            if chunk_text.strip():
                                chunk_doc = doc.copy()
                                chunk_doc['response'] = chunk_text
                                chunk_doc['id'] = f"{doc_id}_chunk_{i // (self.chunk_size - self.chunk_overlap)}"
                                chunk_doc['is_chunk'] = True
                                chunk_doc['parent_id'] = doc_id
                                chunked_documents.append(chunk_doc)
                else:
                    # Just add the original document
                    chunked_documents.append(doc)
            
            # Save the chunked documents
            with open(self.chunked_file, 'w') as f:
                json.dump(chunked_documents, f, indent=2)
            
            self.stats['chunks'] = len(chunked_documents)
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
            
            # Update metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            return len(chunked_documents)
        
        except Exception as e:
            logger.error(f"Error chunking documents: {e}", exc_info=True)
            return 0
    
    def _create_sample_data(self) -> int:
        """
        Create sample data as fallback when real data cannot be processed.
        
        Returns:
            int: Number of sample documents created
        """
        logger.info("Creating sample data")
        
        sample_data = [
            {
                "id": "1",
                "content": "How do I update my system to the latest Ubuntu version?",
                "response": "To update your Ubuntu system to the latest version, you can use the following commands in terminal:\n\n```\nsudo apt update\nsudo apt upgrade\nsudo do-release-upgrade\n```\n\nThe first command refreshes your package lists, the second updates installed packages, and the third initiates the release upgrade process.",
                "source": "Ubuntu Dialogue Corpus"
            },
            {
                "id": "2",
                "content": "My printer isn't working with Ubuntu 22.04",
                "response": "To troubleshoot printer issues on Ubuntu 22.04:\n\n1. Check if the printer is properly connected and powered on\n2. Open System Settings > Printers to see if your printer is listed\n3. If not, click 'Add' to install a new printer\n4. You may need to install drivers using:\n   ```\n   sudo apt install cups printer-driver-all\n   sudo systemctl restart cups\n   ```\n5. For specific printer models, you might need to download drivers from the manufacturer's website",
                "source": "Ubuntu Dialogue Corpus"
            },
            {
                "id": "3",
                "content": "How do I install software from a PPA?",
                "response": "To install software from a PPA (Personal Package Archive) on Ubuntu:\n\n1. Add the PPA using:\n   ```\n   sudo add-apt-repository ppa:repository-name/ppa\n   ```\n\n2. Update package lists:\n   ```\n   sudo apt update\n   ```\n\n3. Install the software:\n   ```\n   sudo apt install package-name\n   ```\n\nReplace 'repository-name/ppa' and 'package-name' with the specific PPA and package you want to install.",
                "source": "Ubuntu Dialogue Corpus"
            },
            {
                "id": "4",
                "content": "My Ubuntu system is running slow after recent updates",
                "response": "If your Ubuntu system is running slow after updates, try these troubleshooting steps:\n\n1. Check system resources: Open System Monitor (gnome-system-monitor) to see which processes are consuming resources\n\n2. Clear package cache: `sudo apt clean`\n\n3. Remove old kernels: `sudo apt autoremove`\n\n4. Check startup applications: Open 'Startup Applications' and disable unnecessary programs\n\n5. Consider lighter desktop environments if you're on older hardware: `sudo apt install xubuntu-desktop` or `sudo apt install lubuntu-desktop`\n\n6. If the issue persists, try booting with an older kernel from the GRUB menu at startup.",
                "source": "Ubuntu Dialogue Corpus"
            },
            {
                "id": "5",
                "content": "How do I setup dual monitors on Ubuntu?",
                "response": "To set up dual monitors on Ubuntu:\n\n1. Connect your second monitor to your computer\n\n2. Go to Settings > Displays (or type 'Displays' in the Activities search)\n\n3. You should see both monitors represented in the configuration screen\n\n4. Arrange the monitors by dragging them to match your physical setup\n\n5. Choose whether to mirror displays or extend them (typically you want 'extend')\n\n6. Configure resolution, refresh rate, and scaling as needed for each display\n\n7. Click 'Apply' to save your changes\n\nIf your second monitor isn't detected, try:\n- Different connection ports/cables\n- Installing proprietary drivers for your graphics card: System Settings > Additional Drivers",
                "source": "Ubuntu Dialogue Corpus"
            }
        ]
        
        # Add more samples for a better test dataset
        sample_data.extend([
            {
                "id": "6",
                "content": "How to install Google Chrome on Ubuntu 22.04?",
                "response": "To install Google Chrome on Ubuntu 22.04:\n\n1. Download the Chrome .deb package from the official website:\n   ```\n   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb\n   ```\n\n2. Install the package using dpkg:\n   ```\n   sudo dpkg -i google-chrome-stable_current_amd64.deb\n   ```\n\n3. If there are any dependency issues, run:\n   ```\n   sudo apt install -f\n   ```\n\n4. You can now launch Chrome from your applications menu or by running `google-chrome` in the terminal.",
                "source": "Ubuntu Dialogue Corpus"
            },
            {
                "id": "7",
                "content": "How to fix 'Unable to locate package' error in Ubuntu?",
                "response": "When you encounter the 'Unable to locate package' error in Ubuntu, try these solutions:\n\n1. Update your package lists:\n   ```\n   sudo apt update\n   ```\n\n2. Make sure the Universe and Multiverse repositories are enabled:\n   ```\n   sudo add-apt-repository universe\n   sudo add-apt-repository multiverse\n   sudo apt update\n   ```\n\n3. Check if you've typed the package name correctly\n\n4. The package might be available under a different name; use apt search to find it:\n   ```\n   apt search keyword\n   ```\n\n5. If you're looking for a specific software that's not in the repositories, you may need to add a PPA or download it from the developer's website.",
                "source": "Ubuntu Dialogue Corpus"
            }
        ])
        
        # Save both processed and chunked copies
        with open(self.processed_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        with open(self.chunked_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.stats['processed_qa_pairs'] = len(sample_data)
        self.stats['chunks'] = len(sample_data)
        self.stats['last_processed'] = datetime.now().isoformat()
        
        # Update metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Created {len(sample_data)} sample documents")
        return len(sample_data)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Returns:
            Dict: Statistics about the processing
        """
        logger.info("Starting data processing pipeline")
        
        # Step 1: Download corpus if needed
        self.download_corpus()
        
        # Step 2: Process dialogue corpus
        qa_count = self.process_dialogue_corpus()
        logger.info(f"Processed {qa_count} QA pairs")
        
        # Step 3: Chunk documents
        chunk_count = self.chunk_documents()
        logger.info(f"Created {chunk_count} chunks")
        
        return self.stats


if __name__ == "__main__":
    # Configure command line options if needed
    import argparse
    
    parser = argparse.ArgumentParser(description="Process the Ubuntu Dialogue Corpus")
    parser.add_argument("--raw-dir", help="Directory for raw data", default="/data/raw")
    parser.add_argument("--processed-dir", help="Directory for processed data", default="/data/processed")
    parser.add_argument("--index-dir", help="Directory for index data", default="/data/index")
    parser.add_argument("--max-samples", help="Maximum samples to process", type=int)
    parser.add_argument("--force-download", help="Force download even if data exists", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize processor with command line options
    processor = UbuntuCorpusProcessor(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        index_data_dir=args.index_dir,
        max_samples=args.max_samples
    )
    
    # Run the pipeline and print results
    stats = processor.run_pipeline()
    print("\nPipeline completed with the following statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")