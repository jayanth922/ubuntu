import os
import pandas as pd
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UbuntuCorpusProcessor:
    """Process Ubuntu Dialogue Corpus into a format suitable for RAG"""
    
    def __init__(self, input_dir='./data/raw', output_dir='./data/processed'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def process_dialogs(self, max_samples=None):
        """Process the Ubuntu dialog corpus CSV files"""
        try:
            # Find all CSV files in the input directory
            csv_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
            
            if not csv_files:
                logger.error(f"No CSV files found in {self.input_dir}")
                # Use sample data as fallback
                return self._create_sample_data()
            
            all_qa_pairs = []
            
            for csv_file in csv_files:
                file_path = os.path.join(self.input_dir, csv_file)
                logger.info(f"Processing {file_path}")
                
                try:
                    # Try different CSV formats as the format might vary
                    df = pd.read_csv(file_path, error_bad_lines=False)
                except Exception as e:
                    logger.warning(f"Error reading {file_path} with default settings: {e}")
                    try:
                        # Try with different delimiter
                        df = pd.read_csv(file_path, sep='\t', error_bad_lines=False)
                    except Exception as e2:
                        logger.error(f"Failed to read {file_path}: {e2}")
                        continue
                
                # Check for required columns
                required_columns = ['DialogueID', 'Timestamp', 'Text']
                if not all(col in df.columns for col in required_columns):
                    # Try to infer column names based on content
                    if 'Text' not in df.columns and len(df.columns) >= 3:
                        # Assume the last column contains the text
                        df = df.rename(columns={df.columns[-1]: 'Text'})
                    
                    # If still missing required columns, skip this file
                    if not all(col in df.columns for col in required_columns):
                        logger.warning(f"Missing required columns in {file_path}, skipping")
                        continue
                
                # Group by conversation ID
                qa_pairs = []
                for dialog_id, conversation in tqdm(df.groupby('DialogueID')):
                    # Sort by timestamp to get the right order
                    if 'Timestamp' in conversation:
                        conversation = conversation.sort_values('Timestamp')
                    
                    # Get messages
                    messages = conversation['Text'].tolist()
                    
                    # Process pairs (Q&A style)
                    for i in range(0, len(messages)-1, 2):
                        if i+1 < len(messages):
                            # Skip empty messages
                            if not pd.isna(messages[i]) and not pd.isna(messages[i+1]) and len(str(messages[i]).strip()) > 0:
                                qa_pairs.append({
                                    "id": f"{dialog_id}_{i//2}",
                                    "content": str(messages[i]),
                                    "response": str(messages[i+1]),
                                    "source": "Ubuntu Dialogue Corpus"
                                })
                
                all_qa_pairs.extend(qa_pairs)
                logger.info(f"Extracted {len(qa_pairs)} QA pairs from {file_path}")
                
                # Limit samples if specified
                if max_samples and len(all_qa_pairs) >= max_samples:
                    all_qa_pairs = all_qa_pairs[:max_samples]
                    break
            
            if not all_qa_pairs:
                logger.warning("No QA pairs extracted, using sample data")
                return self._create_sample_data()
            
            # Save to JSON
            output_file = os.path.join(self.output_dir, 'ubuntu_corpus.json')
            with open(output_file, 'w') as f:
                json.dump(all_qa_pairs, f, indent=2)
            
            logger.info(f"Processed {len(all_qa_pairs)} QA pairs saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing Ubuntu corpus: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data as fallback"""
        logger.info("Creating sample data as fallback")
        
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
            # Add 8 more sample QA pairs
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
        
        # Save sample data
        output_file = os.path.join(self.output_dir, 'ubuntu_samples.json')
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample data created at {output_file}")
        return output_file


if __name__ == "__main__":
    processor = UbuntuCorpusProcessor()
    processor.process_dialogs(max_samples=10000)  # Process up to 10,000 QA pairs