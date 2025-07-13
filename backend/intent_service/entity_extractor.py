import re
import json
from typing import List, Dict, Any, Optional, Tuple

class UbuntuEntityExtractor:
    """
    Specialized entity extractor for Ubuntu technical conversations
    Identifies software names, versions, commands, file paths, and technical concepts
    """
    
    def __init__(self, use_spacy=False):
        self.use_spacy = use_spacy
        # Note: Keeping spacy optional for deployment flexibility
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except ImportError:
                print("Warning: spaCy not available. Using pattern matching only.")
                self.use_spacy = False
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Using pattern matching only.")
                self.use_spacy = False
        
        # Ubuntu-specific entity patterns with confidence scores
        self.patterns = {
            "ubuntu_version": {
                "pattern": r"(ubuntu|xubuntu|lubuntu|kubuntu|ubuntu\s+server)\s+(\d+\.\d+(?:\.\d+)?(?:\s+LTS)?)",
                "confidence": 0.95
            },
            "package_name": {
                "pattern": r"(?:package|install|remove|purge|apt\s+(?:install|remove))\s+([a-z0-9][a-z0-9\-_.+]*[a-z0-9])",
                "confidence": 0.85
            },
            "command": {
                "pattern": r"(?:command|run|execute|type)\s+(?:['\"`]?)([a-z0-9\s\-_./]+)(?:['\"`]?)",
                "confidence": 0.80
            },
            "file_path": {
                "pattern": r"((?:\/[a-zA-Z0-9_.-]+)+(?:\/[a-zA-Z0-9_.-]*)?)",
                "confidence": 0.90
            },
            "error_code": {
                "pattern": r"error(?:\s+code)?:?\s*([A-Z]?[0-9]+|[A-Z]+[-_]?[0-9]*)",
                "confidence": 0.95
            },
            "ppa": {
                "pattern": r"ppa:([a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+)",
                "confidence": 0.95
            },
            "service_name": {
                "pattern": r"(?:service|daemon|systemctl)\s+([a-z0-9][a-z0-9\-_.]*[a-z0-9])",
                "confidence": 0.85
            },
            "port_number": {
                "pattern": r"port\s+(\d{1,5})",
                "confidence": 0.90
            },
            "ip_address": {
                "pattern": r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                "confidence": 0.95
            }
        }
        
        # Technical software names common in Ubuntu (with categories)
        self.software_database = {
            "package_managers": ["apt", "apt-get", "dpkg", "snap", "flatpak", "pip", "npm"],
            "desktop_environments": ["gnome", "kde", "xfce", "lxde", "mate", "cinnamon", "unity"],
            "web_browsers": ["firefox", "chrome", "chromium", "opera", "brave"],
            "media_players": ["vlc", "totem", "rhythmbox", "audacity", "obs"],
            "office_suites": ["libreoffice", "openoffice", "onlyoffice"],
            "development": ["vscode", "vim", "nano", "emacs", "git", "docker", "nodejs", "python"],
            "system_tools": ["systemd", "bash", "zsh", "fish", "tmux", "screen"],
            "servers": ["nginx", "apache", "mysql", "postgresql", "mongodb", "redis"],
            "virtualization": ["virtualbox", "vmware", "qemu", "kvm", "wine"],
            "graphics": ["gimp", "inkscape", "blender", "krita"],
            "network": ["openssh", "scp", "rsync", "wget", "curl", "netcat"]
        }
        
        # Flatten software list for quick lookup
        self.software_list = []
        for category, software in self.software_database.items():
            self.software_list.extend(software)
        
        # Common Ubuntu-specific terms and concepts
        self.ubuntu_concepts = [
            "repository", "ppa", "dependencies", "kernel", "driver", "bootloader", "grub",
            "mount", "partition", "filesystem", "ext4", "swap", "home directory", "root",
            "sudo", "permissions", "chmod", "chown", "desktop", "terminal", "shell",
            "package", "deb", "source", "compile", "build", "configure", "make"
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract Ubuntu-specific entities from text
        Returns a dict of entity types with values and confidence scores
        """
        entities = {
            "software": [],
            "version": [],
            "command": [],
            "file_path": [],
            "error_code": [],
            "ppa": [],
            "service": [],
            "network": [],
            "ubuntu_concept": [],
            "general": []
        }
        
        # Use spaCy for general NER if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities from spaCy
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"]:
                    entities["software"].append({
                        "value": ent.text,
                        "type": "software",
                        "confidence": 0.7,
                        "source": "spacy"
                    })
                elif ent.label_ == "VERSION" or (ent.label_ == "CARDINAL" and "." in ent.text):
                    entities["version"].append({
                        "value": ent.text,
                        "type": "version",
                        "confidence": 0.6,
                        "source": "spacy"
                    })
                else:
                    entities["general"].append({
                        "value": ent.text,
                        "type": ent.label_.lower(),
                        "confidence": 0.5,
                        "source": "spacy"
                    })
        
        # Pattern matching for Ubuntu-specific entities
        text_lower = text.lower()
        
        # Ubuntu version extraction
        version_matches = re.finditer(self.patterns["ubuntu_version"]["pattern"], text_lower, re.IGNORECASE)
        for match in version_matches:
            entities["version"].append({
                "value": f"{match.group(1)} {match.group(2)}",
                "type": "ubuntu_version", 
                "confidence": self.patterns["ubuntu_version"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # Package names
        package_matches = re.finditer(self.patterns["package_name"]["pattern"], text_lower)
        for match in package_matches:
            package_name = match.group(1)
            if len(package_name) > 1 and package_name.isalnum() or '-' in package_name or '_' in package_name:
                entities["software"].append({
                    "value": package_name,
                    "type": "package",
                    "confidence": self.patterns["package_name"]["confidence"],
                    "source": "pattern",
                    "position": match.span()
                })
        
        # Commands
        command_matches = re.finditer(self.patterns["command"]["pattern"], text_lower)
        for match in command_matches:
            command = match.group(1).strip()
            if command and len(command) > 1:
                entities["command"].append({
                    "value": command,
                    "type": "command",
                    "confidence": self.patterns["command"]["confidence"],
                    "source": "pattern",
                    "position": match.span()
                })
        
        # File paths
        path_matches = re.finditer(self.patterns["file_path"]["pattern"], text)
        for match in path_matches:
            path = match.group(1)
            if len(path) > 3:  # Avoid very short matches
                entities["file_path"].append({
                    "value": path,
                    "type": "file_path",
                    "confidence": self.patterns["file_path"]["confidence"],
                    "source": "pattern",
                    "position": match.span()
                })
        
        # Error codes
        error_matches = re.finditer(self.patterns["error_code"]["pattern"], text_lower)
        for match in error_matches:
            error_code = match.group(1)
            entities["error_code"].append({
                "value": error_code,
                "type": "error_code",
                "confidence": self.patterns["error_code"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # PPA sources
        ppa_matches = re.finditer(self.patterns["ppa"]["pattern"], text_lower)
        for match in ppa_matches:
            ppa = match.group(1)
            entities["ppa"].append({
                "value": ppa,
                "type": "ppa",
                "confidence": self.patterns["ppa"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # Service names
        service_matches = re.finditer(self.patterns["service_name"]["pattern"], text_lower)
        for match in service_matches:
            service = match.group(1)
            entities["service"].append({
                "value": service,
                "type": "service",
                "confidence": self.patterns["service_name"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # Port numbers
        port_matches = re.finditer(self.patterns["port_number"]["pattern"], text_lower)
        for match in port_matches:
            port = match.group(1)
            entities["network"].append({
                "value": port,
                "type": "port",
                "confidence": self.patterns["port_number"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # IP addresses
        ip_matches = re.finditer(self.patterns["ip_address"]["pattern"], text)
        for match in ip_matches:
            ip = match.group(1)
            entities["network"].append({
                "value": ip,
                "type": "ip_address",
                "confidence": self.patterns["ip_address"]["confidence"],
                "source": "pattern",
                "position": match.span()
            })
        
        # Check for known software names
        for software in self.software_list:
            pattern = r'\b' + re.escape(software) + r'\b'
            if re.search(pattern, text_lower):
                # Find category for this software
                category = "unknown"
                for cat, soft_list in self.software_database.items():
                    if software in soft_list:
                        category = cat
                        break
                
                entities["software"].append({
                    "value": software,
                    "type": "software",
                    "confidence": 0.8,
                    "source": "database",
                    "category": category
                })
        
        # Check for Ubuntu concepts
        for concept in self.ubuntu_concepts:
            pattern = r'\b' + re.escape(concept) + r'\b'
            if re.search(pattern, text_lower):
                entities["ubuntu_concept"].append({
                    "value": concept,
                    "type": "ubuntu_concept",
                    "confidence": 0.7,
                    "source": "concept_database"
                })
        
        # Deduplicate entities within each category
        for category in entities:
            entities[category] = self._deduplicate_entities(entities[category])
        
        return entities
    
    def extract_flat_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities as a flat list with confidence scores for simple context tracking
        """
        entity_dict = self.extract_entities(text)
        flat_entities = []
        
        # Prioritize by importance and confidence
        priority_order = ["error_code", "software", "version", "command", "service", "ppa", "file_path", "network", "ubuntu_concept"]
        
        for category in priority_order:
            if category in entity_dict:
                flat_entities.extend(entity_dict[category])
        
        # Sort by confidence and return top entities
        flat_entities.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Return top 10 entities to avoid overwhelming context
        return flat_entities[:10]
    
    def extract_for_intent_service(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities in format compatible with existing intent service
        """
        flat_entities = self.extract_flat_entities(text)
        
        # Convert to intent service format
        result = []
        for entity in flat_entities:
            result.append({
                "type": entity.get("type", "unknown"),
                "value": entity["value"],
                "confidence": entity.get("confidence", 0.5)
            })
        
        return result
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities, keeping the one with highest confidence
        """
        seen = {}
        result = []
        
        for entity in entities:
            value = entity["value"].lower()
            if value not in seen or entity.get("confidence", 0) > seen[value].get("confidence", 0):
                seen[value] = entity
        
        result = list(seen.values())
        
        # Sort by confidence
        result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return result
    
    def analyze_technical_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the technical complexity of the input text
        """
        entities = self.extract_entities(text)
        
        # Count different types of technical entities
        complexity_score = 0
        technical_indicators = {
            "commands": len(entities.get("command", [])),
            "file_paths": len(entities.get("file_path", [])),
            "error_codes": len(entities.get("error_code", [])),
            "software": len(entities.get("software", [])),
            "services": len(entities.get("service", [])),
            "network_elements": len(entities.get("network", []))
        }
        
        # Calculate complexity score
        complexity_score += technical_indicators["commands"] * 2
        complexity_score += technical_indicators["file_paths"] * 1.5
        complexity_score += technical_indicators["error_codes"] * 3
        complexity_score += technical_indicators["software"] * 1
        complexity_score += technical_indicators["services"] * 2
        complexity_score += technical_indicators["network_elements"] * 2
        
        # Determine complexity level
        if complexity_score >= 10:
            complexity_level = "high"
        elif complexity_score >= 5:
            complexity_level = "medium"
        else:
            complexity_level = "low"
        
        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "technical_indicators": technical_indicators,
            "total_entities": sum(len(entities[cat]) for cat in entities)
        }


# Example usage and test function
def test_entity_extractor():
    """Test function to verify entity extraction capabilities"""
    extractor = UbuntuEntityExtractor()
    
    test_cases = [
        "How do I install nginx on Ubuntu 22.04 LTS?",
        "I'm getting error code E: Unable to lock /var/lib/dpkg/lock",
        "Run sudo apt update && sudo apt upgrade to update your system",
        "Add the ppa:deadsnakes/ppa repository for Python 3.11",
        "The nginx service is not starting on port 80",
        "Check the log file at /var/log/nginx/error.log",
        "My IP address is 192.168.1.100 and I can't connect to SSH"
    ]
    
    print("🧪 Testing Ubuntu Entity Extractor")
    print("=" * 50)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")
        print("-" * 30)
        
        entities = extractor.extract_entities(test_text)
        for category, entity_list in entities.items():
            if entity_list:
                print(f"{category.upper()}:")
                for entity in entity_list:
                    print(f"  - {entity['value']} (confidence: {entity['confidence']:.2f})")
        
        # Test complexity analysis
        complexity = extractor.analyze_technical_complexity(test_text)
        print(f"Complexity: {complexity['complexity_level']} (score: {complexity['complexity_score']})")


if __name__ == "__main__":
    test_entity_extractor()
