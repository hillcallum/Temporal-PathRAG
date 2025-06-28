#!/usr/bin/env python3
"""
Converts raw_datasets/ to datasets/ with proper TimeR4-compatible structure
"""

import shutil
from pathlib import Path
import json
import argparse


class DatasetProcessor:
    """Process raw datasets into research-ready format"""
    
    def __init__(self, clean_existing=False):
        self.raw_path = Path("raw_datasets")
        self.output_path = Path("datasets") 
        self.clean_existing = clean_existing
        
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw datasets directory not found: {self.raw_path}")
        
        print(f"Dataset Processor")
        print(f"Input: {self.raw_path}")
        print(f"Output: {self.output_path}")
        
    def process_multitq(self):
        """Process MultiTQ dataset"""
        print("\nProcessing MultiTQ dataset")
        
        src = self.raw_path / "MultiTQ"
        dst = self.output_path / "MultiTQ"
        
        if dst.exists() and self.clean_existing:
            print("Removing existing MultiTQ output")
            shutil.rmtree(dst)
        
        print("Copying MultiTQ structure")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Add metadata for research tracking
        metadata = {
            "dataset_type": "MultiTQ",
            "format": "TimeR4_compatible", 
            "description": "MultiTQ dataset processed for temporal knowledge graph research",
            "structure": {
                "kg/": "Knowledge graph files",
                "questions/": "Question datasets for evaluation",
                "prompt/": "Prompt files (if available)"
            }
        }
        
        with open(dst / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"MultiTQ processing complete: {dst}")
        
    def process_timequestions(self):
        """Process TimeQuestions dataset"""
        print("\nProcessing TimeQuestions dataset")
        
        src = self.raw_path / "TimeQuestions"
        dst = self.output_path / "TimeQuestions"
        
        if dst.exists() and self.clean_existing:
            print("Removing existing TimeQuestions output")
            shutil.rmtree(dst)
        
        print("Copying TimeQuestions structure")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Add metadata for research tracking
        metadata = {
            "dataset_type": "TimeQuestions",
            "format": "TimeR4_compatible",
            "description": "TimeQuestions dataset processed for temporal knowledge graph research", 
            "structure": {
                "kg/": "Knowledge graph files with Wikidata mappings",
                "questions/": "Question datasets for evaluation",
                "prompt/": "Prompt files (if available)"
            }
        }
        
        with open(dst / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"TimeQuestions processing complete: {dst}")
        
    def validate_output(self):
        """Validate processed datasets"""
        print("\nValidating processed datasets")
        
        required_files = {
            "MultiTQ": [
                "kg/full.txt",
                "kg/entity2id.json", 
                "kg/relation2id.json",
                "questions/dev.json",
                "questions/test.json"
            ],
            "TimeQuestions": [
                "kg/full.txt",
                "kg/wd_id2entity_text.txt",
                "questions/dev.json", 
                "questions/test.json"
            ]
        }
        
        for dataset_name, files in required_files.items():
            dataset_path = self.output_path / dataset_name
            print(f"\nValidating {dataset_name}")
            
            if not dataset_path.exists():
                print(f"ERROR: {dataset_name} directory missing")
                continue
                
            missing_files = []
            for file_path in files:
                if not (dataset_path / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"WARNING: Missing files in {dataset_name}:")
                for missing in missing_files:
                    print(f" - {missing}")
            else:
                print(f"All required files present in {dataset_name}")
        
    def run(self):
        """Run the complete processing pipeline"""
        print("Starting dataset processing pipeline")
        
        # Create output directory
        self.output_path.mkdir(exist_ok=True)
        
        # Process datasets
        if (self.raw_path / "MultiTQ").exists():
            self.process_multitq()
        else:
            print("Warning: MultiTQ raw dataset not found")
            
        if (self.raw_path / "TimeQuestions").exists():
            self.process_timequestions()
        else:
            print("Warning: TimeQuestions raw dataset not found")
        
        # Validate results
        self.validate_output()
        
        print("\nDataset processing complete")
        print(f"Processed datasets available in: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process raw datasets')
    parser.add_argument('--clean', action='store_true', 
                       help='Remove existing processed datasets before processing')
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(clean_existing=args.clean)
    processor.run()


if __name__ == "__main__":
    main()