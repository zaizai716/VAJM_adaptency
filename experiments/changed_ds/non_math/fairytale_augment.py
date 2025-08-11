#!/usr/bin/env python3
"""
FairytaleQA Dataset Augmentation for Adaptation Latency Experiments

Uses Qwen2.5-7B to generate alternative questions from the same story content.
Creates coherent, semantically different questions about the same fairy tales.
"""

import json
import random
import re
import torch
import os
import shutil
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

class FairytaleQAugmenter:
    """Augments FairytaleQA dataset with LLM-generated alternative questions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.dataset = None
        self.stories_by_name = {}
        
        # Initialize LLM for question generation
        print(f"Loading LLM: {model_name}")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Use cached model - will find your existing cache
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            trust_remote_code=True
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Greedy generation config for consistency
        self.generation_config = GenerationConfig(
            do_sample=False,  # Greedy for reproducibility
            max_new_tokens=150,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        self.device = device
        
    def load_dataset(self):
        """Load FairytaleQA dataset from local file."""
        print("Loading FairytaleQA dataset...")
        
        # Try local file first
        local_file = Path("train.json")
        if local_file.exists():
            print(f"Loading from local file: {local_file}")
            data = []
            
            # Try JSONL format first (one JSON object per line)
            try:
                with open(local_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                print(f"Loaded {len(data)} entries from JSONL format")
            except:
                # Fallback to single JSON object
                with open(local_file, 'r') as f:
                    data = json.load(f)
                print(f"Loaded data as single JSON object")
                
                # Handle nested structure
                if isinstance(data, dict):
                    for key in ['data', 'train', 'instances', 'examples']:
                        if key in data:
                            data = data[key]
                            break
            
            # Trim to 500 questions if dataset is larger
            if len(data) > 500:
                print(f"Dataset has {len(data)} questions, trimming to 500...")
                random.shuffle(data)
                data = data[:500]
            
            self.dataset = data
            print(f"Total entries: {len(self.dataset)}")
        else:
            print(f"Local file not found: {local_file}")
            print("Please download train.json from:")
            print("https://huggingface.co/datasets/GEM/FairytaleQA/tree/main/data")
            print("Using test data instead...")
            self._create_test_data()
        
        # Group questions by story
        for item in self.dataset:
            story_name = item['story_name']
            if story_name not in self.stories_by_name:
                self.stories_by_name[story_name] = {
                    'content': item['content'],
                    'questions': []
                }
            self.stories_by_name[story_name]['questions'].append({
                'question': item['question'],
                'answer': item['answer'],
                'attribute': item.get('attribute', ''),
                'ex_or_im': item.get('ex_or_im', ''),
                'gem_id': item.get('gem_id', '')
            })
        
        print(f"Loaded {len(self.dataset)} questions from {len(self.stories_by_name)} stories")
    
    def _create_test_data(self):
        """Create minimal test data if dataset loading fails."""
        print("Creating minimal test data...")
        
        test_stories = [
            {
                'story_name': 'test_story_1',
                'content': 'Once upon a time, there was a brave little princess named Luna. She lived in a beautiful castle with her pet dragon, Spark. One day, Luna decided to explore the enchanted forest near her castle.',
                'question': 'What was the name of the princess?',
                'answer': 'Luna',
                'attribute': 'character',
                'ex_or_im': 'explicit',
                'gem_id': 'test_001'
            },
            {
                'story_name': 'test_story_1', 
                'content': 'Once upon a time, there was a brave little princess named Luna. She lived in a beautiful castle with her pet dragon, Spark. One day, Luna decided to explore the enchanted forest near her castle.',
                'question': 'Where did Luna live?',
                'answer': 'In a beautiful castle',
                'attribute': 'setting',
                'ex_or_im': 'explicit',
                'gem_id': 'test_002'
            },
            {
                'story_name': 'test_story_2',
                'content': 'In a small village, there lived a clever boy named Tom. Tom loved to solve puzzles and riddles. His grandmother gave him a magical book that could answer any question.',
                'question': 'Who gave Tom the magical book?',
                'answer': 'His grandmother',
                'attribute': 'character',
                'ex_or_im': 'explicit', 
                'gem_id': 'test_003'
            }
        ]
        
        # Convert to dataset-like format
        self.dataset = test_stories
    
    def generate_alternative_question(self, story: str, original_question: str, original_answer: str) -> Dict[str, str]:
        """Use LLM to generate an alternative question about the same story."""
        
        prompt = f"""Read this fairy tale story and the original question. Generate a DIFFERENT question about the same story that asks about a different aspect or detail.

Story: {story}

Original Question: {original_question}
Original Answer: {original_answer}

Instructions:
- Create a NEW question that is semantically different from the original
- The question should be about the same story but focus on different characters, events, or details
- Make it a valid comprehension question that can be answered from the story
- Provide the answer based on the story content

Format your response as:
Alternative Question: [your new question]
Alternative Answer: [the answer to your new question]"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse response
        alt_question, alt_answer = self._parse_llm_response(response)
        
        return {
            'alt_question': alt_question,
            'alt_answer': alt_answer,
            'raw_response': response
        }
    
    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract alternative question and answer."""
        
        # Try to extract structured response
        alt_question = ""
        alt_answer = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Alternative Question:"):
                alt_question = line.replace("Alternative Question:", "").strip()
            elif line.startswith("Alternative Answer:"):
                alt_answer = line.replace("Alternative Answer:", "").strip()
        
        # Fallback: try regex patterns
        if not alt_question:
            question_match = re.search(r'(?:Alternative Question:|Question:)\s*(.+?)(?:\n|Alternative Answer:|$)', response, re.IGNORECASE | re.DOTALL)
            if question_match:
                alt_question = question_match.group(1).strip()
        
        if not alt_answer:
            answer_match = re.search(r'(?:Alternative Answer:|Answer:)\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            if answer_match:
                alt_answer = answer_match.group(1).strip()
        
        # Final fallback - use first sentence as question if nothing found
        if not alt_question and response:
            sentences = response.split('.')
            alt_question = sentences[0].strip() + "?"
            alt_answer = "Unable to extract answer from response"
        
        return alt_question, alt_answer

    def create_alternative_pairs(self, num_samples: int = 500) -> List[Dict[str, Any]]:
        """Create alternative question pairs ensuring story diversity."""
        
        augmented_data = []
        
        # First, ensure we get diverse stories
        print(f"\n📚 Analyzing story diversity...")
        story_names = list(self.stories_by_name.keys())
        print(f"Total unique stories: {len(story_names)}")
        
        # Show sample of story titles to verify diversity
        print("\n📖 Sample story titles (showing variety):")
        sample_names = random.sample(story_names, min(20, len(story_names)))
        for name in sample_names:
            print(f"  - {name}")
        
        # Prioritize diverse story selection
        selected_stories = []
        
        # Look for stories with specific keywords to ensure variety
        keywords = ['troll', 'pancake', 'wedding', 'princess', 'dragon', 'magic', 
                    'forest', 'castle', 'witch', 'fairy', 'king', 'queen', 'wolf', 
                    'bear', 'gold', 'giant', 'three', 'little']
        
        print("\n🎯 Selecting diverse stories...")
        for keyword in keywords:
            matching = [s for s in story_names if keyword.lower() in s.lower()]
            if matching:
                selected_stories.extend(matching[:2])  # Take up to 2 per keyword
                print(f"  Found {len(matching)} stories with '{keyword}'")
        
        # Add random stories to fill out
        remaining = [s for s in story_names if s not in selected_stories]
        random.shuffle(remaining)
        selected_stories.extend(remaining)
        
        # Remove duplicates
        selected_stories = list(dict.fromkeys(selected_stories))
        print(f"\n✨ Selected {len(selected_stories)} diverse stories")
        
        # Sample questions from diverse stories
        all_questions = []
        for story_name in selected_stories:
            story_data = self.stories_by_name[story_name]
            for q in story_data['questions']:
                all_questions.append({
                    'story_name': story_name,
                    'story_content': story_data['content'],
                    'question_data': q
                })
        
        random.shuffle(all_questions)
        
        # Generate variations (80%)
        num_variations = int(num_samples * 0.8)
        print(f"Generating {num_variations} LLM-created variations...")
        
        for i in tqdm(range(min(num_variations, len(all_questions))), desc="Generating alternatives"):
            item = all_questions[i]
            
            try:
                # Generate alternative using LLM
                result = self.generate_alternative_question(
                    item['story_content'],
                    item['question_data']['question'],
                    item['question_data']['answer']
                )
                
                # Classify variation type based on content
                variation_type = self._classify_llm_variation(
                    item['question_data']['question'],
                    result['alt_question']
                )
                
                entry = {
                    "story_name": item['story_name'],
                    "story_content": item['story_content'], 
                    "Question": item['question_data']['question'],
                    "Answer": item['question_data']['answer'],
                    "alt_problem": result['alt_question'],
                    "alt_answer": result['alt_answer'],
                    
                    # Metadata
                    "original_attribute": item['question_data'].get('attribute', ''),
                    "original_type": item['question_data'].get('ex_or_im', ''),
                    "variation_type": variation_type,
                    "generation_method": "llm_generated",
                    
                    # Debug info
                    "llm_raw_response": result['raw_response'],
                    "original_gem_id": item['question_data'].get('gem_id', ''),
                    "experiment_id": f"fairytale_var_{i:04d}"
                }
                
                augmented_data.append(entry)
                
            except Exception as e:
                print(f"Error generating alternative for item {i}: {e}")
                continue
        
        # Add controls (20%) - same question twice
        num_controls = min(num_samples - len(augmented_data), len(all_questions) - num_variations)
        print(f"Adding {num_controls} control pairs...")
        
        for i in range(num_controls):
            idx = num_variations + i
            if idx >= len(all_questions):
                break
            
            item = all_questions[idx]
            
            entry = {
                "story_name": item['story_name'],
                "story_content": item['story_content'],
                "Question": item['question_data']['question'],
                "Answer": item['question_data']['answer'],
                "alt_problem": item['question_data']['question'],  # SAME
                "alt_answer": item['question_data']['answer'],     # SAME
                
                "original_attribute": item['question_data'].get('attribute', ''),
                "original_type": item['question_data'].get('ex_or_im', ''),
                "variation_type": None,  # True control
                "generation_method": "control_identical",
                
                "original_gem_id": item['question_data'].get('gem_id', ''),
                "experiment_id": f"fairytale_ctrl_{i:04d}"
            }
            
            augmented_data.append(entry)
        
        # Shuffle final dataset
        random.shuffle(augmented_data)
        
        print(f"Created {len(augmented_data)} augmented question pairs")
        return augmented_data
    
    def _classify_llm_variation(self, original_question: str, alt_question: str) -> str:
        """Classify the type of variation between LLM-generated questions."""
        
        # Simple keyword-based classification
        original_lower = original_question.lower()
        alt_lower = alt_question.lower()
        
        # Character-focused variations
        character_words = ['who', 'character', 'person', 'boy', 'girl', 'man', 'woman', 'king', 'queen']
        if any(word in original_lower for word in character_words) and any(word in alt_lower for word in character_words):
            return "character_to_character"
        elif any(word in original_lower for word in character_words):
            return "character_to_other"
        elif any(word in alt_lower for word in character_words):
            return "other_to_character"
        
        # Action/event variations
        action_words = ['what', 'how', 'did', 'does', 'action', 'happen', 'do']
        if any(word in original_lower for word in action_words) and any(word in alt_lower for word in action_words):
            return "action_to_action"
        elif any(word in original_lower for word in action_words):
            return "action_to_other"
        elif any(word in alt_lower for word in action_words):
            return "other_to_action"
        
        # Location/setting variations  
        location_words = ['where', 'place', 'location', 'setting']
        if any(word in original_lower for word in location_words) or any(word in alt_lower for word in location_words):
            return "location_focused"
        
        # Causal/reason variations
        causal_words = ['why', 'because', 'reason', 'cause']
        if any(word in original_lower for word in causal_words) or any(word in alt_lower for word in causal_words):
            return "causal_reasoning"
        
        # Temporal variations
        time_words = ['when', 'time', 'first', 'last', 'before', 'after']
        if any(word in original_lower for word in time_words) or any(word in alt_lower for word in time_words):
            return "temporal_focused"
        
        # Default semantic variation
        return "semantic_variation"
    
    def analyze_dataset_stats(self, augmented_data: List[Dict]) -> Dict[str, Any]:
        """Analyze statistics of the augmented dataset."""
        
        stats = {
            "total_pairs": len(augmented_data),
            "unique_stories": len(set(item['story_name'] for item in augmented_data)),
            "controls": sum(1 for item in augmented_data if item['variation_type'] is None),
            "variations": len(augmented_data) - sum(1 for item in augmented_data if item['variation_type'] is None),
        }
        
        # Variation type distribution
        variation_counts = {}
        for item in augmented_data:
            var_type = item['variation_type'] or 'control'
            variation_counts[var_type] = variation_counts.get(var_type, 0) + 1
        
        stats["variation_distribution"] = variation_counts
        
        # Attribute analysis (handle missing keys)
        original_attrs = [item.get('original_attribute', 'unknown') for item in augmented_data]
        alt_attrs = [item.get('alt_attribute', 'unknown') for item in augmented_data]
        
        stats["original_attributes"] = {attr: original_attrs.count(attr) for attr in set(original_attrs) if attr}
        stats["alt_attributes"] = {attr: alt_attrs.count(attr) for attr in set(alt_attrs) if attr}
        
        # Story length statistics
        story_lengths = [len(item['story_content'].split()) for item in augmented_data]
        stats["story_length_stats"] = {
            "min": min(story_lengths),
            "max": max(story_lengths), 
            "mean": sum(story_lengths) / len(story_lengths),
            "median": sorted(story_lengths)[len(story_lengths)//2]
        }
        
        return stats

def main():
    """Main function to create augmented FairytaleQA dataset."""
    
    # Set random seed for reproducibility  
    random.seed(42)
    torch.manual_seed(42)
    
    print("🧚‍♀️ FairytaleQA Dataset Augmentation with LLM Generation")
    print("=" * 60)
    
    # Use existing cached Qwen model
    print("🔄 Loading Qwen2.5-7B-Instruct from existing cache...")
    augmenter = FairytaleQAugmenter(model_name="Qwen/Qwen2.5-7B-Instruct")
    
    try:
        
        # Load dataset
        augmenter.load_dataset()
        
        # Create augmented pairs using LLM  
        print("\n🤖 Generating alternative questions using LLM...")
        augmented_data = augmenter.create_alternative_pairs(num_samples=500)
        
        # Analyze dataset
        stats = augmenter.analyze_dataset_stats(augmented_data)
        
        # Save results
        output_dir = Path(__file__).parent

        # Save main dataset
        output_file = output_dir / "fairytale_qa_augmented.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = output_dir / "fairytale_qa_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 Dataset: {output_file}")
        print(f"📊 Stats: {stats_file}")
        print(f"\n📈 DATASET SUMMARY:")
        print(f"   Total pairs: {stats['total_pairs']}")
        print(f"   Unique stories: {stats['unique_stories']}")
        print(f"   Controls: {stats['controls']} ({stats['controls']/stats['total_pairs']*100:.1f}%)")
        print(f"   Variations: {stats['variations']} ({stats['variations']/stats['total_pairs']*100:.1f}%)")
        
        # Show story diversity
        print(f"\n🎭 Story diversity (first 15):")
        unique_stories = list(set(item['story_name'] for item in augmented_data))[:15]
        for story in unique_stories:
            print(f"   - {story}")
        
        print(f"\n📚 Story lengths (words):")
        print(f"   Min: {stats['story_length_stats']['min']}")
        print(f"   Max: {stats['story_length_stats']['max']}")
        print(f"   Mean: {stats['story_length_stats']['mean']:.1f}")
        
        print(f"\n🎯 Variation types:")
        for var_type, count in sorted(stats['variation_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:8]:
            print(f"   {var_type}: {count} ({count/stats['total_pairs']*100:.1f}%)")
        
        print(f"\n🔧 Generation methods:")
        methods = {}
        for item in augmented_data:
            method = item.get('generation_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        for method, count in methods.items():
            print(f"   {method}: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()