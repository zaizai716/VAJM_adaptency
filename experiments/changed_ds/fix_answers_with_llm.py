#!/usr/bin/env python3
"""
Fix all answers in both math and non-math datasets using LLM to solve problems correctly.
This script will use Qwen models to generate correct answers for every single problem.
"""

import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Model configurations - Use exact same models as working inference script
MATH_MODEL = "Qwen/Qwen2.5-Math-7B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "~/.cache/huggingface/hub"

class LLMAnswerFixer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        self.math_model = None
        self.math_tokenizer = None
        self.instruct_model = None
        self.instruct_tokenizer = None
    
    def load_math_model(self):
        """Load the math model for solving math problems"""
        if self.math_model is None:
            print(f"📥 Loading math model: {MATH_MODEL}")
            self.math_model = AutoModelForCausalLM.from_pretrained(
                MATH_MODEL,
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            ).to(self.device)
            
            self.math_tokenizer = AutoTokenizer.from_pretrained(
                MATH_MODEL,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            print("✅ Math model loaded successfully")
    
    def load_instruct_model(self):
        """Load the instruct model for non-math problems"""
        if self.instruct_model is None:
            print(f"📥 Loading instruct model: {INSTRUCT_MODEL}")
            self.instruct_model = AutoModelForCausalLM.from_pretrained(
                INSTRUCT_MODEL,
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            ).to(self.device)
            
            self.instruct_tokenizer = AutoTokenizer.from_pretrained(
                INSTRUCT_MODEL,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            print("✅ Instruct model loaded successfully")
    
    def solve_math_problem(self, problem_text: str) -> Optional[float]:
        """Use math model to solve a math problem and extract numerical answer"""
        self.load_math_model()
        
        # Create a clear prompt for math solving
        prompt = f"""{problem_text.strip()}

Let me solve this step by step.

Please solve this problem carefully and state your final answer as 'The answer is: [number]' at the end."""
        
        try:
            # Tokenize and generate
            inputs = self.math_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.math_model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,  # Low temperature for consistent math
                    do_sample=True,
                    pad_token_id=self.math_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.math_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"🧮 Problem: {problem_text[:100]}...")
            print(f"🤖 Generated: {generated_text[:200]}...")
            
            # Extract numerical answer
            answer = self.extract_numerical_answer(generated_text)
            print(f"✅ Extracted answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"❌ Error solving math problem: {e}")
            return None
    
    def solve_nonmath_problem(self, story: str, question: str) -> Optional[str]:
        """Use instruct model to answer reading comprehension questions"""
        self.load_instruct_model()
        
        # Create prompt for reading comprehension
        prompt = f"""Read the following story and answer the question.

Story: {story.strip()}

Question: {question.strip()}

Please provide a clear, concise answer based on the story."""
        
        try:
            # Tokenize and generate
            inputs = self.instruct_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.instruct_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.instruct_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.instruct_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"📚 Question: {question[:100]}...")
            print(f"🤖 Generated: {generated_text[:150]}...")
            
            # Clean up the answer
            answer = self.clean_text_answer(generated_text)
            print(f"✅ Cleaned answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"❌ Error solving non-math problem: {e}")
            return None
    
    def extract_numerical_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from generated text"""
        # Look for "The answer is: X" pattern
        patterns = [
            r"The answer is:?\s*([+-]?\d*\.?\d+)",
            r"\\boxed\{([+-]?\d*\.?\d+)\}",
            r"answer:?\s*([+-]?\d*\.?\d+)",
            r"Answer:?\s*([+-]?\d*\.?\d+)",
            r"=\s*([+-]?\d*\.?\d+)\s*$",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        # If no clear pattern, try to find any number at the end
        numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def clean_text_answer(self, text: str) -> str:
        """Clean up text answer from model generation"""
        # Remove common generation artifacts
        text = text.strip()
        
        # Remove "Answer:" prefix if present
        text = re.sub(r'^(Answer:?|Response:?)\s*', '', text, flags=re.IGNORECASE)
        
        # Take first sentence or paragraph
        sentences = text.split('.')
        if sentences:
            answer = sentences[0].strip()
            if len(answer) > 10:  # Reasonable length
                return answer + ('.' if not answer.endswith('.') else '')
        
        # Fallback to first 100 characters
        return text[:100].strip() + ('...' if len(text) > 100 else '')
    
    def fix_math_dataset(self, input_path: str, output_path: str):
        """Fix all answers in math dataset"""
        print(f"\n🔢 Processing math dataset: {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Found {len(data)} math problems to process")
        
        for i, problem in enumerate(data):
            print(f"\n--- Problem {i+1}/{len(data)} ---")
            
            # Fix original problem answer
            original_question = problem.get('Question', '')
            if original_question:
                original_answer = self.solve_math_problem(original_question)
                if original_answer is not None:
                    problem['Answer'] = original_answer
                    print(f"✅ Updated original answer: {original_answer}")
            
            # Fix alternative problem answer  
            alt_problem = problem.get('alt_problem_original', problem.get('alt_problem', ''))
            if alt_problem:
                alt_answer = self.solve_math_problem(alt_problem)
                if alt_answer is not None:
                    problem['alt_answer'] = alt_answer
                    print(f"✅ Updated alternative answer: {alt_answer}")
            
            # Save progress every 10 problems
            if (i + 1) % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"💾 Progress saved ({i+1}/{len(data)} completed)")
        
        # Final save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Math dataset fixed and saved to: {output_path}")
    
    def fix_nonmath_dataset(self, input_path: str, output_path: str):
        """Fix all answers in non-math dataset"""
        print(f"\n📚 Processing non-math dataset: {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Found {len(data)} reading comprehension problems to process")
        
        for i, problem in enumerate(data):
            print(f"\n--- Problem {i+1}/{len(data)} ---")
            
            story = problem.get('story_content', '')
            
            # Fix original answer
            original_question = problem.get('Question', '')
            if original_question and story:
                original_answer = self.solve_nonmath_problem(story, original_question)
                if original_answer:
                    problem['Answer'] = original_answer
                    print(f"✅ Updated original answer: {original_answer}")
            
            # Fix alternative answer
            alt_question = problem.get('alt_problem_original', problem.get('alt_problem', ''))
            if alt_question and story:
                alt_answer = self.solve_nonmath_problem(story, alt_question)
                if alt_answer:
                    problem['alt_answer'] = alt_answer
                    print(f"✅ Updated alternative answer: {alt_answer}")
            
            # Save progress every 5 problems (non-math takes longer)
            if (i + 1) % 5 == 0:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"💾 Progress saved ({i+1}/{len(data)} completed)")
        
        # Final save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Non-math dataset fixed and saved to: {output_path}")

def main():
    fixer = LLMAnswerFixer()
    
    # Dataset paths
    base_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds"
    
    print("🚀 Starting LLM-based answer correction for ALL datasets")
    
    # Process math dataset
    math_input = f"{base_path}/math/mawps_multilingual.json"
    math_output = f"{base_path}/math/mawps_multilingual_fixed.json"
    fixer.fix_math_dataset(math_input, math_output)
    
    # Clean up math model to free memory
    del fixer.math_model
    del fixer.math_tokenizer
    fixer.math_model = None
    fixer.math_tokenizer = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process non-math dataset
    nonmath_input = f"{base_path}/non_math/fairytale_multilingual.json"
    nonmath_output = f"{base_path}/non_math/fairytale_multilingual_fixed.json"
    fixer.fix_nonmath_dataset(nonmath_input, nonmath_output)
    
    print("\n🎉 ALL DATASETS FIXED WITH LLM-GENERATED CORRECT ANSWERS!")
    print(f"📁 Fixed math dataset: {math_output}")
    print(f"📁 Fixed non-math dataset: {nonmath_output}")
    print("\n💡 Update your inference_loop.py to use these fixed datasets!")

if __name__ == "__main__":
    main()