#!/usr/bin/env python3
"""
Fixed MAWPS Dataset Generator
- 20% true controls (identical)
- 80% with actual calculation changes
"""

import json
import random
from typing import Dict, List, Tuple
import re
from datasets import load_dataset
import sympy as sp

class DatasetImprover:
    def __init__(self):
        self.seed = 42
        random.seed(self.seed)
        
    def substitute_numbers_in_text(self, text: str, numbers: str) -> str:
        """Replace N_XX tokens with actual numbers."""
        try:
            num_values = [float(x) for x in numbers.split()]
            result_text = text
            for i, val in enumerate(num_values):
                token = f"N_{i:02d}"
                if val == int(val):
                    replacement = str(int(val))
                else:
                    replacement = str(val)
                result_text = result_text.replace(token, replacement)
            return result_text
        except:
            return text

    def substitute_numbers_in_equation(self, equation: str, numbers: str) -> str:
        """Replace N_XX tokens in equation with actual numbers."""
        try:
            num_values = [float(x) for x in numbers.split()]
            result_eq = equation
            for i, val in enumerate(num_values):
                token = f"N_{i:02d}"
                result_eq = result_eq.replace(token, str(val))
            return result_eq
        except:
            return equation
    
    def extract_numbers(self, text: str) -> List[Tuple[str, float]]:
        """Extract all numbers from text with their string representation"""
        matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text)
        return [(m.group(1), float(m.group(1))) for m in matches]
    
    def change_single_number(self, question: str, equation: str, answer: float) -> Tuple[str, str, float, str]:
        """Change ONE number and recalculate - most reliable method"""
        numbers = self.extract_numbers(question)
        if not numbers or not equation:
            return question, answer, "failed"
            
        # Try each number until we find one that changes the answer
        for num_str, num_val in numbers:
            # Skip very small numbers (likely indices)
            if num_val < 1:
                continue
                
            # Generate a different number
            if num_val <= 10:
                new_val = random.choice([n for n in range(1, 21) if n != int(num_val)])
            elif num_val <= 100:
                new_val = random.choice([n for n in range(10, 151) if abs(n - num_val) > 5])
            else:
                new_val = int(num_val * random.uniform(0.5, 1.5))
                
            # Replace in both question and equation
            new_question = question.replace(num_str, str(new_val), 1)
            new_equation = equation.replace(num_str, str(new_val), 1)
            
            # Calculate new answer
            try:
                new_answer = float(sp.N(sp.sympify(new_equation)))
                # Only accept if answer actually changed AND is non-negative (for physical objects)
                if abs(new_answer - answer) > 0.01 and new_answer >= 0:
                    return new_question, new_equation, new_answer, "number_change"
            except:
                continue
                
        return question, equation, answer, "failed"
    
    def change_multiple_numbers(self, question: str, equation: str, answer: float) -> Tuple[str, str, float, str]:
        """Change 2-3 numbers for bigger variation"""
        numbers = self.extract_numbers(question)
        if len(numbers) < 2 or not equation:
            return question, equation, answer, "failed"
            
        # Change 2-3 numbers
        num_to_change = min(3, random.randint(2, len(numbers)))
        numbers_to_change = random.sample(numbers, num_to_change)
        
        new_question = question
        new_equation = equation
        
        for num_str, num_val in numbers_to_change:
            if num_val < 1:
                continue
                
            if num_val <= 10:
                new_val = random.choice([n for n in range(1, 21) if n != int(num_val)])
            else:
                new_val = int(num_val * random.uniform(0.7, 1.3))
                
            new_question = new_question.replace(num_str, str(new_val), 1)
            new_equation = new_equation.replace(num_str, str(new_val), 1)
        
        try:
            new_answer = float(sp.N(sp.sympify(new_equation)))
            if abs(new_answer - answer) > 0.01 and new_answer >= 0:
                return new_question, new_equation, new_answer, "multi_number_change"
        except:
            pass
            
        return question, equation, answer, "failed"
    
    def add_extra_step(self, question: str, equation: str, answer: float) -> Tuple[str, str, float, str]:
        """Add an extra calculation step to the problem"""
        # Add phrases like "then add 5 more" or "minus 3"
        extra_operations = [
            (" Then add {} more .", " + {}"),
            (" Then subtract {} .", " - {}"),
            (" If {} more are added , what is the final total ?", " + {}"),
            (" After removing {} , how many remain ?", " - {}")
        ]
        
        op_text, op_eq = random.choice(extra_operations)
        extra_val = random.randint(1, 10)
        
        new_question = question.rstrip(' .?') + op_text.format(extra_val)
        new_equation = f"({equation}) {op_eq.format(extra_val)}"
        
        try:
            new_answer = float(sp.N(sp.sympify(new_equation)))
            if abs(new_answer - answer) > 0.01 and new_answer >= 0:
                return new_question, new_equation, new_answer, "added_step"
        except:
            pass
            
        return question, equation, answer, "failed"
    
    def reverse_direction(self, question: str, equation: str, answer: float) -> Tuple[str, str, float, str]:
        """Reverse the problem direction - make what was given unknown and what was unknown given"""
        if not equation:
            return question, equation, answer, "failed"
            
        # Extract numbers from question and equation
        numbers = self.extract_numbers(question)
        if len(numbers) < 2:
            return question, equation, answer, "failed"
            
        # Try to reverse addition problems: "A + B = ?" becomes "A + ? = C" where C is the original answer
        if re.search(r'\+', equation):
            # Original: "John has 5 toys, gets 3 more. How many total?" (5 + 3 = 8)
            # Reverse: "John wants 8 toys total, has 5 now. How many more does he need?" (8 - 5 = 3)
            
            # Find the structure
            nums = [n[1] for n in numbers]
            if len(nums) >= 2:
                first_num = nums[0]
                second_num = nums[1]
                new_total = answer  # The original answer becomes a given
                new_answer = second_num  # What was added becomes the unknown
                
                # Create reversed question
                new_question = re.sub(r'\b' + str(int(second_num)) + r'\b', 'X_PLACEHOLDER', question, count=1)
                new_question = re.sub(r'\b' + str(int(new_total)) + r'\b', str(int(new_total)), new_question)
                new_question = re.sub(r'X_PLACEHOLDER', '?', new_question)
                
                # Update question phrasing
                new_question = re.sub(r'How many .* total\?', f'If the total is {int(new_total)}, how many more are needed?', new_question, flags=re.IGNORECASE)
                new_question = re.sub(r'gets (\d+) more', f'wants {int(new_total)} total', new_question)
                new_question = re.sub(r'finds? (\d+)', f'needs {int(new_total)} total', new_question)
                
                new_equation = f"{first_num} + X = {new_total}"  # X will be solved as new_total - first_num
                
                if abs(new_answer - answer) > 0.01 and new_question != question and new_answer >= 0:
                    return new_question, new_equation, new_answer, "reversed_direction"
        
        # Try to reverse subtraction problems: "A - B = ?" becomes "A - ? = C"  
        if re.search(r'-', equation):
            # Original: "Mary has 10 apples, gives away 3. How many left?" (10 - 3 = 7)
            # Reverse: "Mary wants to have 7 apples left, has 10 now. How many should she give away?" (10 - 7 = 3)
            
            nums = [n[1] for n in numbers]
            if len(nums) >= 2:
                first_num = nums[0]
                second_num = nums[1]
                new_remaining = answer
                new_answer = second_num
                
                new_question = re.sub(r'\b' + str(int(second_num)) + r'\b', '?', question, count=1)
                
                # Update phrasing for reverse subtraction
                new_question = re.sub(r'How many .* left\?', f'To have {int(new_remaining)} left, how many should be removed?', new_question, flags=re.IGNORECASE)
                new_question = re.sub(r'gives? away \?', f'wants {int(new_remaining)} left. How many to give away?', new_question)
                new_question = re.sub(r'ate \?', f'wants {int(new_remaining)} left. How many to eat?', new_question)
                
                new_equation = f"{first_num} - X = {new_remaining}"
                
                if abs(new_answer - answer) > 0.01 and new_question != question and new_answer >= 0:
                    return new_question, new_equation, new_answer, "reversed_direction"
        
        # Try to reverse multiplication: "A * B = ?" becomes "? * B = C"
        if re.search(r'\*', equation):
            nums = [n[1] for n in numbers] 
            if len(nums) >= 2:
                first_num = nums[0]
                second_num = nums[1]
                new_total = answer
                new_answer = first_num
                
                new_question = re.sub(r'\b' + str(int(first_num)) + r'\b', '?', question, count=1)
                new_question = re.sub(r'How many total\?', f'If there are {int(new_total)} total, how many groups?', new_question, flags=re.IGNORECASE)
                
                new_equation = f"X * {second_num} = {new_total}"
                
                if abs(new_answer - answer) > 0.01 and new_question != question and new_answer >= 0:
                    return new_question, new_equation, new_answer, "reversed_direction"
        
        # Try to reverse division: "A / B = ?" becomes "A / ? = C"
        if re.search(r'/', equation):
            nums = [n[1] for n in numbers]
            if len(nums) >= 2:
                first_num = nums[0]
                second_num = nums[1]
                new_per_group = answer
                new_answer = second_num
                
                new_question = re.sub(r'\b' + str(int(second_num)) + r'\b', '?', question, count=1)
                new_question = re.sub(r'How many .* each\?', f'To get {int(new_per_group)} each, how many groups?', new_question, flags=re.IGNORECASE)
                
                new_equation = f"{first_num} / X = {new_per_group}"
                
                if abs(new_answer - answer) > 0.01 and new_question != question and new_answer >= 0:
                    return new_question, new_equation, new_answer, "reversed_direction"
                    
        return question, equation, answer, "failed"
    
    def generate_variation(self, question: str, equation: str, answer: float, strategy: str) -> Tuple[str, str, float, str]:
        """Generate variation using specified strategy"""
        
        if strategy == "single_number":
            return self.change_single_number(question, equation, answer)
        elif strategy == "multi_number":
            return self.change_multiple_numbers(question, equation, answer)
        elif strategy == "add_step":
            return self.add_extra_step(question, equation, answer)
        elif strategy == "reverse_direction":
            return self.reverse_direction(question, equation, answer)
        else:
            # Try all strategies until one works
            for fallback in ["single_number", "multi_number", "add_step"]:
                new_q, new_eq, new_a, result = self.generate_variation(question, equation, answer, fallback)
                if result != "failed":
                    return new_q, new_eq, new_a, result
            # If all fail, return as control
            return question, equation, answer, "failed"
    
    def improve_dataset(self, dataset, target_size: int = 500) -> List[Dict]:
        """Create dataset with 20% controls, 80% calculation changes"""
        
        # Strategy distribution
        num_controls = int(target_size * 0.20)  # 20% controls
        num_changes = target_size - num_controls  # 80% changes
        
        # Distribute change strategies evenly across the three types
        strategies = (
            ["control"] * num_controls +
            ["single_number"] * int(num_changes * 0.333) +   # ~26.7% of total
            ["multi_number"] * int(num_changes * 0.333) +    # ~26.7% of total
            ["add_step"] * int(num_changes * 0.333)          # ~26.7% of total
        )
        
        # Pad to exact size if needed
        while len(strategies) < target_size:
            strategies.append("single_number")
        strategies = strategies[:target_size]
        
        # Shuffle for random distribution
        random.shuffle(strategies)
        
        improved_data = []
        failed_count = 0
        
        for i in range(target_size):
            rec = dataset[i % len(dataset)]
            strategy = strategies[i]
            
            # Process original record
            question = rec["Question"]
            answer = rec["Answer"] 
            equation = rec["Equation"]
            numbers = rec["Numbers"]
            
            # Substitute N_XX tokens
            real_question = self.substitute_numbers_in_text(question, numbers)
            real_equation = self.substitute_numbers_in_equation(equation, numbers)
            
            # Generate variation
            if strategy == "control":
                alt_question, alt_answer, variation_type = real_question, answer, None
                alt_equation = None
            else:
                alt_question, alt_equation, alt_answer, variation_type = self.generate_variation(
                    real_question, real_equation, answer, strategy
                )
            
            # Track failures
            if variation_type == "failed":
                failed_count += 1
                variation_type = None  # Convert to control
                alt_question = real_question
                alt_answer = answer
                alt_equation = None  # Control gets null equation
            
            improved_problem = {
                "Question": real_question,
                "Answer": answer,
                "alt_problem": alt_question,
                "alt_answer": alt_answer,
                "variation_type": variation_type,
                "Equation": real_equation,
                "alt_equation": alt_equation
            }
            
            improved_data.append(improved_problem)
        
        if failed_count > 0:
            print(f"⚠️  {failed_count} variations failed and became controls")
        
        return improved_data
    
    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """Analyze the distribution of variation types"""
        type_counts = {}
        control_count = 0
        changed_count = 0
        
        for problem in data:
            var_type = problem.get("variation_type")
            if var_type is None:
                control_count += 1
                type_counts["control"] = type_counts.get("control", 0) + 1
            else:
                changed_count += 1
                type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        return {
            "total_problems": len(data),
            "variation_types": type_counts,
            "control_count": control_count,
            "changed_count": changed_count,
            "control_percentage": (control_count / len(data)) * 100,
            "changed_percentage": (changed_count / len(data)) * 100
        }

def main():
    improver = DatasetImprover()
    
    print("Loading MAWPS dataset...")
    ds = load_dataset("mwpt5/MAWPS", split="train")
    
    print("Generating improved dataset with calculation changes...")
    improved_data = improver.improve_dataset(ds, target_size=500)
    
    # Analyze results
    analysis = improver.analyze_dataset(improved_data)
    print(f"\n📊 Dataset Analysis:")
    print(f"Total problems: {analysis['total_problems']}")
    print(f"Controls (identical): {analysis['control_count']} ({analysis['control_percentage']:.1f}%)")
    print(f"Changed calculations: {analysis['changed_count']} ({analysis['changed_percentage']:.1f}%)")
    print(f"\nVariation types:")
    for var_type, count in analysis['variation_types'].items():
        percentage = (count / analysis['total_problems']) * 100
        print(f"  {var_type}: {count} ({percentage:.1f}%)")
    
    # Show examples
    print(f"\n📝 Examples:")
    examples_shown = set()
    for problem in improved_data:
        var_type = problem["variation_type"]
        if var_type not in examples_shown:
            type_name = var_type if var_type else "CONTROL (identical)"
            print(f"\n{type_name}:")
            print(f"  Original: {problem['Question'][:80]}...")
            print(f"  Modified: {problem['alt_problem'][:80]}...")
            print(f"  Answers: {problem['Answer']} → {problem['alt_answer']}")
            if problem['Answer'] != problem['alt_answer']:
                diff = problem['alt_answer'] - problem['Answer']
                print(f"  Difference: {diff:+.2f}")
            examples_shown.add(var_type)
            
        if len(examples_shown) >= 6:
            break
    
    # Save
    output_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/math/mawps_augmented.json"
    with open(output_path, 'w') as f:
        json.dump(improved_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"✅ Ready for experiments with proper 80/20 split!")

if __name__ == "__main__":
    main()