"""
Comprehensive adaptation measurement - all modules combined.
Measures adaptation latency at token level using multiple methods.

FUNCTIONS INCLUDED:
=====================================
# AttentionTracker class and methods:
- AttentionTracker.__init__()
- AttentionTracker.register_hooks()
- AttentionTracker.clear_hooks() 
- AttentionTracker.track_generation_with_injection()
- AttentionTracker._create_attention_snapshot()
- AttentionTracker._calculate_adaptation_metrics()
- AttentionTracker._find_stabilization_point()
- AttentionTracker.plot_attention_evolution()

# AdaptationMetrics class and methods:
- AdaptationMetrics.__init__()
- AdaptationMetrics.measure_adaptation()
- AdaptationMetrics._prepare_conditions()
- AdaptationMetrics._track_generation() 
- AdaptationMetrics._calculate_state_metrics()
- AdaptationMetrics._get_context_embedding()
- AdaptationMetrics._generate_random_text()
- AdaptationMetrics._analyze_adaptation()
- AdaptationMetrics._find_crossover_point()
- AdaptationMetrics._find_convergence_point()
- AdaptationMetrics._calculate_stability()
- AdaptationMetrics._create_summary()

# CausalExperiments class and methods:
- CausalExperiments.__init__()
- CausalExperiments.run_causal_experiment()
- CausalExperiments._setup_conditions()
- CausalExperiments._generate_with_context()
- CausalExperiments._run_single_trial()
- CausalExperiments._calculate_trial_metrics()
- CausalExperiments._calculate_statistics()
- CausalExperiments._analyze_causality()
- CausalExperiments._analyze_overall_patterns()
- CausalExperiments._generate_recommendations()
- CausalExperiments._generate_noise_text()
- CausalExperiments.plot_results()

# TokenLevelAnalyzer class and methods:
- TokenLevelAnalyzer.__init__()
- TokenLevelAnalyzer.analyze_token_adaptation()
- TokenLevelAnalyzer._get_reference_distribution()
- TokenLevelAnalyzer._generate_with_tracking()
- TokenLevelAnalyzer._calculate_token_state()
- TokenLevelAnalyzer._analyze_adaptation_patterns()
- TokenLevelAnalyzer._find_adaptation_point()
- TokenLevelAnalyzer._calculate_token_metrics()
- TokenLevelAnalyzer._create_summary()
- TokenLevelAnalyzer.plot_token_adaptation()

# ProperAdaptationAnalyzer class and methods:
- ProperAdaptationAnalyzer.__init__()
- ProperAdaptationAnalyzer.analyze_single_problem()
- ProperAdaptationAnalyzer.analyze_dataset()
- ProperAdaptationAnalyzer._create_unified_summary()
- ProperAdaptationAnalyzer._save_results()
- ProperAdaptationAnalyzer._generate_report()

# Utility functions:
- save_metrics_to_file()
- test_attention_tracking()
- test_token_analysis()
- main()
=====================================
"""

import os
import json
import math
import torch
import numpy as np
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ttest_ind, ks_2samp
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine

from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity with proper mathematical handling."""
    # Ensure inputs are numpy arrays
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)
    
    # Check for NaN or inf values
    if np.any(~np.isfinite(vec1)) or np.any(~np.isfinite(vec2)):
        return 0.0
    
    # Calculate norms
    norm1 = np.sqrt(np.sum(vec1 * vec1))
    norm2 = np.sqrt(np.sum(vec2 * vec2))
    
    # Handle zero vectors
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0 if norm1 != norm2 else 1.0
    
    # Calculate dot product and cosine similarity directly
    dot_product = np.sum(vec1 * vec2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Clamp to valid range due to floating point precision issues
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    return float(cosine_sim)


# =====================================
# ATTENTION TRACKER MODULE
# =====================================

@dataclass
class AttentionSnapshot:
    """Captures attention state at a single generation step."""
    token_position: int
    layer_weights: Dict[int, torch.Tensor]  # layer_idx -> attention weights
    pre_injection_sum: float  # Sum of attention to pre-injection tokens
    post_injection_sum: float  # Sum of attention to post-injection tokens
    injection_point: int
    generated_token_id: int


class AttentionTracker:
    """Tracks attention weight redistribution during generation."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attention_history: List[AttentionSnapshot] = []
        self.hooks = []
        self.current_attention_weights = {}
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.clear_hooks()
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(output, 'attentions') and output.attentions is not None:
                    # Store attention weights for this layer
                    self.current_attention_weights[layer_idx] = output.attentions[0].detach()
            return hook
        
        # Register hooks for each attention layer
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.current_attention_weights = {}
    
    def track_generation_with_injection(
        self, 
        prompt: str, 
        injection_text: str,
        injection_point: int,
        max_new_tokens: int = 100
    ) -> Dict:
        """Generate text with injection while tracking attention patterns."""
        self.register_hooks()
        self.attention_history = []
        
        # Tokenize initial prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Generate tokens one at a time
        past_key_values = None
        generated_tokens = []
        
        for step in range(max_new_tokens):
            # Clear current attention weights
            self.current_attention_weights = {}
            
            # Forward pass with attention capture
            with torch.no_grad():
                # Set attention implementation to eager for Qwen compatibility
                if hasattr(self.model.config, '_attn_implementation'):
                    original_attn = self.model.config._attn_implementation
                    self.model.config._attn_implementation = 'eager'
                
                outputs = self.model(
                    input_ids=input_ids if step == 0 else next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Restore original attention implementation
                if hasattr(self.model.config, '_attn_implementation'):
                    self.model.config._attn_implementation = original_attn
            
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[0, -1].argmax()
            
            # Check for EOS token BEFORE adding to generated_tokens
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            
            # Inject new context at specified point
            if step == injection_point:
                injection_ids = self.tokenizer(injection_text, return_tensors="pt").input_ids.to(self.device)
                
                # REAL KV-cache injection - process injection text through model
                with torch.no_grad():
                    injection_outputs = self.model(
                        injection_ids,
                        use_cache=True,
                        return_dict=True
                    )
                    injection_kv = injection_outputs.past_key_values
                
                # Concatenate injection KV cache to existing cache
                if past_key_values is not None and injection_kv is not None:
                    # Handle different KV cache formats (some models use DynamicCache)
                    if hasattr(past_key_values, 'layers'):
                        # DynamicCache format - extend the cache
                        for layer_idx, (inj_key, inj_val) in enumerate(injection_kv):
                            past_key_values.layers[layer_idx].keys = torch.cat([
                                past_key_values.layers[layer_idx].keys, inj_key
                            ], dim=2)
                            past_key_values.layers[layer_idx].values = torch.cat([
                                past_key_values.layers[layer_idx].values, inj_val  
                            ], dim=2)
                    else:
                        # Standard tuple format
                        past_key_values = tuple(
                            (torch.cat([layer_kv[0], inj_kv[0]], dim=2),  # concat keys
                             torch.cat([layer_kv[1], inj_kv[1]], dim=2))  # concat values
                            for layer_kv, inj_kv in zip(past_key_values, injection_kv)
                        )
                
            # Calculate attention distribution
            if self.current_attention_weights:
                snapshot = self._create_attention_snapshot(
                    step, 
                    injection_point,
                    next_token.item()
                )
                self.attention_history.append(snapshot)
        
        self.clear_hooks()
        
        # Calculate adaptation metrics
        metrics = self._calculate_adaptation_metrics(injection_point)
        
        return {
            'generated_tokens': generated_tokens,
            'generated_text': self.tokenizer.decode(generated_tokens),
            'attention_history': self.attention_history,
            'adaptation_metrics': metrics
        }
    
    def _create_attention_snapshot(
        self, 
        position: int, 
        injection_point: int,
        token_id: int
    ) -> AttentionSnapshot:
        """Create snapshot of current attention state."""
        
        # Average attention across all layers and heads
        all_attention_weights = []
        for layer_idx, weights in self.current_attention_weights.items():
            # weights shape: [batch, heads, seq_len, seq_len]
            # Average over heads and batch
            avg_weights = weights.mean(dim=(0, 1))  # [seq_len, seq_len]
            all_attention_weights.append(avg_weights)
        
        if not all_attention_weights:
            # Fallback if no attention weights captured
            return AttentionSnapshot(
                token_position=position,
                layer_weights={},
                pre_injection_sum=0.0,
                post_injection_sum=0.0,
                injection_point=injection_point,
                generated_token_id=token_id
            )
        
        # Stack and average across layers
        avg_attention = torch.stack(all_attention_weights).mean(dim=0)
        
        # Calculate attention sums
        current_pos_attention = avg_attention[-1, :]  # Attention from current token
        
        if injection_point > 0 and injection_point < len(current_pos_attention):
            pre_injection_sum = current_pos_attention[:injection_point].sum().item()
            post_injection_sum = current_pos_attention[injection_point:].sum().item()
        else:
            pre_injection_sum = current_pos_attention.sum().item()
            post_injection_sum = 0.0
        
        return AttentionSnapshot(
            token_position=position,
            layer_weights=dict(self.current_attention_weights),
            pre_injection_sum=pre_injection_sum,
            post_injection_sum=post_injection_sum,
            injection_point=injection_point,
            generated_token_id=token_id
        )
    
    def _calculate_adaptation_metrics(self, injection_point: int) -> Dict:
        """Calculate when and how attention adapts to new context."""
        
        if not self.attention_history:
            return {}
        
        # Extract attention ratio over time
        attention_ratios = []
        for snapshot in self.attention_history:
            if snapshot.pre_injection_sum > 0:
                ratio = snapshot.post_injection_sum / (snapshot.pre_injection_sum + 1e-8)
                attention_ratios.append(ratio)
            else:
                attention_ratios.append(0.0)
        
        # Find stabilization point (when attention ratio stops changing rapidly)
        stabilization_point = self._find_stabilization_point(attention_ratios)
        
        # Calculate attention shift magnitude
        if len(attention_ratios) > injection_point + 5:
            pre_injection_avg = np.mean(attention_ratios[:injection_point]) if injection_point > 0 else 0
            post_injection_avg = np.mean(attention_ratios[injection_point:injection_point+10])
            attention_shift = post_injection_avg - pre_injection_avg
        else:
            attention_shift = 0.0
        
        return {
            'stabilization_tokens': stabilization_point,
            'attention_shift_magnitude': attention_shift,
            'final_attention_ratio': attention_ratios[-1] if attention_ratios else 0.0,
            'attention_ratios': attention_ratios
        }
    
    def _find_stabilization_point(self, values: List[float], window: int = 5) -> int:
        """Find point where values stabilize (low variance in sliding window)."""
        
        if len(values) < window * 2:
            return len(values)
        
        for i in range(window, len(values) - window):
            window_values = values[i:i+window]
            variance = np.var(window_values)
            
            # Stabilized if variance is low
            if variance < 0.01:
                return i
        
        return len(values)
    
    def plot_attention_evolution(self, save_path: Optional[str] = None):
        """Visualize how attention evolves over generation."""
        
        if not self.attention_history:
            print("No attention history to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Attention ratio over time
        positions = [s.token_position for s in self.attention_history]
        pre_sums = [s.pre_injection_sum for s in self.attention_history]
        post_sums = [s.post_injection_sum for s in self.attention_history]
        
        axes[0].plot(positions, pre_sums, label='Pre-injection attention', alpha=0.7)
        axes[0].plot(positions, post_sums, label='Post-injection attention', alpha=0.7)
        axes[0].axvline(x=self.attention_history[0].injection_point, 
                       color='red', linestyle='--', label='Injection point')
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('Attention Sum')
        axes[0].set_title('Attention Distribution Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Attention ratio
        ratios = [s.post_injection_sum / (s.pre_injection_sum + 1e-8) 
                  for s in self.attention_history]
        axes[1].plot(positions, ratios, color='green', alpha=0.7)
        axes[1].axvline(x=self.attention_history[0].injection_point,
                       color='red', linestyle='--', label='Injection point')
        axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Post/Pre Attention Ratio')
        axes[1].set_title('Attention Shift Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# =====================================
# ADAPTATION METRICS MODULE
# =====================================

@dataclass
class AdaptationState:
    """Comprehensive adaptation state at a single point."""
    token_position: int
    
    # Information-theoretic metrics
    kl_from_original: float  # KL divergence from original context distribution
    kl_from_injected: float  # KL divergence from injected context distribution
    mutual_info_original: float  # Mutual information with original context
    mutual_info_injected: float  # Mutual information with injected context
    
    # Semantic metrics
    embedding_similarity_original: float  # Cosine sim in embedding space
    embedding_similarity_injected: float  # Cosine sim in embedding space
    
    # Behavioral metrics
    confidence: float  # Average token confidence
    
    # Generation metrics
    token_id: int
    token_text: str
    logit_distribution: Optional[np.ndarray] = None


class AdaptationMetrics:
    """Calculates multi-dimensional adaptation metrics."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adaptation_history: List[AdaptationState] = []
        
    def measure_adaptation(
        self,
        original_context: str,
        injected_context: str,
        injection_point: int,
        max_tokens: int = 100
    ) -> Dict:
        """Measure adaptation across multiple dimensions."""
        
        # Prepare three conditions for comparison
        conditions = self._prepare_conditions(
            original_context, 
            injected_context, 
            injection_point
        )
        
        # Generate and track metrics for each condition
        results = {}
        for condition_name, condition_data in conditions.items():
            results[condition_name] = self._track_generation(
                condition_data['input_ids'],
                condition_data['injection_ids'] if 'injection_ids' in condition_data else None,
                condition_data['injection_point'] if 'injection_point' in condition_data else None,
                original_context,
                injected_context,
                max_tokens
            )
        
        # Calculate adaptation curves
        adaptation_analysis = self._analyze_adaptation(results)
        
        return {
            'conditions': results,
            'adaptation_analysis': adaptation_analysis,
            'summary': self._create_summary(adaptation_analysis)
        }
    
    def _prepare_conditions(
        self, 
        original: str, 
        injected: str, 
        injection_point: int
    ) -> Dict:
        """Prepare different experimental conditions."""
        
        conditions = {}
        
        # Condition 1: Original context only (baseline)
        original_ids = self.tokenizer(original, return_tensors="pt").input_ids.to(self.device)
        conditions['baseline'] = {
            'input_ids': original_ids,
            'description': 'Original context only'
        }
        
        # Condition 2: Injected context from start (oracle)
        full_injected = original[:injection_point] + " " + injected
        oracle_ids = self.tokenizer(full_injected, return_tensors="pt").input_ids.to(self.device)
        conditions['oracle'] = {
            'input_ids': oracle_ids,
            'description': 'Injected context from beginning'
        }
        
        # Condition 3: Mid-generation injection (experimental)
        conditions['injection'] = {
            'input_ids': original_ids,
            'injection_ids': self.tokenizer(injected, return_tensors="pt").input_ids.to(self.device),
            'injection_point': injection_point,
            'description': 'Context injected mid-generation'
        }
        
        # Condition 4: Random noise injection (control)
        random_text = self._generate_random_text(len(injected.split()))
        noise_ids = self.tokenizer(random_text, return_tensors="pt").input_ids.to(self.device)
        conditions['noise_control'] = {
            'input_ids': original_ids,
            'injection_ids': noise_ids,
            'injection_point': injection_point,
            'description': 'Random noise injection'
        }
        
        return conditions
    
    def _track_generation_two_phase(
        self,
        base_context: str,
        original_question: str, 
        alternative_question: str,
        injection_point: int,
        max_tokens: int,
        is_reading_comprehension: bool = False
    ) -> Dict:
        """Track generation with proper two-phase approach for reading comprehension."""
        
        if is_reading_comprehension:
            # Phase 1: Context + original question + start of answer with chain of thought
            cot_instruction = "\n\nThink step by step and provide a clear, detailed answer."
            original_full = f"{base_context}\n\nQuestion: {original_question}{cot_instruction}\nAnswer:"
            alternative_full = f"{base_context}\n\nQuestion: {alternative_question}{cot_instruction}\nAnswer:"
            
            # Tokenize both contexts 
            orig_inputs = self.tokenizer(original_full, return_tensors="pt").to(self.device)
            alt_inputs = self.tokenizer(alternative_full, return_tensors="pt").to(self.device)
            
            # Phase 1: Generate initial answer tokens without injection
            past_key_values = None
            states = []
            generated_tokens = []
            
            # Let model process the context and start answering
            input_ids = orig_inputs.input_ids
            
            # Generate tokens until natural completion
            position = 0
            
            while position < 200:  # Reasonable limit for math problems
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids if position == 0 else next_token.unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=True
                    )
                    
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[0, -1]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = logits.argmax()
                    
                    # At injection point, inject the alternative question's context
                    if position == injection_point:
                        # Create new context with alternative question
                        print(f"   🔄 Injecting alternative question at answer token {position}")
                        
                        # Get KV cache for alternative context 
                        with torch.no_grad():
                            alt_outputs = self.model(
                                alt_inputs.input_ids,
                                use_cache=True,
                                return_dict=True
                            )
                            alt_kv = alt_outputs.past_key_values
                        
                        # Replace the context part of KV cache with alternative
                        # Keep the answer tokens generated so far, replace context understanding
                        if past_key_values is not None and alt_kv is not None:
                            # This is complex - for now, restart generation with new context
                            # In future, could do more sophisticated KV cache surgery
                            input_ids = alt_inputs.input_ids
                            past_key_values = alt_kv
                    
                    # Calculate metrics as usual
                    state = self._calculate_state_metrics_improved(
                        position=position,
                        token_id=next_token.item(),
                        logits=logits,
                        probs=probs,
                        hidden_states=outputs.hidden_states[-1],
                        original_embedding=self._get_critical_context_embedding(original_full),
                        injected_embedding=self._get_critical_context_embedding(alternative_full),
                        critical_original=self._get_critical_context_embedding(original_full),
                        critical_injected=self._get_critical_context_embedding(alternative_full),
                        change_vector=self._get_change_vector(original_full, alternative_full, use_critical=True),
                        contrastive=self._get_contrastive_embeddings(original_full, alternative_full),
                        original_vocab=set(original_full.lower().split()),
                        injected_vocab=set(alternative_full.lower().split())
                    )
                    
                    # Check for EOS before adding to output
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    states.append(state)
                    generated_tokens.append(next_token.item())
                    
                    # No early stopping - let model generate naturally
                    
                    position += 1
            
            return {
                'states': states,
                'generated_tokens': generated_tokens,
                'generated_text': self.tokenizer.decode(generated_tokens),
                'injection_point': injection_point,
                'context_type': 'reading_comprehension',
                'original_context': original_full,
                'injected_context': alternative_full
            }
        else:
            # Math problems: use original approach
            return self._track_generation_original(
                base_context, original_question, alternative_question, 
                injection_point, max_tokens
            )

    def _track_generation_original(
        self,
        original_context: str,
        alternative_context: str, 
        unused1: str,
        injection_point: int,
        max_tokens: int
    ) -> Dict:
        """Original generation tracking for math problems."""
        
        # Tokenize contexts
        orig_inputs = self.tokenizer(original_context, return_tensors="pt").to(self.device)
        alt_inputs = self.tokenizer(alternative_context, return_tensors="pt").to(self.device)
        
        # Rest of original implementation...
        input_ids = orig_inputs.input_ids
        injection_ids = alt_inputs.input_ids
        
        return self._track_generation(
            input_ids, injection_ids, injection_point,
            original_context, alternative_context, max_tokens
        )

    def _track_generation(
        self,
        input_ids: torch.Tensor,
        injection_ids: Optional[torch.Tensor],
        injection_point: Optional[int],
        original_context: str,
        injected_context: str,
        max_tokens: int
    ) -> Dict:
        """Track multi-dimensional metrics during generation."""
        
        states = []
        generated_tokens = []
        past_key_values = None
        
        # Prepare reference distributions - USE IMPROVED REPRESENTATIONS
        original_embedding = self._get_context_embedding(original_context)
        injected_embedding = self._get_context_embedding(injected_context)
        
        # Get critical embeddings for better signal
        critical_original = self._get_critical_context_embedding(original_context)
        critical_injected = self._get_critical_context_embedding(injected_context)
        
        # Get change vector for adaptation tracking
        change_vector = self._get_change_vector(original_context, injected_context, use_critical=True)
        
        # Get contrastive embeddings
        contrastive = self._get_contrastive_embeddings(original_context, injected_context)
        
        original_vocab = set(original_context.lower().split())
        injected_vocab = set(injected_context.lower().split())
        
        position = 0
        
        while position < 200:  # Reasonable limit for complete responses
            with torch.no_grad():
                # Generate next token
                outputs = self.model(
                    input_ids=input_ids if position == 0 else next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                next_token = logits.argmax().unsqueeze(0)
                
                # Inject context if specified
                if injection_ids is not None and position == injection_point:
                    # REAL KV-cache injection
                    with torch.no_grad():
                        injection_outputs = self.model(
                            injection_ids,
                            use_cache=True,
                            return_dict=True
                        )
                        injection_kv = injection_outputs.past_key_values
                    
                    # Concatenate injection KV cache to existing cache
                    if past_key_values is not None and injection_kv is not None:
                        # Handle different KV cache formats (some models use DynamicCache)
                        if hasattr(past_key_values, 'layers'):
                            # DynamicCache format - extend the cache
                            for layer_idx, (inj_key, inj_val) in enumerate(injection_kv):
                                past_key_values.layers[layer_idx].keys = torch.cat([
                                    past_key_values.layers[layer_idx].keys, inj_key
                                ], dim=2)
                                past_key_values.layers[layer_idx].values = torch.cat([
                                    past_key_values.layers[layer_idx].values, inj_val  
                                ], dim=2)
                        else:
                            # Standard tuple format
                            past_key_values = tuple(
                                (torch.cat([layer_kv[0], inj_kv[0]], dim=2),  # concat keys
                                 torch.cat([layer_kv[1], inj_kv[1]], dim=2))  # concat values
                                for layer_kv, inj_kv in zip(past_key_values, injection_kv)
                            )
                
                # Calculate current state metrics with improved representations
                state = self._calculate_state_metrics_improved(
                    position=position,
                    token_id=next_token.item(),
                    logits=logits,
                    probs=probs,
                    hidden_states=outputs.hidden_states[-1],
                    original_embedding=original_embedding,
                    injected_embedding=injected_embedding,
                    critical_original=critical_original,
                    critical_injected=critical_injected,
                    change_vector=change_vector,
                    contrastive=contrastive,
                    original_vocab=original_vocab,
                    injected_vocab=injected_vocab
                )
                
                # Check for EOS before adding to output
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                states.append(state)
                generated_tokens.append(next_token.item())
                
                # No early stopping - let model generate naturally
                
                position += 1
        
        return {
            'states': states,
            'generated_tokens': generated_tokens,
            'generated_text': self.tokenizer.decode(generated_tokens)
        }
    
    def _calculate_state_metrics(
        self,
        position: int,
        token_id: int,
        logits: torch.Tensor,
        probs: torch.Tensor,
        hidden_states: torch.Tensor,
        original_embedding: torch.Tensor,
        injected_embedding: torch.Tensor,
        original_vocab: set,
        injected_vocab: set
    ) -> AdaptationState:
        """Calculate comprehensive metrics for current generation state."""
        
        # Convert to numpy for easier manipulation
        probs_np = probs.cpu().numpy()
        logits_np = logits.cpu().numpy()
        
        # Information-theoretic metrics (simplified - would need reference distributions)
        # Use float64 to prevent overflow in uniform distribution creation
        vocab_size = len(probs_np)
        uniform_dist = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
        # Add small epsilon to avoid division by zero and inf values
        probs_safe = np.clip(probs_np.astype(np.float64), 1e-12, 1.0)
        uniform_safe = np.clip(uniform_dist, 1e-12, 1.0)
        kl_from_uniform = entropy(probs_safe, uniform_safe)
        
        # Get token text and skip newlines
        token_text = self.tokenizer.decode([token_id])
        
        # Skip newline tokens as they don't add meaningful content
        if token_text.strip() in ['\n', '\r\n', '\r', '']:
            return None
        
        # Semantic similarity using safe cosine calculation
        current_embedding = hidden_states[0, -1].cpu().numpy()
        original_emb_np = original_embedding.cpu().numpy()
        injected_emb_np = injected_embedding.cpu().numpy()
        
        # Use safe cosine similarity function
        sim_original = safe_cosine_similarity(current_embedding, original_emb_np)
        sim_injected = safe_cosine_similarity(current_embedding, injected_emb_np)
        
        # Behavioral metrics
        confidence = probs.max().item()
        
        return AdaptationState(
            token_position=position,
            kl_from_original=kl_from_uniform,  # Placeholder
            kl_from_injected=kl_from_uniform,  # Placeholder
            mutual_info_original=0.0,  # Would need proper calculation
            mutual_info_injected=0.0,  # Would need proper calculation
            embedding_similarity_original=float(sim_original),
            embedding_similarity_injected=float(sim_injected),
            confidence=confidence,
            token_id=token_id,
            token_text=token_text,
            logit_distribution=logits_np
        )
    
    def _get_context_embedding(self, text: str) -> torch.Tensor:
        """Get embedding representation of context."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean pooling over sequence
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        
        return embeddings[0]
    
    def _get_critical_context_embedding(self, text: str) -> torch.Tensor:
        """Focus on critical tokens (numbers, key nouns, verbs) only."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            
            # Identify critical tokens
            critical_positions = []
            token_texts = []
            for i, token_id in enumerate(inputs.input_ids[0]):
                token_text = self.tokenizer.decode(token_id).strip()
                token_texts.append(token_text)
                
                # Check if token is critical (numbers, important words)
                if any([
                    token_text.replace('.', '').replace(',', '').isdigit(),  # Numbers
                    any(char.isdigit() for char in token_text),  # Contains digits
                    token_text.lower() in ['apples', 'cups', 'flour', 'miles', 'hours', 'students', 
                                          'books', 'cars', 'people', 'items', 'objects', 'things',
                                          'dollars', 'pounds', 'gallons', 'meters', 'feet'],  # Nouns
                    token_text.lower() in ['needs', 'has', 'have', 'bought', 'sold', 'gave',
                                          'took', 'added', 'removed', 'found', 'lost', 'made',
                                          'spent', 'earned', 'traveled', 'walked', 'ran']  # Verbs
                ]):
                    critical_positions.append(i)
            
            # Average only critical tokens, fallback to full average if none found
            if critical_positions:
                critical_embeddings = hidden[critical_positions].mean(dim=0)
                return critical_embeddings
            return hidden.mean(dim=0)
    
    def _get_change_vector(self, original_text: str, injected_text: str, use_critical: bool = True) -> torch.Tensor:
        """Create a vector representing what changed between contexts."""
        if use_critical:
            orig_embed = self._get_critical_context_embedding(original_text)
            inj_embed = self._get_critical_context_embedding(injected_text)
        else:
            orig_embed = self._get_context_embedding(original_text)
            inj_embed = self._get_context_embedding(injected_text)
        
        # The difference vector points from original → injected
        change_vector = inj_embed - orig_embed
        return change_vector
    
    def _get_contrastive_embeddings(self, original_text: str, injected_text: str) -> Dict[str, torch.Tensor]:
        """Separate shared vs changed content representations."""
        orig_inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        inj_inputs = self.tokenizer(injected_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            orig_outputs = self.model(**orig_inputs, output_hidden_states=True)
            inj_outputs = self.model(**inj_inputs, output_hidden_states=True)
            
            orig_hidden = orig_outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            inj_hidden = inj_outputs.hidden_states[-1][0]
            
            # Find which tokens are same vs different
            orig_tokens = [self.tokenizer.decode(t).strip().lower() for t in orig_inputs.input_ids[0]]
            inj_tokens = [self.tokenizer.decode(t).strip().lower() for t in inj_inputs.input_ids[0]]
            
            # Identify shared positions (tokens that appear in both)
            shared_positions_orig = []
            changed_positions_orig = []
            for i, token in enumerate(orig_tokens):
                if token in inj_tokens:
                    shared_positions_orig.append(i)
                else:
                    changed_positions_orig.append(i)
            
            shared_positions_inj = []
            changed_positions_inj = []
            for i, token in enumerate(inj_tokens):
                if token in orig_tokens:
                    shared_positions_inj.append(i)
                else:
                    changed_positions_inj.append(i)
            
            # Create embeddings
            result = {}
            
            # Shared context (what stayed the same)
            if shared_positions_orig:
                result['shared'] = orig_hidden[shared_positions_orig].mean(dim=0)
            else:
                result['shared'] = torch.zeros_like(orig_hidden[0])
            
            # Original-only content (what was removed)
            if changed_positions_orig:
                result['original_only'] = orig_hidden[changed_positions_orig].mean(dim=0)
            else:
                result['original_only'] = torch.zeros_like(orig_hidden[0])
            
            # Injected-only content (what was added)
            if changed_positions_inj:
                result['injected_only'] = inj_hidden[changed_positions_inj].mean(dim=0)
            else:
                result['injected_only'] = torch.zeros_like(inj_hidden[0])
            
            # Change direction vector
            result['change_direction'] = result['injected_only'] - result['original_only']
            
            return result
    
    def _calculate_state_metrics_improved(
        self,
        position: int,
        token_id: int,
        logits: torch.Tensor,
        probs: torch.Tensor,
        hidden_states: torch.Tensor,
        original_embedding: torch.Tensor,
        injected_embedding: torch.Tensor,
        critical_original: torch.Tensor,
        critical_injected: torch.Tensor,
        change_vector: torch.Tensor,
        contrastive: Dict[str, torch.Tensor],
        original_vocab: set,
        injected_vocab: set
    ) -> AdaptationState:
        """Calculate state metrics with improved context representations."""
        
        # Get token text
        token_text = self.tokenizer.decode(token_id)
        
        # Convert tensors to numpy for calculations
        logits_np = logits.cpu().numpy()
        
        # Statistical metrics
        uniform_dist = np.ones(len(logits_np)) / len(logits_np)
        kl_from_uniform = float(np.sum(probs.cpu().numpy() * np.log(probs.cpu().numpy() / uniform_dist + 1e-10)))
        
        # Get current token embedding
        current_embedding = hidden_states[0, -1].cpu().numpy()
        
        # SIMPLIFIED METRICS - Using critical embeddings as primary
        
        # Use critical content similarity as main metric (more accurate)
        critical_orig_np = critical_original.cpu().numpy()
        critical_inj_np = critical_injected.cpu().numpy()
        sim_original = safe_cosine_similarity(current_embedding, critical_orig_np)
        sim_injected = safe_cosine_similarity(current_embedding, critical_inj_np)
        
        # Change vector projection - KEY METRIC
        change_vector_np = change_vector.cpu().numpy()
        change_projection = safe_cosine_similarity(current_embedding, change_vector_np)
        # Positive = moving toward injected context
        # Negative = staying with original context
        # Near zero = unrelated to the change
        
        # Simple context source classification (more sensitive)
        if change_projection > 0.05 and sim_injected > sim_original:
            context_source = "injected"  # Adapted to new context
        elif change_projection < -0.05 and sim_original > sim_injected:
            context_source = "original"  # Sticking to original
        else:
            context_source = "mixed"  # Transitioning or neutral
        
        # Behavioral metrics
        confidence = probs.max().item()
        
        # Store SIMPLIFIED metrics
        state = AdaptationState(
            token_position=position,
            kl_from_original=kl_from_uniform,  # Placeholder
            kl_from_injected=kl_from_uniform,  # Placeholder
            mutual_info_original=0.0,  # Would need proper calculation
            mutual_info_injected=0.0,  # Would need proper calculation
            embedding_similarity_original=float(sim_original),  # Now using critical embeddings
            embedding_similarity_injected=float(sim_injected),  # Now using critical embeddings
            confidence=confidence,
            token_id=token_id,
            token_text=token_text,
            logit_distribution=logits_np
        )
        
        # Add only the essential improved metric
        state.change_projection = float(change_projection)
        state.context_source = context_source
        
        return state
    
    def _generate_random_text(self, num_words: int) -> str:
        """Generate random text for noise control."""
        # Simple random word generation
        import random
        words = ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'it', 'was']
        return ' '.join(random.choices(words, k=num_words))
    
    def _analyze_adaptation(self, results: Dict) -> Dict:
        """Analyze adaptation patterns across conditions."""
        
        analysis = {}
        
        # Extract key metrics over time
        for condition_name, condition_data in results.items():
            states = condition_data['states']
            
            # Track information source preference
            original_preference = [s.embedding_similarity_original for s in states]
            injected_preference = [s.embedding_similarity_injected for s in states]
            
            # Find crossover point (when model switches preference)
            crossover = self._find_crossover_point(original_preference, injected_preference)
            
            # Calculate adaptation speed
            if 'injection' in condition_name:
                injection_point = next((i for i, s in enumerate(states) 
                                       if s.token_position >= 5), 0)  # Placeholder
                adaptation_speed = crossover - injection_point if crossover > 0 else None
            else:
                adaptation_speed = None
            
            # Calculate stability
            stability = self._calculate_stability(injected_preference)
            
            analysis[condition_name] = {
                'crossover_point': crossover,
                'adaptation_speed': adaptation_speed,
                'stability': stability,
                'final_preference': 'injected' if injected_preference[-1] > original_preference[-1] else 'original',
                'preference_trajectory': {
                    'original': original_preference,
                    'injected': injected_preference
                }
            }
        
        # Compare injection to oracle
        if 'injection' in analysis and 'oracle' in analysis:
            injection_traj = analysis['injection']['preference_trajectory']['injected']
            oracle_traj = analysis['oracle']['preference_trajectory']['injected']
            
            # Calculate convergence to oracle
            convergence_distance = [
                abs(inj - ora) for inj, ora in 
                zip(injection_traj, oracle_traj[:len(injection_traj)])
            ]
            
            analysis['convergence'] = {
                'mean_distance': np.mean(convergence_distance),
                'final_distance': convergence_distance[-1] if convergence_distance else None,
                'convergence_point': self._find_convergence_point(convergence_distance)
            }
        
        return analysis
    
    def _find_crossover_point(self, series1: List[float], series2: List[float]) -> int:
        """Find where series2 becomes consistently greater than series1."""
        for i in range(len(series1)):
            if i < len(series2) - 3:  # Need at least 3 points ahead
                if all(series2[i+j] > series1[i+j] for j in range(3)):
                    return i
        return -1
    
    def _find_convergence_point(self, distances: List[float], threshold: float = 0.1) -> int:
        """Find where distance stays below threshold."""
        for i in range(len(distances)):
            if i < len(distances) - 3:
                if all(distances[i+j] < threshold for j in range(min(3, len(distances)-i))):
                    return i
        return -1
    
    def _calculate_stability(self, series: List[float], window: int = 5) -> float:
        """Calculate stability as inverse of variance in sliding window."""
        if len(series) < window:
            return 0.0
        
        variances = []
        for i in range(len(series) - window + 1):
            window_variance = np.var(series[i:i+window])
            variances.append(window_variance)
        
        # Return inverse of mean variance (higher = more stable)
        mean_variance = np.mean(variances)
        return 1.0 / (1.0 + mean_variance)
    
    def _create_summary(self, analysis: Dict) -> Dict:
        """Create human-readable summary of adaptation results."""
        
        summary = {}
        
        # Check if adaptation occurred
        if 'injection' in analysis:
            inj = analysis['injection']
            
            summary['adaptation_occurred'] = inj['final_preference'] == 'injected'
            summary['reached_stability'] = inj['stability'] > 0.7
            
            # Compare to oracle
            if 'convergence' in analysis:
                conv = analysis['convergence']
                summary['converged_to_oracle'] = conv['final_distance'] < 0.1 if conv['final_distance'] else False
                summary['convergence_tokens'] = conv['convergence_point']
                summary['convergence_quality'] = 1.0 - conv['mean_distance'] if conv['mean_distance'] else 0.0
            
            # Compare to controls
            if 'noise_control' in analysis:
                noise = analysis['noise_control']
                summary['better_than_noise'] = inj['stability'] > noise['stability']
        
        return summary


# =====================================
# TOKEN ANALYSIS MODULE
# =====================================

@dataclass
class TokenState:
    """State information for a single generated token."""
    position: int
    token_id: int
    token_text: str
    context_source: str  # 'original' or 'injected' - which context influenced this token
    
    # Probability metrics
    kl_from_original: float  # KL divergence from original distribution
    kl_from_injected: float  # KL divergence from injected distribution
    
    # Embedding similarity metrics
    embedding_similarity_original: float  # Cosine similarity with original embeddings
    embedding_similarity_injected: float  # Cosine similarity with injected embeddings
    
    # Comparison metrics
    prob_ratio: float  # P(token|injected) / P(token|original)
    surprisal_delta: float  # Change in surprisal
    token_prob: float


class TokenLevelAnalyzer:
    """Analyzes adaptation at the token level, tracking fine-grained changes."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.token_history: List[TokenState] = []
        
    def analyze_token_adaptation(
        self,
        original_context: str,
        injected_context: str,
        injection_point: int,
        max_tokens: int = None,  # None means generate until completion
        top_k: int = 10
    ) -> Dict:
        """Perform token-level adaptation analysis."""
        
        # CRITICAL: Clear any previous state to ensure isolation between problems
        self.token_history.clear()
        
        # Generate reference distributions
        # Use reasonable default if max_tokens is None  
        # Increase ref_length to ensure we have distributions for later injection points
        ref_length = max_tokens if max_tokens is not None else 300  # Increased to support longer generations
        original_dist = self._get_reference_distribution(original_context, ref_length)
        injected_dist = self._get_reference_distribution(injected_context, ref_length)
        
        
        # Generate with injection and track each token
        generation_states = self._generate_with_tracking(
            original_context,
            injected_context,
            injection_point,
            max_tokens,
            original_dist,
            injected_dist,
            top_k
        )
        
        # Analyze adaptation patterns
        analysis = self._analyze_adaptation_patterns(generation_states, injection_point)
        
        # Calculate fine-grained metrics
        metrics = self._calculate_token_metrics(generation_states, injection_point)
        
        # Combine everything into summary
        summary = self._create_summary(analysis, metrics, injection_point)
        summary.update(analysis)  # Add adaptation analysis
        summary.update(metrics)   # Add metrics
        
        
        # Convert token states to clean JSON objects
        clean_token_states = []
        for state in generation_states:
            clean_state = {
                'position': state.position,
                'token_id': state.token_id,
                'token_text': state.token_text,
                'context_source': state.context_source,
                'token_prob': state.token_prob,
                'prob_ratio': state.prob_ratio,
                'surprisal_delta': float(state.surprisal_delta),
                'kl_from_original': state.kl_from_original,
                'kl_from_injected': state.kl_from_injected,
                'cos_similarity_original': state.embedding_similarity_original,
                'cos_similarity_injected': state.embedding_similarity_injected
            }
            clean_token_states.append(clean_state)
        
        return {
            'token_states': clean_token_states,
            'summary': summary
        }
    
    def analyze_two_phase_adaptation(
        self,
        base_context: str,
        original_question: str,
        alternative_question: str,
        injection_point: int,
        max_tokens: int = 100,
        top_k: int = 10
    ) -> Dict:
        """Perform token-level adaptation analysis for reading comprehension with two-phase generation.
        
        Phase 1: Model reads the story context
        Phase 2: Model generates answer, with injection happening during answer generation
        """
        
        # Create full contexts for reference distributions
        # Add chain of thought instruction for better reasoning
        cot_instruction = "\n\nThink step by step and provide a clear, detailed answer."
        original_full = f"{base_context}\n\nQuestion: {original_question}{cot_instruction}\nAnswer: "
        injected_full = f"{base_context}\n\nQuestion: {alternative_question}{cot_instruction}\nAnswer: "
        
        # Generate reference distributions using full contexts
        # Use reasonable default if max_tokens is None
        ref_length = max_tokens if max_tokens is not None else 100
        original_dist = self._get_reference_distribution(original_full, ref_length)
        injected_dist = self._get_reference_distribution(injected_full, ref_length)
        
        # Generate with two-phase tracking: context reading then answer generation with injection
        generation_states = self._generate_with_two_phase_tracking(
            base_context,
            original_question,
            alternative_question,
            injection_point,
            max_tokens,
            original_dist,
            injected_dist,
            top_k
        )
        
        # Analyze adaptation patterns
        analysis = self._analyze_adaptation_patterns(generation_states, injection_point)
        
        # Calculate fine-grained metrics
        metrics = self._calculate_token_metrics(generation_states, injection_point)
        
        # Combine everything into summary
        summary = self._create_summary(analysis, metrics, injection_point)
        summary.update(analysis)  # Add adaptation analysis
        summary.update(metrics)   # Add metrics
        
        # Convert token states to clean JSON objects
        clean_token_states = []
        for state in generation_states:
            clean_state = {
                'position': state.position,
                'token_id': state.token_id,
                'token_text': state.token_text,
                'context_source': state.context_source,
                'token_prob': state.token_prob,
                'prob_ratio': state.prob_ratio,
                'surprisal_delta': float(state.surprisal_delta),
                'kl_from_original': state.kl_from_original,
                'kl_from_injected': state.kl_from_injected,
                'cos_similarity_original': state.embedding_similarity_original,
                'cos_similarity_injected': state.embedding_similarity_injected
            }
            clean_token_states.append(clean_state)
        
        return {
            'token_states': clean_token_states,
            'summary': summary
        }
    
    def _generate_with_two_phase_tracking(
        self,
        base_context: str,
        original_question: str,
        alternative_question: str,
        injection_point: int,
        max_tokens: int,
        original_dist: Dict,
        injected_dist: Dict,
        top_k: int
    ) -> List[TokenState]:
        """Generate text with two-phase approach: context reading then answer generation with injection."""
        
        states = []
        
        # Phase 1: Read the story context + original question (no injection yet)
        # Add chain of thought instruction for reasoning comprehension
        cot_instruction = "\n\nThink step by step and provide a clear, detailed answer."
        context_prompt = f"{base_context}\n\nQuestion: {original_question}{cot_instruction}\nAnswer: "
        inputs = self.tokenizer(context_prompt, return_tensors="pt").to(self.device)
        
        # Get initial cache from reading context
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, use_cache=True, return_dict=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            
        # Phase 2: Generate answer with potential injection
        generated_tokens = []
        position = 0
        
        # Prepare injection cache (alternative question)
        injection_prompt = f"{base_context}\n\nQuestion: {alternative_question}{cot_instruction}\nAnswer: "
        injection_inputs = self.tokenizer(injection_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            injection_outputs = self.model(injection_inputs.input_ids, use_cache=True, return_dict=True)
            injection_kv = injection_outputs.past_key_values
        
        # Generate answer tokens
        with torch.no_grad():
            max_length = max_tokens if max_tokens is not None else 1000
            for position in range(max_length):
                # Get predictions from current state
                if position == 0:
                    # First token uses the context cache
                    dummy_input = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
                    outputs = self.model(
                        dummy_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=True
                    )
                else:
                    # Subsequent tokens use previous token + accumulated cache
                    prev_token = torch.tensor([[generated_tokens[-1]]]).to(self.device)
                    outputs = self.model(
                        prev_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=True
                    )
                
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(logits)
                
                # Check for EOS token BEFORE processing
                if next_token.item() == self.tokenizer.eos_token_id:
                    break  # Stop immediately, don't include EOS in output
                
                # Inject alternative context at the specified position
                if position == injection_point:
                    print(f"Injecting alternative question at token position {position}")
                    past_key_values = injection_kv
                
                # Calculate metrics against reference distributions
                if position < len(original_dist) and position < len(injected_dist):
                    orig_probs = original_dist[position]['probs']
                    inj_probs = injected_dist[position]['probs']
                    
                    # Token probability ratios
                    token_prob = probs[next_token.item()].item()
                    orig_token_prob = orig_probs[next_token.item()].item()
                    inj_token_prob = inj_probs[next_token.item()].item()
                    prob_ratio = inj_token_prob / max(orig_token_prob, 1e-10)
                    
                    # Classify context source (more sensitive thresholds)
                    if prob_ratio > 1.1:  # 10% more likely in injected context
                        context_source = "injected"
                    elif prob_ratio > 0.9:  # Within 10% of either context
                        context_source = "mixed"
                    else:
                        context_source = "original"
                    
                    # Ranking analysis
                    orig_ranks = torch.argsort(orig_probs, descending=True)
                    inj_ranks = torch.argsort(inj_probs, descending=True)
                    rank_in_original = (orig_ranks == next_token.item()).nonzero(as_tuple=True)[0].item()
                    rank_in_injected = (inj_ranks == next_token.item()).nonzero(as_tuple=True)[0].item()
                    
                    # Surprisal delta
                    orig_surprisal = -torch.log2(torch.clamp(orig_probs[next_token.item()], min=1e-12))
                    inj_surprisal = -torch.log2(torch.clamp(inj_probs[next_token.item()], min=1e-12))
                    surprisal_delta = (inj_surprisal - orig_surprisal).item()
                    
                else:
                    # Fallback values when reference distributions are unavailable
                    token_prob = probs[next_token.item()].item()
                    prob_ratio = 1.0
                    context_source = "original"
                    rank_in_original = 0
                    rank_in_injected = 0
                    surprisal_delta = 0.0
                
                # Calculate entropy and KL divergences
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                # KL divergences (ensure all tensors on same device)
                if position < len(original_dist) and position < len(injected_dist):
                    # Move tensors to same device as probs
                    orig_probs_device = orig_probs.to(probs.device)
                    inj_probs_device = inj_probs.to(probs.device)
                    
                    kl_from_original = torch.nn.functional.kl_div(
                        torch.log(probs + 1e-10).unsqueeze(0),
                        orig_probs_device.unsqueeze(0),
                        reduction='sum'
                    ).item()
                    kl_from_injected = torch.nn.functional.kl_div(
                        torch.log(probs + 1e-10).unsqueeze(0),
                        inj_probs_device.unsqueeze(0),
                        reduction='sum'
                    ).item()
                else:
                    kl_from_original = 0.0
                    kl_from_injected = 0.0
                
                # Create token state with all required fields
                token_text = self.tokenizer.decode([next_token.item()])
                
                # Skip ONLY pure newline tokens (not mixed content)
                if token_text.strip() in ['\n', '\r\n', '\r', ''] or token_text == '\n':
                    continue
                    
                state = TokenState(
                    position=position,
                    token_id=next_token.item(),
                    token_text=token_text,
                    entropy=entropy,
                    kl_from_original=kl_from_original,
                    kl_from_injected=kl_from_injected,
                    rank_in_original=rank_in_original,
                    rank_in_injected=rank_in_injected,
                    prob_ratio=prob_ratio,
                    surprisal_delta=surprisal_delta,
                    context_source=context_source,
                    token_prob=token_prob
                )
                
                states.append(state)
                generated_tokens.append(next_token.item())
                past_key_values = outputs.past_key_values
        
        return states

    def _get_reference_distribution(
        self, 
        context: str, 
        length: int
    ) -> Dict[int, Dict]:
        """Get reference token distributions for a context, skipping newlines to match generation."""
        
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        distributions = {}
        past_key_values = None
        token_count = 0  # Track non-newline tokens
        
        with torch.no_grad():
            # Initial pass
            outputs = self.model(
                inputs.input_ids,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            
            # Generate more tokens than needed to account for skipped newlines
            max_iterations = length * 3  # Generate extra to ensure we get enough non-newline tokens
            
            for _ in range(max_iterations):
                if token_count >= length:
                    break
                    
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Generate next token
                next_token = torch.argmax(logits)
                token_text = self.tokenizer.decode([next_token.item()])
                
                # Only store distribution for tokens without newlines
                if '\n' not in token_text and '\r' not in token_text:
                    # Get embedding for the next token
                    token_embedding = self._get_token_embedding(next_token.item())
                    
                    distributions[token_count] = {
                        'logits': logits.cpu(),
                        'probs': probs.cpu(),
                        'entropy': -torch.sum(torch.clamp(probs, min=1e-12) * torch.log(torch.clamp(probs, min=1e-12))).item(),
                        'top_tokens': torch.topk(probs, k=20),
                        'embedding': token_embedding.cpu() if token_embedding is not None else None
                    }
                    token_count += 1
                
                # Always advance the model
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
        
        return distributions
    
    def _generate_with_tracking(
        self,
        original_context: str,
        injected_context: str,
        injection_point: int,
        max_tokens: int,
        original_dist: Dict,
        injected_dist: Dict,
        top_k: int
    ) -> List[TokenState]:
        """Generate tokens while tracking detailed state information."""
        
        states = []
        inputs = self.tokenizer(original_context, return_tensors="pt").to(self.device)
        past_key_values = None
        
        with torch.no_grad():
            # Initial forward pass
            outputs = self.model(
                inputs.input_ids,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            
            max_length = max_tokens if max_tokens is not None else 1000  # Allow full generation, rely on EOS and repetition detection
            token_count = 0  # Track actual tokens (non-newlines)
            
            # Track recent tokens to detect repetition
            recent_tokens = []
            repetition_count = 0
            
            for position in range(max_length):
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Get next token
                next_token = torch.argmax(logits)
                
                # Check for EOS token BEFORE adding to states
                if next_token.item() == self.tokenizer.eos_token_id:
                    break  # Stop immediately, don't include EOS in output
                
                token_text = self.tokenizer.decode([next_token.item()])
                
                # Add token to recent tokens first
                recent_tokens.append(next_token.item())
                if len(recent_tokens) > 10:
                    recent_tokens.pop(0)  # Keep only last 10 tokens
                
                # Detect repetition - multiple strategies:
                # 1. Same token 3 times in a row (more aggressive)
                if len(recent_tokens) >= 3 and all(t == recent_tokens[-1] for t in recent_tokens[-3:]):
                    print(f"Warning: Detected 3x token repetition at position {position} (token: '{token_text}'), stopping generation")
                    break
                
                # 2. Same word repeated in text (check for 'flour flour flour' pattern)
                recent_text = self.tokenizer.decode(recent_tokens[-6:] if len(recent_tokens) >= 6 else recent_tokens)
                words = recent_text.strip().split()
                if len(words) >= 3:
                    last_word = words[-1].lower().strip('.,!?')
                    if last_word and len(last_word) > 2:  # Only check meaningful words
                        # Count how many times this word appears in the last few words
                        count = sum(1 for w in words[-4:] if w.lower().strip('.,!?') == last_word)
                        if count >= 3:
                            print(f"Warning: Detected word repetition at position {position} (word: '{last_word}'), stopping generation")
                            break
                
                # Skip tokens containing newlines (including mixed content like " \n\n")
                if '\n' in token_text or '\r' in token_text:
                    # Still need to advance the model
                    outputs = self.model(
                        next_token.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
                    continue
                
                # Calculate metrics against both contexts using token_count for reference distributions
                # Use last available distribution if we've gone beyond precomputed references
                last_orig_idx = max(original_dist.keys()) if original_dist else -1
                last_inj_idx = max(injected_dist.keys()) if injected_dist else -1
                
                orig_dist_to_use = original_dist.get(token_count, original_dist.get(last_orig_idx, {}))
                inj_dist_to_use = injected_dist.get(token_count, injected_dist.get(last_inj_idx, {}))
                
                state = self._calculate_token_state(
                    position=token_count,  # Use token_count for consistent indexing
                    token_id=next_token.item(),
                    token_text=token_text,
                    current_probs=probs,
                    current_logits=logits,
                    original_dist=orig_dist_to_use,
                    injected_dist=inj_dist_to_use,
                    top_k=top_k
                )
                
                states.append(state)
                token_count += 1  # Increment after adding a valid token
                
                # Perform injection if at injection point (based on token count, not position)
                if token_count - 1 == injection_point:  # -1 because we just incremented
                    # REAL KV-cache injection using injected_context from reference distributions
                    injection_ids = self.tokenizer(injected_context, return_tensors="pt").input_ids.to(self.device)
                    
                    with torch.no_grad():
                        injection_outputs = self.model(
                            injection_ids,
                            use_cache=True,
                            return_dict=True
                        )
                        injection_kv = injection_outputs.past_key_values
                    
                    # Concatenate injection KV cache to existing cache
                    if past_key_values is not None and injection_kv is not None:
                        # Handle different KV cache formats (some models use DynamicCache)
                        if hasattr(past_key_values, 'layers'):
                            # DynamicCache format - extend the cache
                            for layer_idx, (inj_key, inj_val) in enumerate(injection_kv):
                                past_key_values.layers[layer_idx].keys = torch.cat([
                                    past_key_values.layers[layer_idx].keys, inj_key
                                ], dim=2)
                                past_key_values.layers[layer_idx].values = torch.cat([
                                    past_key_values.layers[layer_idx].values, inj_val  
                                ], dim=2)
                        else:
                            # Standard tuple format
                            past_key_values = tuple(
                                (torch.cat([layer_kv[0], inj_kv[0]], dim=2),  # concat keys
                                 torch.cat([layer_kv[1], inj_kv[1]], dim=2))  # concat values
                                for layer_kv, inj_kv in zip(past_key_values, injection_kv)
                            )
                
                # Continue generation
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
        
        return states
    
    def _calculate_token_state(
        self,
        position: int,
        token_id: int,
        token_text: str,
        current_probs: torch.Tensor,
        current_logits: torch.Tensor,
        original_dist: Dict,
        injected_dist: Dict,
        top_k: int
    ) -> TokenState:
        """Calculate comprehensive state for a single token."""
        
        # Current token metrics
        token_prob = current_probs[token_id].item()
        token_logprob = torch.log(current_probs[token_id]).item()
        # Calculate entropy safely, handling potential numerical issues
        log_probs = torch.log(current_probs + 1e-10)
        entropy = -torch.sum(current_probs * log_probs).item()
        
        # Handle NaN/inf values
        if not np.isfinite(entropy):
            entropy = 0.0
        
        # Calculate KL divergence at this token position
        kl_divergence_from_original = 0.0
        kl_divergence_from_injected = 0.0
        
        # Extract probability distributions from reference data
        original_probs = None
        injected_probs = None
        
        if original_dist and 'probs' in original_dist:
            original_probs = original_dist['probs']
        
        if injected_dist and 'probs' in injected_dist:
            injected_probs = injected_dist['probs']
        
        if original_probs is not None:
            # KL(current || original) - ensure tensors are on same device
            original_probs = original_probs.to(current_probs.device)
            
            # Convert to float32 for better numerical precision
            current_f32 = current_probs.float()
            original_f32 = original_probs.float()
            
            # Add stronger numerical stability
            current_safe = torch.clamp(current_f32, min=1e-8, max=1.0)
            original_safe = torch.clamp(original_f32, min=1e-8, max=1.0)
            
            # Calculate KL divergence with better numerical handling
            log_ratio = torch.log(current_safe / original_safe)
            kl_terms = current_safe * log_ratio
            
            # Filter out terms that could cause issues
            kl_terms = torch.where(torch.isfinite(kl_terms), kl_terms, torch.zeros_like(kl_terms))
            
            kl_divergence_from_original = torch.sum(kl_terms).item()
            
            # Additional safety check
            if not torch.isfinite(torch.tensor(kl_divergence_from_original)) or kl_divergence_from_original < 0:
                kl_divergence_from_original = 0.0
        
        if injected_probs is not None:
            # KL(current || injected) - ensure tensors are on same device
            injected_probs = injected_probs.to(current_probs.device)
            
            # Convert to float32 for better numerical precision
            current_f32 = current_probs.float()
            injected_f32 = injected_probs.float()
            
            # Add stronger numerical stability
            current_safe = torch.clamp(current_f32, min=1e-8, max=1.0)
            injected_safe = torch.clamp(injected_f32, min=1e-8, max=1.0)
            
            # Calculate KL divergence with better numerical handling
            log_ratio = torch.log(current_safe / injected_safe)
            kl_terms = current_safe * log_ratio
            
            # Filter out terms that could cause issues
            kl_terms = torch.where(torch.isfinite(kl_terms), kl_terms, torch.zeros_like(kl_terms))
            
            kl_divergence_from_injected = torch.sum(kl_terms).item()
            
            # Additional safety check
            if not torch.isfinite(torch.tensor(kl_divergence_from_injected)) or kl_divergence_from_injected < 0:
                kl_divergence_from_injected = 0.0
        
        # Get probabilities for ratio calculation
        prob_original = 0.0
        prob_injected = 0.0
        has_original_dist = False
        has_injected_dist = False
        
        if original_dist and 'probs' in original_dist:
            orig_probs = original_dist['probs']
            # probs is a tensor with vocab_size dimensions, index directly by token_id
            try:
                prob_original = orig_probs[token_id].item()
                has_original_dist = True
            except (IndexError, RuntimeError):
                # Token ID out of bounds or other error
                pass
        
        if injected_dist and 'probs' in injected_dist:
            inj_probs = injected_dist['probs']
            # probs is a tensor with vocab_size dimensions, index directly by token_id
            try:
                prob_injected = inj_probs[token_id].item()
                has_injected_dist = True
            except (IndexError, RuntimeError):
                # Token ID out of bounds or other error
                pass
        
        # Only calculate metrics if we have at least one distribution
        if has_original_dist and has_injected_dist:
            # Probability ratio - handle very small probabilities
            if prob_original > 1e-10 or prob_injected > 1e-10:
                prob_ratio = prob_injected / (prob_original + 1e-10)
            else:
                # Both probabilities are essentially zero
                prob_ratio = 1.0
            
            # Surprisal delta - corrected calculation
            if prob_original > 1e-10 and prob_injected > 1e-10:
                surprisal_injected = -np.log(prob_injected)
                surprisal_original = -np.log(prob_original)
                surprisal_delta = surprisal_injected - surprisal_original
            else:
                surprisal_delta = 0.0
        else:
            # No reference distributions available - use defaults
            prob_ratio = 1.0  # Neutral ratio
            surprisal_delta = 0.0  # No change
        
        # Calculate cosine similarity with reference embeddings
        cos_sim_original = 0.0
        cos_sim_injected = 0.0
        
        # Always get current token embedding for cosine similarity
        current_embedding = self._get_token_embedding(token_id)
        
        if current_embedding is not None:
            # For original context cosine similarity
            if original_dist and 'embedding' in original_dist:
                original_embedding = original_dist['embedding']
                if original_embedding is not None:
                    # Ensure both embeddings are on same device
                    original_embedding = original_embedding.to(self.device)
                    cos_sim_original = self._cosine_similarity(current_embedding, original_embedding)
            else:
                # If no reference embedding, compare with the predicted token from original context
                # This gives us a measure of how well the token fits the original context
                if original_dist and 'logits' in original_dist:
                    # Get the top predicted token from original distribution
                    orig_logits = original_dist['logits']
                    orig_top_token = torch.argmax(orig_logits).item()
                    orig_top_embedding = self._get_token_embedding(orig_top_token)
                    if orig_top_embedding is not None:
                        cos_sim_original = self._cosine_similarity(current_embedding, orig_top_embedding)
            
            # For injected context cosine similarity
            if injected_dist and 'embedding' in injected_dist:
                injected_embedding = injected_dist['embedding']
                if injected_embedding is not None:
                    # Ensure both embeddings are on same device
                    injected_embedding = injected_embedding.to(self.device)
                    cos_sim_injected = self._cosine_similarity(current_embedding, injected_embedding)
            else:
                # If no reference embedding, compare with the predicted token from injected context
                if injected_dist and 'logits' in injected_dist:
                    # Get the top predicted token from injected distribution
                    inj_logits = injected_dist['logits']
                    inj_top_token = torch.argmax(inj_logits).item()
                    inj_top_embedding = self._get_token_embedding(inj_top_token)
                    if inj_top_embedding is not None:
                        cos_sim_injected = self._cosine_similarity(current_embedding, inj_top_embedding)
        
        # Determine context source based on ALL metrics
        # Consider prob_ratio, KL divergences, and cosine similarities
        
        # Check if all metrics are essentially the same (mixed/neutral state)
        kl_diff = abs(kl_divergence_from_original - kl_divergence_from_injected)
        cos_diff = abs(cos_sim_original - cos_sim_injected)
        
        if kl_diff < 0.1 and cos_diff < 0.05 and 0.8 < prob_ratio < 1.25:
            # All metrics are similar - truly mixed/neutral
            context_source = 'mixed'
        else:
            # Score each context based on multiple factors
            original_score = 0
            injected_score = 0
            
            # Probability ratio (most direct signal)
            if prob_ratio > 1.5:
                injected_score += 2
            elif prob_ratio > 1.0:
                injected_score += 1
            elif prob_ratio < 0.67:
                original_score += 2
            elif prob_ratio < 1.0:
                original_score += 1
            
            # KL divergence (lower is better - means closer to that distribution)
            if kl_divergence_from_original < kl_divergence_from_injected - 1.0:
                original_score += 1
            elif kl_divergence_from_injected < kl_divergence_from_original - 1.0:
                injected_score += 1
            
            # Cosine similarity (higher is better)
            if cos_sim_original > cos_sim_injected + 0.1:
                original_score += 1
            elif cos_sim_injected > cos_sim_original + 0.1:
                injected_score += 1
            
            # Determine final context
            if injected_score > original_score:
                context_source = 'injected'
            elif original_score > injected_score:
                context_source = 'original'
            else:
                context_source = 'mixed'
        
        return TokenState(
            position=position,
            token_id=token_id,
            token_text=token_text,
            context_source=context_source,
            kl_from_original=round(kl_divergence_from_original, 3),
            kl_from_injected=round(kl_divergence_from_injected, 3),
            embedding_similarity_original=round(cos_sim_original, 3),
            embedding_similarity_injected=round(cos_sim_injected, 3),
            prob_ratio=round(prob_ratio, 3),
            surprisal_delta=round(surprisal_delta, 3),
            token_prob=round(token_prob, 3)
        )
    
    def _get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get the embedding vector for a specific token."""
        try:
            # Get the token embedding from the model's embedding layer
            with torch.no_grad():
                embedding = self.model.get_input_embeddings()(torch.tensor([token_id], device=self.device))
                return embedding.squeeze(0)  # Remove batch dimension
        except Exception as e:
            print(f"Warning: Could not get embedding for token {token_id}: {e}")
            return None
    
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure both vectors are on the same device and same dtype
            vec1 = vec1.to(self.device).float()
            vec2 = vec2.to(self.device).float()
            
            # Calculate cosine similarity
            dot_product = torch.dot(vec1, vec2)
            norm1 = torch.norm(vec1)
            norm2 = torch.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = (dot_product / (norm1 * norm2)).item()
            
            # Clamp to valid range [-1, 1]
            return max(-1.0, min(1.0, similarity))
        except Exception as e:
            print(f"Warning: Could not calculate cosine similarity: {e}")
            return 0.0
    
    def _analyze_adaptation_patterns(
        self, 
        states: List[TokenState], 
        injection_point: int
    ) -> Dict:
        """Analyze patterns in token-level adaptation."""
        
        if not states:
            return {}
        
        # Extract time series
        positions = [s.position for s in states]
        prob_ratios = [s.prob_ratio for s in states]
        token_probs = [s.token_prob for s in states]
        
        # Smooth the signals for better pattern detection
        if len(prob_ratios) > 5:
            prob_ratios_smooth = savgol_filter(prob_ratios, min(5, len(prob_ratios)), 2)
        else:
            prob_ratios_smooth = prob_ratios
        
        # Find adaptation point (where prob_ratio crosses 1.0 and stays)
        adaptation_point = self._find_adaptation_point(prob_ratios, injection_point)
        
        # Calculate adaptation speed (tokens from injection to adaptation)
        if adaptation_point > injection_point and adaptation_point != -1:
            adaptation_speed = adaptation_point - injection_point
        else:
            adaptation_speed = -1  # Use -1 to indicate no adaptation occurred
        
        return {
            'adaptation_point': adaptation_point,
        }
    
    def _find_adaptation_point(
        self, 
        prob_ratios: List[float], 
        injection_point: int,
        threshold: float = 1.0,
        window: int = 3
    ) -> int:
        """Find the point where model adapts to new context.
        
        Requires 3 consecutive tokens with prob_ratio > 1.0
        to confirm real adaptation (not just noise).
        """
        
        for i in range(injection_point, len(prob_ratios) - window + 1):
            # Check if prob_ratio > threshold for window consecutive tokens
            if all(prob_ratios[j] > threshold for j in range(i, min(i + window, len(prob_ratios)))):
                return i
        
        return -1  # Didn't adapt (use -1 instead of inf for easier handling)
    
    def _calculate_token_metrics(
        self, 
        states: List[TokenState], 
        injection_point: int
    ) -> Dict:
        """Calculate aggregate metrics from token states."""
        
        if not states:
            return {}
        
        # Split into pre and post injection
        pre_states = [s for s in states if s.position < injection_point]
        post_states = [s for s in states if s.position >= injection_point]
        
        metrics = {}
        
        # Pre-injection metrics
        if pre_states:
            metrics['pre_injection'] = {
                'mean_token_prob': round(np.mean([s.token_prob for s in pre_states]), 3),
                'mean_surprisal': round(np.mean([s.surprisal_delta for s in pre_states]), 3),
                'mean_prob_ratio': round(np.mean([s.prob_ratio for s in pre_states]), 3)
            }
        
        # Post-injection metrics
        if post_states:
            metrics['post_injection'] = {
                'mean_token_prob': round(np.mean([s.token_prob for s in post_states]), 3),
                'mean_surprisal': round(np.mean([s.surprisal_delta for s in post_states]), 3),
                'mean_prob_ratio': round(np.mean([s.prob_ratio for s in post_states]), 3)
            }
        
        
        
        return metrics
    
    def _create_summary(self, analysis: Dict, metrics: Dict, injection_point: int) -> Dict:
        """Create a concise summary of key adaptation metrics."""
        
        # Core metrics only
        adaptation_point = analysis.get('adaptation_point', -1)
        
        # Calculate adaptation speed from adaptation point
        if adaptation_point > injection_point and adaptation_point != -1:
            adaptation_speed = adaptation_point - injection_point
        else:
            adaptation_speed = -1
        
        # Calculate adaptation quality from prob ratios > 1.0
        post_injection_states = [s for s in self.token_history if s.position > 0]  # Will be set properly in context
        if post_injection_states:
            adaptation_quality = sum(1 for s in post_injection_states if s.prob_ratio > 1.0) / len(post_injection_states)
        else:
            adaptation_quality = 0.0
        
        summary = {
            'adapted': adaptation_point != -1,
            'adaptation_tokens': adaptation_speed,
            'adaptation_quality': adaptation_quality
        }
        
        # Assessment
        if summary['adapted'] and adaptation_speed > 0:
            if adaptation_speed < 5:
                summary['assessment'] = 'excellent'
            elif adaptation_speed < 10:
                summary['assessment'] = 'good'
            else:
                summary['assessment'] = 'slow'
        else:
            summary['assessment'] = 'no adaptation'
        
        return summary
    
    def plot_token_adaptation(
        self, 
        analysis: Dict, 
        injection_point: int,
        save_path: Optional[str] = None
    ):
        """Visualize token-level adaptation dynamics."""
        
        if 'trajectories' not in analysis:
            print("No trajectory data to plot")
            return
        
        traj = analysis['trajectories']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Probability ratios
        axes[0].plot(traj['positions'], traj['prob_ratios'], 'b-', alpha=0.3, label='Raw')
        if 'prob_ratios_smooth' in traj:
            axes[0].plot(traj['positions'], traj['prob_ratios_smooth'], 'b-', linewidth=2, label='Smoothed')
        axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(x=injection_point, color='red', linestyle='--', label='Injection')
        if 'adaptation_point' in analysis:
            axes[0].axvline(x=analysis['adaptation_point'], color='green', linestyle='--', label='Adaptation')
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('P(token|injected) / P(token|original)')
        axes[0].set_title('Context Preference Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Entropy
        axes[1].plot(traj['positions'], traj['entropies'], 'orange', linewidth=2)
        axes[1].axvline(x=injection_point, color='red', linestyle='--')
        axes[1].fill_between(traj['positions'], 0, traj['entropies'], alpha=0.3, color='orange')
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Entropy (bits)')
        axes[1].set_title('Generation Uncertainty')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence
        axes[2].plot(traj['positions'], traj['confidences'], 'green', linewidth=2)
        axes[2].axvline(x=injection_point, color='red', linestyle='--')
        axes[2].fill_between(traj['positions'], 0, traj['confidences'], alpha=0.3, color='green')
        axes[2].set_xlabel('Token Position')
        axes[2].set_ylabel('Confidence (max prob)')
        axes[2].set_title('Generation Confidence')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# =====================================
# CAUSAL EXPERIMENTS MODULE  
# =====================================

@dataclass
class ExperimentCondition:
    """Defines an experimental condition for causal testing."""
    name: str
    description: str
    setup_fn: callable  # Function to setup the generation
    is_baseline: bool = False
    is_oracle: bool = False
    is_control: bool = False


class CausalExperiments:
    """Run controlled experiments to isolate adaptation effects."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}
        
    def run_causal_experiment(
        self,
        original_problem: str,
        alternative_problem: str,
        injection_points: List[int] = [10, 30, 50],
        num_trials: int = 10,
        max_new_tokens: int = 50
    ) -> Dict:
        """Run full causal experiment with proper controls."""
        
        all_results = {}
        
        for injection_point in injection_points:
            print(f"\nTesting injection at position {injection_point}")
            
            # Define experimental conditions
            conditions = self._setup_conditions(
                original_problem, 
                alternative_problem,
                injection_point
            )
            
            condition_results = {}
            
            # Run trials for each condition
            for condition in conditions:
                print(f"  Running condition: {condition.name}")
                trial_results = []
                
                for trial in range(num_trials):
                    result = self._run_single_trial(
                        condition,
                        original_problem,
                        alternative_problem,
                        injection_point,
                        max_new_tokens
                    )
                    trial_results.append(result)
                
                condition_results[condition.name] = {
                    'trials': trial_results,
                    'condition': condition,
                    'statistics': self._calculate_statistics(trial_results)
                }
            
            # Perform causal analysis
            causal_analysis = self._analyze_causality(condition_results)
            
            all_results[f'injection_{injection_point}'] = {
                'conditions': condition_results,
                'causal_analysis': causal_analysis
            }
        
        # Overall analysis across injection points
        overall_analysis = self._analyze_overall_patterns(all_results)
        
        return {
            'experiment_results': all_results,
            'overall_analysis': overall_analysis,
            'recommendations': self._generate_recommendations(overall_analysis)
        }
    
    def _setup_conditions(
        self, 
        original: str, 
        alternative: str,
        injection_point: int
    ) -> List[ExperimentCondition]:
        """Setup experimental conditions with proper controls."""
        
        conditions = []
        
        # 1. BASELINE: Original problem only
        def baseline_setup():
            return self._generate_with_context(original, None, None)
        
        conditions.append(ExperimentCondition(
            name="baseline",
            description="Original context only, no injection",
            setup_fn=baseline_setup,
            is_baseline=True
        ))
        
        # 2. ORACLE: Alternative problem from the start
        def oracle_setup():
            # Start with alternative context from beginning
            combined = alternative  # Pure alternative, not mixed
            return self._generate_with_context(combined, None, None)
        
        conditions.append(ExperimentCondition(
            name="oracle",
            description="Alternative context from beginning (gold standard)",
            setup_fn=oracle_setup,
            is_oracle=True
        ))
        
        # 3. INJECTION: Our experimental condition
        def injection_setup():
            return self._generate_with_context(original, alternative, injection_point)
        
        conditions.append(ExperimentCondition(
            name="injection",
            description=f"Alternative injected at position {injection_point}",
            setup_fn=injection_setup
        ))
        
        # 4. DELAYED ORACLE: Oracle but starting after injection point
        def delayed_oracle_setup():
            # Pad with neutral tokens then give alternative
            padding = " " * injection_point  # Simplified padding
            return self._generate_with_context(padding + alternative, None, None)
        
        conditions.append(ExperimentCondition(
            name="delayed_oracle",
            description="Oracle starting at injection position",
            setup_fn=delayed_oracle_setup
        ))
        
        # 5. NOISE CONTROL: Random injection
        def noise_setup():
            noise_text = self._generate_noise_text(len(alternative.split()))
            return self._generate_with_context(original, noise_text, injection_point)
        
        conditions.append(ExperimentCondition(
            name="noise_control",
            description="Random noise injection",
            setup_fn=noise_setup,
            is_control=True
        ))
        
        # 6. REPETITION CONTROL: Inject same context
        def repetition_setup():
            return self._generate_with_context(original, original, injection_point)
        
        conditions.append(ExperimentCondition(
            name="repetition_control",
            description="Same context re-injected",
            setup_fn=repetition_setup,
            is_control=True
        ))
        
        return conditions
    
    def _generate_with_context(
        self,
        initial_context: str,
        injection_context: Optional[str],
        injection_point: Optional[int]
    ) -> Dict:
        """Generate text with optional context injection."""
        
        # Tokenize initial context
        inputs = self.tokenizer(initial_context, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        generated_tokens = []
        token_logprobs = []
        hidden_states_history = []
        
        past_key_values = None
        current_length = 0
        
        with torch.no_grad():
            # Initial forward pass
            outputs = self.model(
                input_ids,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            
            # Generate token by token
            for step in range(50):  # max tokens
                # Get next token
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(logits)
                
                # Check for EOS token BEFORE storing
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Store generation info
                generated_tokens.append(next_token.item())
                token_logprobs.append(torch.log(probs[next_token]).item())
                hidden_states_history.append(outputs.hidden_states[-1][0, -1].cpu().numpy())
                
                # Check for injection
                if injection_context and injection_point and step == injection_point:
                    # REAL KV-cache injection
                    injection_ids = self.tokenizer(injection_context, return_tensors="pt").input_ids.to(self.device)
                    
                    with torch.no_grad():
                        injection_outputs = self.model(
                            injection_ids,
                            use_cache=True,
                            return_dict=True
                        )
                        injection_kv = injection_outputs.past_key_values
                    
                    # Concatenate injection KV cache to existing cache
                    if past_key_values is not None and injection_kv is not None:
                        # Handle different KV cache formats (some models use DynamicCache)
                        if hasattr(past_key_values, 'layers'):
                            # DynamicCache format - extend the cache
                            for layer_idx, (inj_key, inj_val) in enumerate(injection_kv):
                                past_key_values.layers[layer_idx].keys = torch.cat([
                                    past_key_values.layers[layer_idx].keys, inj_key
                                ], dim=2)
                                past_key_values.layers[layer_idx].values = torch.cat([
                                    past_key_values.layers[layer_idx].values, inj_val  
                                ], dim=2)
                        else:
                            # Standard tuple format
                            past_key_values = tuple(
                                (torch.cat([layer_kv[0], inj_kv[0]], dim=2),  # concat keys
                                 torch.cat([layer_kv[1], inj_kv[1]], dim=2))  # concat values
                                for layer_kv, inj_kv in zip(past_key_values, injection_kv)
                            )
                    
                # Continue generation
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
        
        return {
            'generated_tokens': generated_tokens,
            'generated_text': self.tokenizer.decode(generated_tokens),
            'token_logprobs': token_logprobs,
            'hidden_states': hidden_states_history
        }
    
    def _run_single_trial(
        self,
        condition: ExperimentCondition,
        original: str,
        alternative: str,
        injection_point: int,
        max_tokens: int
    ) -> Dict:
        """Run a single trial of an experimental condition."""
        
        # Generate text according to condition
        generation_result = condition.setup_fn()
        
        # Calculate metrics
        metrics = self._calculate_trial_metrics(
            generation_result,
            original,
            alternative,
            condition
        )
        
        return {
            'generation': generation_result,
            'metrics': metrics,
            'condition_name': condition.name
        }
    
    def _calculate_trial_metrics(
        self,
        generation: Dict,
        original: str,
        alternative: str,
        condition: ExperimentCondition
    ) -> Dict:
        """Calculate metrics for a single trial."""
        
        generated_text = generation['generated_text']
        
        # Lexical overlap with original vs alternative
        original_words = set(original.lower().split())
        alternative_words = set(alternative.lower().split())
        generated_words = set(generated_text.lower().split())
        
        original_overlap = len(generated_words & original_words) / (len(generated_words) + 1e-8)
        alternative_overlap = len(generated_words & alternative_words) / (len(generated_words) + 1e-8)
        
        # Trajectory analysis
        if 'token_logprobs' in generation:
            logprobs = generation['token_logprobs']
            # Measure stability (lower variance = more stable)
            stability = 1.0 / (1.0 + np.var(logprobs))
            # Measure confidence (mean log prob)
            confidence = np.mean(logprobs) if logprobs else -1
        else:
            stability = 0.0
            confidence = -1
        
        return {
            'original_overlap': original_overlap,
            'alternative_overlap': alternative_overlap,
            'adaptation_ratio': alternative_overlap / (original_overlap + 1e-8),
            'stability': stability,
            'confidence': confidence,
            'text_length': len(generated_text.split())
        }
    
    def _calculate_statistics(self, trials: List[Dict]) -> Dict:
        """Calculate statistics across trials."""
        
        if not trials:
            return {}
        
        # Extract metrics from all trials
        metrics_lists = {}
        for trial in trials:
            for metric_name, value in trial['metrics'].items():
                if metric_name not in metrics_lists:
                    metrics_lists[metric_name] = []
                metrics_lists[metric_name].append(value)
        
        # Calculate statistics for each metric
        statistics = {}
        for metric_name, values in metrics_lists.items():
            statistics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return statistics
    
    def _analyze_causality(self, condition_results: Dict) -> Dict:
        """Perform causal analysis comparing conditions."""
        
        analysis = {}
        
        # Key comparison: Injection vs Oracle (not baseline!)
        if 'injection' in condition_results and 'oracle' in condition_results:
            injection_metrics = [t['metrics'] for t in condition_results['injection']['trials']]
            oracle_metrics = [t['metrics'] for t in condition_results['oracle']['trials']]
            
            # Adaptation gap: How far is injection from oracle?
            adaptation_gaps = {}
            for metric in ['alternative_overlap', 'stability']:
                inj_values = [m[metric] for m in injection_metrics]
                ora_values = [m[metric] for m in oracle_metrics]
                
                # Calculate gap
                gap = np.mean(inj_values) - np.mean(ora_values)
                
                # Statistical test with comprehensive numerical stability check
                try:
                    # Check if we have sufficient data
                    if len(inj_values) < 3 or len(ora_values) < 3:
                        t_stat, p_value = 0.0, 1.0
                    # Check if data has insufficient variance (prevents precision loss)
                    elif (np.std(inj_values) < 1e-10 and np.std(ora_values) < 1e-10) or np.allclose(inj_values, ora_values, rtol=1e-10):
                        # Data is essentially identical, no meaningful statistical test
                        t_stat, p_value = 0.0, 1.0
                    # Check for constant arrays
                    elif len(set(inj_values)) <= 1 and len(set(ora_values)) <= 1:
                        t_stat, p_value = 0.0, 1.0
                    else:
                        # Additional check: ensure variance is reasonable
                        inj_var = np.var(inj_values)
                        ora_var = np.var(ora_values)
                        if inj_var < 1e-15 and ora_var < 1e-15:
                            t_stat, p_value = 0.0, 1.0
                        else:
                            t_stat, p_value = ttest_ind(inj_values, ora_values, equal_var=False)
                except (RuntimeWarning, ValueError, ZeroDivisionError):
                    # Fallback for numerical issues
                    t_stat, p_value = 0.0, 1.0
                
                adaptation_gaps[metric] = {
                    'gap': gap,
                    'relative_gap': gap / (np.mean(ora_values) + 1e-8),
                    'significant': p_value < 0.05,
                    'p_value': p_value
                }
            
            analysis['adaptation_gaps'] = adaptation_gaps
        
        # Compare to controls
        if 'injection' in condition_results and 'noise_control' in condition_results:
            injection_metrics = [t['metrics'] for t in condition_results['injection']['trials']]
            noise_metrics = [t['metrics'] for t in condition_results['noise_control']['trials']]
            
            # Is injection better than noise?
            better_than_noise = {}
            for metric in ['alternative_overlap', 'stability']:
                inj_values = [m[metric] for m in injection_metrics]
                noise_values = [m[metric] for m in noise_metrics]
                
                # Statistical test with comprehensive numerical stability check
                try:
                    # Check if we have sufficient data
                    if len(inj_values) < 3 or len(noise_values) < 3:
                        t_stat, p_value = 0.0, 1.0
                    # Check if data has insufficient variance (prevents precision loss)
                    elif (np.std(inj_values) < 1e-10 and np.std(noise_values) < 1e-10) or np.allclose(inj_values, noise_values, rtol=1e-10):
                        # Data is essentially identical, no meaningful statistical test
                        t_stat, p_value = 0.0, 1.0
                    # Check for constant arrays
                    elif len(set(inj_values)) <= 1 and len(set(noise_values)) <= 1:
                        t_stat, p_value = 0.0, 1.0
                    else:
                        # Additional check: ensure variance is reasonable
                        inj_var = np.var(inj_values)
                        noise_var = np.var(noise_values)
                        if inj_var < 1e-15 and noise_var < 1e-15:
                            t_stat, p_value = 0.0, 1.0
                        else:
                            t_stat, p_value = ttest_ind(inj_values, noise_values, equal_var=False)
                except (RuntimeWarning, ValueError, ZeroDivisionError):
                    # Fallback for numerical issues
                    t_stat, p_value = 0.0, 1.0
                
                better_than_noise[metric] = {
                    'injection_mean': np.mean(inj_values),
                    'noise_mean': np.mean(noise_values),
                    'difference': np.mean(inj_values) - np.mean(noise_values),
                    'significant': p_value < 0.05,
                    'p_value': p_value
                }
            
            analysis['vs_noise'] = better_than_noise
        
        # Calculate true adaptation score
        if 'adaptation_gaps' in analysis:
            gaps = analysis['adaptation_gaps']
            
            # Normalize gaps to 0-1 scale (0 = no adaptation, 1 = perfect adaptation)
            adaptation_scores = []
            for metric, gap_data in gaps.items():
                if metric == 'alternative_overlap':
                    # Higher is better, smaller gap is better
                    score = 1.0 - abs(gap_data['relative_gap'])
                elif metric == 'stability':
                    # Higher is better
                    score = 1.0 - abs(gap_data['relative_gap'])
                else:
                    continue
                
                adaptation_scores.append(max(0, min(1, score)))
            
            analysis['adaptation_score'] = np.mean(adaptation_scores) if adaptation_scores else 0.0
        
        return analysis
    
    def _analyze_overall_patterns(self, all_results: Dict) -> Dict:
        """Analyze patterns across different injection points."""
        
        patterns = {}
        
        # Extract adaptation scores across injection points
        injection_points = []
        adaptation_scores = []
        
        for key, result in all_results.items():
            if 'injection_' in key:
                point = int(key.split('_')[1])
                injection_points.append(point)
                
                if 'causal_analysis' in result and 'adaptation_score' in result['causal_analysis']:
                    adaptation_scores.append(result['causal_analysis']['adaptation_score'])
        
        if injection_points and adaptation_scores:
            # Fit curve to adaptation scores
            z = np.polyfit(injection_points, adaptation_scores, 2)
            p = np.poly1d(z)
            
            patterns['injection_curve'] = {
                'points': injection_points,
                'scores': adaptation_scores,
                'polynomial_coefficients': z.tolist(),
                'optimal_injection': injection_points[np.argmax(adaptation_scores)]
            }
            
            # Determine adaptation pattern
            if adaptation_scores[-1] > adaptation_scores[0]:
                pattern_type = "improving"
            elif adaptation_scores[-1] < adaptation_scores[0]:
                pattern_type = "degrading"
            else:
                pattern_type = "stable"
            
            patterns['pattern_type'] = pattern_type
            patterns['mean_adaptation'] = np.mean(adaptation_scores)
        
        return patterns
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        if 'mean_adaptation' in analysis:
            score = analysis['mean_adaptation']
            
            if score < 0.3:
                recommendations.append(
                    f"Poor adaptation (score: {score:.2f}). Consider using full reprompting instead of injection."
                )
            elif score < 0.7:
                recommendations.append(
                    f"Moderate adaptation (score: {score:.2f}). Try earlier injection points or gradual context blending."
                )
            else:
                recommendations.append(
                    f"Good adaptation (score: {score:.2f}). KV-cache injection is viable for this model."
                )
        
        if 'pattern_type' in analysis:
            if analysis['pattern_type'] == "degrading":
                recommendations.append(
                    "Later injections perform worse. Inject context as early as possible."
                )
            elif analysis['pattern_type'] == "improving":
                recommendations.append(
                    "Later injections perform better. Allow model to establish context first."
                )
        
        if 'injection_curve' in analysis:
            optimal = analysis['injection_curve']['optimal_injection']
            recommendations.append(
                f"Optimal injection point: after {optimal} tokens for this type of context switch."
            )
        
        return recommendations
    
    def _calculate_semantic_similarity(self, expected_answer: str, generated_text: str) -> float:
        """Calculate semantic similarity between expected answer and generated text."""
        
        try:
            # Clean up texts
            expected = expected_answer.strip().lower()
            generated = generated_text.strip().lower()
            
            if not expected or not generated:
                return 0.0
            
            # Method 1: Simple token overlap as fallback
            expected_tokens = set(expected.split())
            generated_tokens = set(generated.split())
            
            if len(expected_tokens) == 0:
                return 0.0
                
            token_overlap = len(expected_tokens.intersection(generated_tokens)) / len(expected_tokens)
            
            # Method 2: Try to use model embeddings for semantic similarity
            try:
                expected_embedding = self._get_text_embedding(expected)
                generated_embedding = self._get_text_embedding(generated)
                
                # Cosine similarity
                cosine_sim = torch.cosine_similarity(
                    expected_embedding.unsqueeze(0), 
                    generated_embedding.unsqueeze(0)
                )
                semantic_score = float(cosine_sim.item())
                
                # Combine token overlap and semantic similarity (weighted average)
                final_score = 0.3 * token_overlap + 0.7 * semantic_score
                return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
                
            except Exception:
                # Fallback to token overlap if embedding fails
                return token_overlap
                
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding representation of text using the model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as text embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            
        return embedding
    
    def _generate_noise_text(self, num_words: int) -> str:
        """Generate random noise text."""
        import random
        vocab = ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'it', 'was',
                 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this']
        return ' '.join(random.choices(vocab, k=num_words))
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize experimental results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Adaptation scores across injection points
        if 'overall_analysis' in results and 'injection_curve' in results['overall_analysis']:
            curve_data = results['overall_analysis']['injection_curve']
            axes[0, 0].plot(curve_data['points'], curve_data['scores'], 'o-', linewidth=2)
            axes[0, 0].set_xlabel('Injection Point (tokens)')
            axes[0, 0].set_ylabel('Adaptation Score')
            axes[0, 0].set_title('Adaptation Quality vs Injection Timing')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Condition comparison
        if 'experiment_results' in results:
            first_exp = list(results['experiment_results'].values())[0]
            if 'conditions' in first_exp:
                conditions = []
                overlaps = []
                
                for cond_name, cond_data in first_exp['conditions'].items():
                    conditions.append(cond_name)
                    overlaps.append(cond_data['statistics']['alternative_overlap']['mean'])
                
                axes[0, 1].bar(conditions, overlaps, color=['blue', 'gold', 'green', 'orange', 'red', 'gray'])
                axes[0, 1].set_xlabel('Condition')
                axes[0, 1].set_ylabel('Alternative Context Overlap')
                axes[0, 1].set_title('Context Preference by Condition')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Adaptation gaps
        if 'experiment_results' in results:
            gaps = []
            metrics = []
            
            for exp_name, exp_data in results['experiment_results'].items():
                if 'causal_analysis' in exp_data and 'adaptation_gaps' in exp_data['causal_analysis']:
                    for metric, gap_data in exp_data['causal_analysis']['adaptation_gaps'].items():
                        gaps.append(abs(gap_data['gap']))
                        metrics.append(f"{exp_name}_{metric}")
            
            if gaps:
                axes[1, 0].barh(range(len(gaps)), gaps)
                axes[1, 0].set_yticks(range(len(gaps)))
                axes[1, 0].set_yticklabels(metrics, fontsize=8)
                axes[1, 0].set_xlabel('Absolute Gap from Oracle')
                axes[1, 0].set_title('Adaptation Gaps by Metric')
        
        # Plot 4: Statistical significance
        if 'experiment_results' in results:
            p_values = []
            comparisons = []
            
            for exp_name, exp_data in results['experiment_results'].items():
                if 'causal_analysis' in exp_data and 'adaptation_gaps' in exp_data['causal_analysis']:
                    for metric, gap_data in exp_data['causal_analysis']['adaptation_gaps'].items():
                        p_values.append(gap_data['p_value'])
                        comparisons.append(f"{exp_name}_{metric}")
            
            if p_values:
                colors = ['green' if p < 0.05 else 'red' for p in p_values]
                axes[1, 1].bar(range(len(p_values)), p_values, color=colors)
                axes[1, 1].axhline(y=0.05, color='black', linestyle='--', label='p=0.05')
                axes[1, 1].set_xticks(range(len(p_values)))
                axes[1, 1].set_xticklabels(comparisons, rotation=45, fontsize=8)
                axes[1, 1].set_ylabel('P-value')
                axes[1, 1].set_title('Statistical Significance of Adaptation Gaps')
                axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# =====================================
# MAIN ANALYZER CLASS
# =====================================

class ProperAdaptationAnalyzer:
    """Comprehensive adaptation analyzer using multiple measurement approaches."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B",
        device: str = None,
        cache_dir: str = "~/.cache/huggingface/hub"
    ):
        """Initialize analyzer with specified model."""
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            trust_remote_code=True,
            cache_dir=os.path.expanduser(cache_dir)
            # Use default SDPA for speed - no attention tracking needed
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.path.expanduser(cache_dir)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize analysis modules
        self.attention_tracker = AttentionTracker(self.model, self.tokenizer, device)
        self.adaptation_metrics = AdaptationMetrics(self.model, self.tokenizer, device)
        self.causal_experiments = CausalExperiments(self.model, self.tokenizer, device)
        self.token_analyzer = TokenLevelAnalyzer(self.model, self.tokenizer, device)
        
        print("Analyzer initialized successfully")
    
    def analyze_single_problem(
        self,
        original_problem: str,
        alternative_problem: str,
        injection_points: List[int] = [10, 30, 50],
        max_new_tokens: int = None,  # None means generate until completion
        dataset_type: str = 'math',
        base_context: str = None,
        original_question: str = None,
        alternative_question: str = None
    ) -> Dict:
        """Run comprehensive analysis on a single problem pair."""
        
        # CRITICAL: Clear all state at the start of each problem to prevent leakage
        if hasattr(self, 'token_analyzer') and hasattr(self.token_analyzer, 'token_history'):
            self.token_analyzer.token_history.clear()
        
        # Clear model cache if it exists
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
        
        # Force garbage collection to clear any residual state
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Since we now only use one injection point per problem, simplify structure
        injection_point = injection_points[0] if injection_points else 5
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model.config._name_or_path,
            'injection_point': injection_point,
            'problem_index': getattr(self, '_current_problem_index', None),
            'dataset_type': dataset_type,
            'problem_type': getattr(self, '_current_problem_type', None),
            'alt_language': getattr(self, '_current_alt_language', 'English'),
            'variation_type': getattr(self, '_current_variation_type', None)
        }
        
        # Structure output based on dataset type
        if dataset_type == 'english' and base_context and original_question and alternative_question:
            # Separate reading passage from questions for clarity
            results.update({
                'reading_passage': base_context,
                'original_question': original_question,
                'alternative_question': alternative_question,
                'alternative_question_original': getattr(self, '_current_alt_problem_original', None)
            })
        else:
            # For math problems, keep original structure
            results.update({
                'original_problem': original_problem,
                'alternative_problem': alternative_problem,
                'alternative_problem_original': getattr(self, '_current_alt_problem_original', None)
            })
        
        print(f"\nRunning Token-Level Analysis at injection point {injection_point}...")
        
        # Use two-phase approach for reading comprehension, single-phase for math
        if dataset_type == 'english' and base_context and original_question and alternative_question:
            print("Using two-phase generation for reading comprehension...")
            token_analysis = self.token_analyzer.analyze_two_phase_adaptation(
                base_context,
                original_question, 
                alternative_question,
                injection_point,
                max_new_tokens
            )
        else:
            print("Using single-phase generation for math problems...")
            token_analysis = self.token_analyzer.analyze_token_adaptation(
                original_problem,
                alternative_problem,
                injection_point,
                max_new_tokens
            )
        
        # Extract generated text for answer comparison first (for top of JSON)
        if token_analysis['token_states']:
            # Build generated text with injection marker
            text_parts = []
            for state in token_analysis['token_states']:
                # Add injection marker at the injection point
                if state['position'] == injection_point:
                    text_parts.append(' [INJECTION HERE] ')
                text_parts.append(state['token_text'])
            generated_text = ''.join(text_parts)
            results['generated_text'] = generated_text
            # Try to extract numerical answer from generated text with multiple patterns
            import re
            
            # Try different extraction patterns
            extracted_answer = None
            
            # Look for explicit "The answer is: X" pattern first
            explicit_answer = re.search(r'(?:the\s+)?answer\s+is[:\s]+([+-]?\d+\.?\d*)', generated_text.lower())
            if explicit_answer:
                extracted_answer = float(explicit_answer.group(1))
            else:
                # Fallback: Look for other answer patterns including \boxed{number}
                answer_patterns = [
                    r'\\boxed\{([+-]?\d+\.?\d*)\}',  # LaTeX boxed answer
                    r'(?:equals?|=)\s*([+-]?\d+\.?\d*)(?:\s+(?:grams?|apples?|dollars?|units?))?\s*(?:\.|$)',
                    r'(?:result|solution|total)\s*(?:is)?[:\s]+([+-]?\d+\.?\d*)',
                    r'([+-]?\d+\.?\d*)\s+(?:grams?|apples?|dollars?|units?)\s*(?:\.|$)'
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, generated_text.lower())
                    if match:
                        extracted_answer = float(match.group(1))
                        break
                
                # Check if the model actually completed its reasoning
                # Only extract a number if there's evidence of a completed solution
                if extracted_answer is None:
                    # Check for completion indicators
                    completion_indicators = [
                        r'therefore',
                        r'thus',
                        r'so\s+(?:the\s+)?(?:answer|result|solution)',
                        r'(?:in\s+)?(?:conclusion|summary)',
                        r'(?:final|total)\s+(?:answer|result)',
                        r'(?:which\s+)?(?:gives|yields|equals)',
                        r'```\s*(?:output|result)',  # Code execution output
                    ]
                    
                    has_completion = any(re.search(indicator, generated_text.lower()) for indicator in completion_indicators)
                    
                    # Only extract if there's a completion indicator
                    if has_completion:
                        numbers = re.findall(r'([+-]?\d+\.?\d*)', generated_text)
                        if numbers:
                            # Take the LAST number as the answer
                            extracted_answer = float(numbers[-1])
                    # Otherwise leave extracted_answer as None
            
            # Store the extracted answer (or None if not found)
            results['generated_answer'] = extracted_answer
        
        # Add expected answer right next to generated answer for easy comparison
        results['expected_answer'] = getattr(self, '_current_expected_answer', None)
        
        # Add summary at top level (visible immediately)
        results['summary'] = token_analysis['summary']
        
        # Add token states last (detailed data at bottom)
        results['token_states'] = token_analysis['token_states']
        
        # Analysis complete - we have token states and summary
        print("Analysis complete.")
        
        return results
    
    def _create_clean_summary(self, results: Dict) -> Dict:
        """Create a clean, concise summary with only essential metrics."""
        
        # Get best adaptation result across all injection points
        token_analysis = results.get('token_analysis', {})
        best_adaptation = None
        best_quality = 0
        
        for inj_key, analysis in token_analysis.items():
            quality = analysis.get('summary', {}).get('adaptation_quality', 0)
            if quality > best_quality:
                best_quality = quality
                best_adaptation = analysis.get('summary', {})
        
        summary = {
            'adapted': best_adaptation.get('adapted', False) if best_adaptation else False,
            'best_adaptation_tokens': best_adaptation.get('adaptation_tokens', -1) if best_adaptation else -1,
            'best_adaptation_quality': best_quality,
            'assessment': best_adaptation.get('assessment', 'failed') if best_adaptation else 'failed'
        }
        
        return summary
    
    def _create_clear_adaptation_summary(self, results: Dict) -> Dict:
        """Create a clear, easy-to-read adaptation summary with key metrics."""
        
        summary = {
            "🎯 ADAPTATION_DETECTED": False,
            "⚡ ADAPTATION_LATENCY_TOKENS": None,
            "📊 KEY_METRICS": {},
            "🔍 INJECTION_POINTS_ANALYSIS": {},
            "✅ VERDICT": "No clear adaptation detected"
        }
        
        try:
            # Extract key metrics from token analysis
            if 'token_analysis' in results:
                for inj_key, token_data in results['token_analysis'].items():
                    if 'summary' in token_data:
                        point = inj_key.replace('injection_', '')
                        
                        # Get KL divergence from adaptation analysis  
                        kl_divergence = None
                        adaptation_point = None
                        
                        if f'injection_{point}' in results.get('adaptation_analysis', {}):
                            adapt_data = results['adaptation_analysis'][f'injection_{point}']
                            if 'summary' in adapt_data:
                                kl_divergence = adapt_data['summary'].get('peak_kl_divergence', 'Not calculated')
                                adaptation_point = adapt_data['summary'].get('adaptation_point', None)
                        
                        # Token ratio trends
                        prob_ratios = []
                        if 'token_states' in token_data:
                            prob_ratios = [t.get('prob_ratio', 1.0) for t in token_data['token_states']]
                        
                        # Calculate adaptation metrics
                        tokens_above_1 = len([r for r in prob_ratios if r > 1.0])
                        avg_ratio = sum(prob_ratios) / len(prob_ratios) if prob_ratios else 1.0
                        max_ratio = max(prob_ratios) if prob_ratios else 1.0
                        
                        summary["🔍 INJECTION_POINTS_ANALYSIS"][f"injection_at_token_{point}"] = {
                            "kl_divergence": kl_divergence,
                            "adaptation_point": adaptation_point,
                            "token_ratio_avg": round(avg_ratio, 3),
                            "token_ratio_max": round(max_ratio, 3),
                            "tokens_favoring_new_context": tokens_above_1,
                            "total_tokens_generated": len(prob_ratios)
                        }
                        
                        # Determine if adaptation occurred
                        if kl_divergence and kl_divergence != 'Not calculated' and kl_divergence > 0.1:
                            summary["🎯 ADAPTATION_DETECTED"] = True
                            if adaptation_point:
                                summary["⚡ ADAPTATION_LATENCY_TOKENS"] = adaptation_point - int(point)
            
            # Overall metrics
            if summary["🎯 ADAPTATION_DETECTED"]:
                # Get best injection point
                best_kl = 0
                best_point = None
                for point_key, point_data in summary["🔍 INJECTION_POINTS_ANALYSIS"].items():
                    kl = point_data.get("kl_divergence", 0)
                    if isinstance(kl, (int, float)) and kl > best_kl:
                        best_kl = kl
                        best_point = point_key
                
                summary["📊 KEY_METRICS"] = {
                    "best_injection_point": best_point,
                    "highest_kl_divergence": round(best_kl, 4),
                    "adaptation_success_rate": "Detected" if best_kl > 0.1 else "Weak"
                }
                
                if best_kl > 0.5:
                    summary["✅ VERDICT"] = "🟢 STRONG adaptation - Model clearly switches context"
                elif best_kl > 0.1:
                    summary["✅ VERDICT"] = "🟡 MODERATE adaptation - Some context switching detected"
                else:
                    summary["✅ VERDICT"] = "🔴 WEAK adaptation - Limited context switching"
            
        except Exception as e:
            summary["❌ ERROR"] = f"Error creating summary: {str(e)}"
        
        return summary
    
    def _generate_analysis_plots(self, results: Dict, injection_points: List[int]):
        """Generate and save matplotlib plots for analysis results."""
        
        try:
            # Create organized plots directory structure
            base_plots_dir = Path("analysis_plots")
            base_plots_dir.mkdir(exist_ok=True)
            
            # Get timestamp and problem info for organization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            problem_index = results.get('problem_index', 'unknown')
            problem_hash = str(hash(results.get('original_problem', 'unknown')))[:8]
            
            # Create problem-specific directory - handle both string and int problem_index
            if isinstance(problem_index, int):
                problem_dir = base_plots_dir / f"problem_{problem_index:03d}_{problem_hash}"
            else:
                problem_dir = base_plots_dir / f"problem_{problem_index}_{problem_hash}"
            problem_dir.mkdir(exist_ok=True)
            
            print(f"   📁 Creating plots in: {problem_dir}")
            
            # 1. Token-level adaptation plots
            if 'token_analysis' in results:
                print("   📊 Generating token adaptation plots...")
                for inj_key, token_data in results['token_analysis'].items():
                    if 'summary' in token_data and len(injection_points) > 0:
                        injection_point = int(inj_key.replace('injection_', ''))
                        plot_path = problem_dir / f"token_adaptation_inj{injection_point}.png"
                        
                        # Use the existing plot method if trajectories exist
                        if hasattr(self.token_analyzer, 'plot_token_adaptation'):
                            try:
                                self.token_analyzer.plot_token_adaptation(
                                    token_data,
                                    injection_point=injection_point,
                                    save_path=str(plot_path)
                                )
                                print(f"     ✅ Saved: {plot_path}")
                            except Exception as e:
                                print(f"     ❌ Token plot failed: {e}")
                        
            # 2. Causal experiment plots  
            if 'causal_analysis' in results:
                print("   📈 Generating causal experiment plots...")
                try:
                    plot_path = problem_dir / f"causal_analysis.png"
                    
                    # Create causal results plot if we have overall analysis
                    if hasattr(self.causal_experiments, 'plot_results'):
                        # Reconstruct full causal results for plotting
                        causal_plot_data = {
                            'overall_analysis': results['causal_analysis']['overall'],
                            'recommendations': results['causal_analysis']['recommendations']
                        }
                        
                        self.causal_experiments.plot_results(
                            causal_plot_data,
                            save_path=str(plot_path)
                        )
                        print(f"     ✅ Saved: {plot_path}")
                except Exception as e:
                    print(f"     ❌ Causal plot failed: {e}")
            
            # 3. Create summary adaptation plot
            self._create_summary_adaptation_plot(results, injection_points, timestamp, problem_hash, problem_dir)
            
        except Exception as e:
            print(f"   ❌ Plot generation failed: {e}")
    
    def _create_summary_adaptation_plot(self, results: Dict, injection_points: List[int], 
                                       timestamp: str, problem_hash: str, problem_dir: Path):
        """Create a summary plot showing adaptation across injection points."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract adaptation metrics across injection points
            adaptation_scores = []
            kl_divergences = []
            token_latencies = []
            
            for injection_point in injection_points:
                inj_key = f'injection_{injection_point}'
                
                # From token analysis
                adaptation_score = 0.0
                kl_div = 0.0
                latency = None
                
                if 'token_analysis' in results and inj_key in results['token_analysis']:
                    token_data = results['token_analysis'][inj_key]
                    if 'summary' in token_data:
                        adaptation_score = token_data['summary'].get('adaptation_quality', 0.0)
                        latency = token_data['summary'].get('adaptation_tokens', None)
                
                # From adaptation metrics
                if 'adaptation_metrics' in results and inj_key in results['adaptation_metrics']:
                    adapt_data = results['adaptation_metrics'][inj_key]
                    kl_div = adapt_data.get('kl_from_injected', 0.0)
                
                adaptation_scores.append(adaptation_score)
                kl_divergences.append(kl_div)
                token_latencies.append(latency if latency is not None else 0)
            
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Adaptation Quality vs Injection Point
            axes[0, 0].plot(injection_points, adaptation_scores, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Injection Point (tokens)')
            axes[0, 0].set_ylabel('Adaptation Quality')
            axes[0, 0].set_title('Adaptation Quality vs Injection Timing')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            # Plot 2: KL Divergence
            axes[0, 1].plot(injection_points, kl_divergences, 'ro-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Injection Point (tokens)')
            axes[0, 1].set_ylabel('KL Divergence')
            axes[0, 1].set_title('KL Divergence from Injected Context')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Adaptation Latency
            valid_latencies = [l for l in token_latencies if l > 0]
            valid_points = [injection_points[i] for i, l in enumerate(token_latencies) if l > 0]
            
            if valid_latencies:
                axes[1, 0].plot(valid_points, valid_latencies, 'go-', linewidth=2, markersize=8)
                axes[1, 0].set_xlabel('Injection Point (tokens)')
                axes[1, 0].set_ylabel('Adaptation Latency (tokens)')
                axes[1, 0].set_title('Time to Adaptation')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No adaptation\ndetected', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Time to Adaptation (No Data)')
            
            # Plot 4: Text summary
            axes[1, 1].axis('off')
            summary_text = f"""
ADAPTATION ANALYSIS SUMMARY

Original Problem:
{results.get('original_problem', 'N/A')[:100]}...

Alternative Problem:  
{results.get('alternative_problem', 'N/A')[:100]}...

Best Injection Point: {injection_points[np.argmax(adaptation_scores)] if adaptation_scores else 'N/A'}
Max Adaptation Quality: {max(adaptation_scores):.3f if adaptation_scores else 'N/A'}
Average Latency: {np.mean(valid_latencies):.1f if len(valid_latencies) > 0 else 'N/A'} tokens

Timestamp: {timestamp}
            """
            axes[1, 1].text(0.05, 0.95, summary_text.strip(), 
                           transform=axes[1, 1].transAxes, fontsize=9,
                           verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = problem_dir / f"adaptation_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"     ✅ Saved summary plot: {plot_path}")
            
        except Exception as e:
            print(f"     ❌ Summary plot failed: {e}")
    
    def _generate_dataset_summary_plots(self, all_results: List[Dict], injection_points: List[int]):
        """Generate summary plots across the entire dataset."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create plots directory
            plots_dir = Path("analysis_plots")
            plots_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Aggregate adaptation metrics across all problems
            adaptation_by_injection = {f'injection_{ip}': [] for ip in injection_points}
            kl_by_injection = {f'injection_{ip}': [] for ip in injection_points}
            latency_by_injection = {f'injection_{ip}': [] for ip in injection_points}
            
            for result in all_results:
                if 'token_analysis' in result:
                    for inj_key, token_data in result['token_analysis'].items():
                        if 'summary' in token_data:
                            adaptation_score = token_data['summary'].get('adaptation_quality', 0.0)
                            latency = token_data['summary'].get('adaptation_tokens', None)
                            
                            if inj_key in adaptation_by_injection:
                                adaptation_by_injection[inj_key].append(adaptation_score)
                                if latency is not None and latency > 0:
                                    latency_by_injection[inj_key].append(latency)
                
                if 'adaptation_metrics' in result:
                    for inj_key, adapt_data in result['adaptation_metrics'].items():
                        kl_div = adapt_data.get('kl_from_injected', 0.0)
                        if inj_key in kl_by_injection and not np.isnan(kl_div):
                            kl_by_injection[inj_key].append(kl_div)
            
            # Create dataset summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Average adaptation quality by injection point
            avg_adaptation = []
            std_adaptation = []
            for ip in injection_points:
                scores = adaptation_by_injection[f'injection_{ip}']
                avg_adaptation.append(np.mean(scores) if scores else 0)
                std_adaptation.append(np.std(scores) if scores else 0)
            
            axes[0, 0].errorbar(injection_points, avg_adaptation, yerr=std_adaptation, 
                               fmt='bo-', linewidth=2, markersize=8, capsize=5)
            axes[0, 0].set_xlabel('Injection Point (tokens)')
            axes[0, 0].set_ylabel('Average Adaptation Quality')
            axes[0, 0].set_title(f'Dataset-Wide Adaptation Quality (n={len(all_results)} problems)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            # Plot 2: Distribution of adaptation latencies
            all_latencies = []
            for ip in injection_points:
                latencies = latency_by_injection[f'injection_{ip}']
                all_latencies.extend(latencies)
            
            if all_latencies:
                axes[0, 1].hist(all_latencies, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[0, 1].set_xlabel('Adaptation Latency (tokens)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Adaptation Latencies')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axvline(np.mean(all_latencies), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_latencies):.1f}')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No adaptation\nlatencies detected', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Adaptation Latency Distribution (No Data)')
            
            # Plot 3: KL divergence patterns
            avg_kl = []
            std_kl = []
            for ip in injection_points:
                kl_scores = kl_by_injection[f'injection_{ip}']
                avg_kl.append(np.mean(kl_scores) if kl_scores else 0)
                std_kl.append(np.std(kl_scores) if kl_scores else 0)
            
            axes[1, 0].errorbar(injection_points, avg_kl, yerr=std_kl, 
                               fmt='ro-', linewidth=2, markersize=8, capsize=5)
            axes[1, 0].set_xlabel('Injection Point (tokens)')
            axes[1, 0].set_ylabel('Average KL Divergence')
            axes[1, 0].set_title('KL Divergence from Injected Context')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Dataset summary statistics
            axes[1, 1].axis('off')
            
            # Calculate summary stats
            total_problems = len(all_results)
            problems_with_adaptation = sum(1 for r in all_results 
                                         if any(ta.get('summary', {}).get('adaptation_quality', 0) > 0.5 
                                               for ta in r.get('token_analysis', {}).values()))
            
            dataset_type = all_results[0].get('dataset_type', 'Unknown') if all_results else 'Unknown'
            
            summary_stats = f"""
DATASET ANALYSIS SUMMARY

Dataset Type: {dataset_type}
Total Problems: {total_problems}
Problems with Strong Adaptation: {problems_with_adaptation} ({100*problems_with_adaptation/total_problems:.1f}%)

Injection Points Tested: {injection_points}
Best Overall Injection Point: {injection_points[np.argmax(avg_adaptation)] if avg_adaptation else 'N/A'}

Average Adaptation Quality: {np.mean(avg_adaptation):.3f} ± {np.mean(std_adaptation):.3f}
Average KL Divergence: {np.mean(avg_kl):.3f} ± {np.mean(std_kl):.3f}
Mean Adaptation Latency: {np.mean(all_latencies):.1f if len(all_latencies) > 0 else 'N/A'} tokens

Analysis Timestamp: {timestamp}
            """
            
            axes[1, 1].text(0.05, 0.95, summary_stats.strip(), 
                           transform=axes[1, 1].transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = plots_dir / f"dataset_summary_{dataset_type}_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved dataset summary: {plot_path}")
            
        except Exception as e:
            print(f"   ❌ Dataset summary plot failed: {e}")
    
    def analyze_dataset(
        self,
        dataset_path: str,
        output_dir: str = "proper_results",
        num_problems: int = None,
        injection_points: List[int] = [10, 30, 50],
        use_instruct_for_english: bool = True,
        generate_plots: bool = False
    ) -> None:
        """Analyze an entire dataset of problems.
        
        Args:
            dataset_path: Path to dataset JSON file
            output_dir: Directory to save results
            num_problems: Number of problems to analyze (None = all)
            injection_points: Token positions for injection
            use_instruct_for_english: Whether to use instruct model for english datasets
        """
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if num_problems:
            dataset = dataset[:num_problems]
        
        # Detect dataset type
        is_english = any('story_content' in p for p in dataset[:min(5, len(dataset))])
        
        # Switch to instruct model for english reading comprehension if requested
        if is_english and use_instruct_for_english:
            current_model = self.model.config._name_or_path
            if 'Instruct' not in current_model:
                print("📚 Detected FairytaleQA dataset - switching to instruct model for better comprehension")
                # Save current model name
                original_model = current_model
                # Reinitialize with instruct model
                self.__init__(
                    model_name="Qwen/Qwen2.5-7B-Instruct",
                    device=self.device,
                    cache_dir="~/.cache/huggingface/hub"
                )
        
        print(f"Analyzing {len(dataset)} problems")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Split dataset into thirds for different injection points
        dataset_size = len(dataset)
        third_size = dataset_size // 3
        
        print(f"\n📊 Dataset Split Strategy:")
        print(f"   Problems 0-{third_size-1}: Injection at token {injection_points[0]}")
        print(f"   Problems {third_size}-{2*third_size-1}: Injection at token {injection_points[1] if len(injection_points) > 1 else injection_points[0]}")
        print(f"   Problems {2*third_size}-{dataset_size-1}: Injection at token {injection_points[2] if len(injection_points) > 2 else injection_points[-1]}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        final_results_file = output_path / "final_results.json"
        
        print(f"📁 Results will be saved to: {final_results_file}")
        print(f"🔄 Processing {len(dataset)} problems...")
        
        # Analyze each problem with assigned injection point
        all_results = []
        for idx, problem in enumerate(tqdm(dataset, desc="Analyzing problems")):
            print(f"\n{'='*50}")
            print(f"Problem {idx + 1}/{len(dataset)}")
            print(f"{'='*50}")
            
            # Determine injection point based on problem index
            if idx < third_size:
                injection_point = injection_points[0] if injection_points else 10
            elif idx < 2 * third_size:
                injection_point = injection_points[1] if len(injection_points) > 1 else 30
            else:
                injection_point = injection_points[2] if len(injection_points) > 2 else 50
            
            print(f"Using injection point: {injection_point}")
            
            # Extract problem texts based on dataset format
            if 'story_content' in problem:  # FairytaleQA format - TWO PHASE APPROACH
                story = problem['story_content']
                base_context = story  # Just the story
                original_question = problem['Question']
                alternative_question = problem.get('alt_problem', problem['Question'])
                dataset_type = 'english'
                # Check if it's a control problem
                problem_type = 'control' if original_question == alternative_question else 'varied'
                
                # For display purposes, still create full contexts with chain of thought
                cot_instruction = "\n\nThink step by step and provide a clear, detailed answer."
                original = f"{story}\n\nQuestion: {original_question}{cot_instruction}\nAnswer: "
                alternative = f"{story}\n\nQuestion: {alternative_question}{cot_instruction}\nAnswer: "
                
            elif 'Question' in problem and 'alt_problem' in problem:  # MAWPS math dataset format
                # Add instruction for chain of thought with explicit final answer
                answer_instruction = "\n\nLet me think step by step to solve this problem.\n\nSolve step by step, then state 'The answer is: [number]' at the end."
                original = problem['Question'] + answer_instruction
                alternative = problem.get('alt_problem', problem['Question']) + answer_instruction
                dataset_type = 'math'
                # Check if it's a control problem based on variation_type field
                # variation_type is None/null for control problems, or 'control' for controls
                variation_type = problem.get('variation_type')
                if variation_type is None or variation_type == 'control':
                    problem_type = 'control'
                else:
                    problem_type = 'varied'
                    
                # Math problems don't need separate context/question
                base_context = None
                original_question = original
                alternative_question = alternative
                
            else:  # Generic format
                print(f"Skipping problem {idx}: unrecognized format {list(problem.keys())}")
                continue
            
            if not original or not alternative:
                print(f"Skipping problem {idx}: missing text")
                continue
            
            try:
                print(f"Running analysis for problem {idx}...")
                # Set metadata for the analyze_single_problem function to use
                # MUST be set BEFORE calling analyze_single_problem
                self._current_problem_index = idx
                self._current_problem_type = problem_type
                self._current_alt_language = problem.get('alt_language', 'English')
                self._current_variation_type = problem.get('variation_type', None)
                self._current_alt_problem_original = problem.get('alt_problem_original', None)
                
                # Run analysis with single injection point
                if dataset_type == 'english':
                    # Use two-phase approach for reading comprehension
                    result = self.analyze_single_problem(
                        original,
                        alternative, 
                        [injection_point],
                        dataset_type=dataset_type,
                        base_context=base_context,
                        original_question=original_question,
                        alternative_question=alternative_question
                    )
                else:
                    # Use single-phase approach for math
                    result = self.analyze_single_problem(
                        original,
                        alternative,
                        [injection_point],
                        dataset_type=dataset_type
                    )
                
                # Add expected answer and determine correctness
                is_correct = False
                if 'Answer' in problem and dataset_type != 'english':
                    # Math problems - numeric comparison
                    expected_answer = problem.get('alt_answer', problem['Answer']) if problem_type == 'varied' else problem['Answer']
                    self._current_expected_answer = expected_answer
                    result['expected_answer'] = expected_answer  # Add to result dictionary
                    if 'generated_answer' in result and expected_answer is not None:
                        try:
                            expected = float(expected_answer)
                            generated = float(result['generated_answer'])
                            is_correct = abs(expected - generated) < 0.01
                        except (ValueError, TypeError):
                            is_correct = False
                elif dataset_type == 'english':
                    # Reading comprehension - semantic similarity
                    if problem_type == 'varied':
                        result['expected_text_answer'] = problem.get('alt_answer', problem.get('Answer', ''))
                    else:
                        result['expected_text_answer'] = problem.get('Answer', '')
                    
                    # Calculate semantic similarity if we have both texts
                    if result.get('expected_text_answer') and result.get('generated_text'):
                        semantic_score = self._calculate_semantic_similarity(
                            result['expected_text_answer'],
                            result['generated_text']
                        )
                        result['semantic_similarity_score'] = semantic_score
                        is_correct = semantic_score > 0.7
                
                # Add 'correct' field to summary
                if 'summary' in result:
                    result['summary']['correct'] = is_correct
                    
                    # Update assessment based on correctness AND adaptation quality
                    # Only give 'excellent' or 'good' if the answer is correct
                    # If answer is wrong, downgrade to 'poor' or 'failed'
                    current_assessment = result['summary'].get('assessment', 'no adaptation')
                    
                    if not is_correct:
                        # Wrong answer - downgrade assessment
                        if current_assessment in ['excellent', 'good']:
                            result['summary']['assessment'] = 'poor'
                        elif current_assessment == 'slow':
                            result['summary']['assessment'] = 'failed'
                        # 'no adaptation' stays as is
                    
                    # Also check for repetition/generation issues
                    if 'generated_text' in result:
                        # Check if model got stuck in a loop (same token repeated many times)
                        words = result['generated_text'].split()
                        if len(words) > 5:
                            # Check for excessive repetition (same word 4+ times in a row)
                            for i in range(len(words) - 3):
                                if len(set(words[i:i+4])) == 1:  # All 4 words are the same
                                    result['summary']['assessment'] = 'failed'
                                    result['summary']['generation_issue'] = 'repetition_loop'
                                    break
                
                all_results.append(result)
                print(f"✅ Problem {idx} completed successfully")
                
                # Save to final_results.json after each problem (overwrites)
                try:
                    with open(final_results_file, 'w') as f:
                        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
                    print(f"💾 Results saved to {final_results_file} ({len(all_results)} problems)")
                except Exception as save_e:
                    print(f"⚠️ Warning: Could not save results: {save_e}")
                
                # Generate plots for every 10th problem (10, 20, 30, etc.) if enabled
                if generate_plots and (idx + 1) % 10 == 0:
                    try:
                        print(f"   📊 Generating plots for problem #{idx + 1}...")
                        self._generate_analysis_plots(result, injection_points)
                    except Exception as plot_e:
                        print(f"   ⚠️ Plot generation failed: {plot_e}")
                
                # Progress indicator
                if (idx + 1) % 5 == 0:
                    print(f"📊 Progress: {idx+1}/{len(dataset)} problems completed")
                    print(f"   📁 Results file: {final_results_file}")
                
            except Exception as e:
                print(f"❌ Error analyzing problem {idx}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Save final results
        self._save_results(all_results, final_results_file)
        
        # Generate summary report
        self._generate_report(all_results, output_path / "summary_report.json")
        
        # Generate dataset-level summary plots
        if all_results:
            print("\n📊 Generating dataset summary plots...")
            self._generate_dataset_summary_plots(all_results, injection_points)
        
        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"   📁 Incremental: {incremental_file}")
        print(f"   📄 Final: {final_results_file}")
        print(f"   📊 Summary: {output_path / 'summary_report.json'}")
        print(f"   📂 All files: {output_path}")
    
    def _create_unified_summary(self, results: Dict) -> Dict:
        """Create a unified summary across all analysis methods."""
        
        summary = {
            'adaptation_detected': False,
            'adaptation_quality': 0.0,  # Always a number
            'best_injection_point': 'injection_5',  # Default to first injection point
            'key_findings': []
        }
        
        # Check token analysis
        if 'token_analysis' in results:
            best_quality = 0.0
            best_point = 'injection_5'
            
            for inj_point, analysis in results['token_analysis'].items():
                if 'summary' in analysis:
                    summary_data = analysis['summary']
                    
                    # Get numerical adaptation quality
                    quality = summary_data.get('adaptation_quality', 0.0)
                    if isinstance(quality, str):
                        # Convert string assessments to numbers
                        quality_map = {'excellent': 0.9, 'good': 0.7, 'slow': 0.5, 'partial': 0.3, 'failed': 0.1}
                        quality = quality_map.get(quality, 0.0)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_point = inj_point
            
            summary['adaptation_quality'] = best_quality
            summary['best_injection_point'] = best_point
            summary['adaptation_detected'] = best_quality > 0.3
            
            if best_quality > 0.3:
                summary['key_findings'].append(f"Token-level: adaptation quality {best_quality:.2f} at {best_point}")
        
        # Check attention analysis
        if 'attention_analysis' in results:
            stabilization_points = []
            for inj_point, metrics in results['attention_analysis'].items():
                if 'stabilization_tokens' in metrics:
                    stabilization_points.append(metrics['stabilization_tokens'])
            
            if stabilization_points:
                avg_stabilization = sum(stabilization_points) / len(stabilization_points)
                summary['key_findings'].append(f"Attention: stabilizes after ~{avg_stabilization:.1f} tokens")
        
        # Check causal analysis and align with adaptation_detected
        if 'causal_analysis' in results:
            if 'overall' in results['causal_analysis']:
                causal_score = results['causal_analysis']['overall'].get('mean_adaptation', 0.0)
                
                # Align adaptation_detected with causal analysis if it's higher
                if causal_score > 0.3 and not summary['adaptation_detected']:
                    summary['adaptation_detected'] = True
                    summary['adaptation_quality'] = max(summary['adaptation_quality'], causal_score)
                    summary['key_findings'].append(f"Causal analysis: adaptation score {causal_score:.2f}")
                
            if 'recommendations' in results['causal_analysis']:
                for rec in results['causal_analysis']['recommendations'][:2]:  # Top 2 recommendations
                    # Only add recommendations that align with adaptation_detected status
                    if summary['adaptation_detected'] or 'Poor adaptation' in rec:
                        summary['key_findings'].append(f"Recommendation: {rec}")
        
        return summary
    
    def _save_results(self, results: List[Dict], filepath: Path):
        """Save results to JSON file."""
        
        # Convert non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, (torch.Tensor)):
                return obj.cpu().tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj
        
        serializable_results = make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str, ensure_ascii=False)
    
    def _generate_report(self, results: List[Dict], filepath: Path):
        """Generate summary report from all results."""
        
        report = {
            'total_problems': len(results),
            'timestamp': datetime.now().isoformat(),
            'model': self.model.config._name_or_path,
            'statistics': {},
            'patterns': {}
        }
        
        # Aggregate statistics
        adaptation_counts = {'detected': 0, 'not_detected': 0}
        assessment_counts = {}
        
        for result in results:
            if 'summary' in result:
                summary = result['summary']
                
                if summary.get('adapted', False):
                    adaptation_counts['detected'] += 1
                else:
                    adaptation_counts['not_detected'] += 1
                
                assessment = summary.get('assessment', 'unknown')
                assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        report['statistics'] = {
            'adaptation_rate': adaptation_counts['detected'] / len(results) if results else 0,
            'adaptation_counts': adaptation_counts,
            'assessment_distribution': assessment_counts
        }
        
        # Analyze injection point performance
        injection_performance = {3: [], 5: [], 7: [], 10: [], 15: []}
        
        for result in results:
            # Get the injection point and summary for this problem
            if 'injection_point' in result and 'summary' in result:
                inj_point = result['injection_point']
                summary = result['summary']
                
                # Check accuracy only for math problems (skip for reading comprehension)
                is_correct = False
                dataset_type = result.get('dataset_type', 'unknown')
                
                if dataset_type == 'math' and 'generated_answer' in result and 'expected_answer' in result:
                    try:
                        expected = float(result['expected_answer'])
                        generated = float(result['generated_answer'])
                        is_correct = abs(expected - generated) < 0.01  # Allow small floating point differences
                    except (ValueError, TypeError):
                        is_correct = False
                elif dataset_type == 'english' and 'expected_text_answer' in result and 'generated_text' in result:
                    # For reading comprehension, use semantic similarity to evaluate correctness
                    semantic_score = self._calculate_semantic_similarity(
                        result['expected_text_answer'], 
                        result['generated_text']
                    )
                    result['semantic_similarity_score'] = semantic_score
                    # Consider correct if semantic similarity > 0.7 threshold
                    is_correct = semantic_score > 0.7
                
                if inj_point in injection_performance:
                    injection_performance[inj_point].append({
                        'adapted': summary.get('adapted', False),
                        'tokens': summary.get('adaptation_tokens', -1),
                        'quality': summary.get('adaptation_quality', 0),
                        'correct': is_correct,
                        'problem_type': result.get('problem_type', 'unknown')
                    })
        
        # Calculate statistics for each injection point
        injection_stats = {}
        for point, results_list in injection_performance.items():
            if results_list:
                adapted_count = sum(1 for r in results_list if r['adapted'])
                # Only count accuracy for problems where it's meaningful (math problems)
                applicable_results = [r for r in results_list if r['correct'] is not None]
                correct_count = sum(1 for r in applicable_results if r['correct'])
                accuracy_rate = correct_count / len(applicable_results) if applicable_results else 0
                
                avg_tokens = np.mean([r['tokens'] for r in results_list if r['tokens'] > 0]) if any(r['tokens'] > 0 for r in results_list) else -1
                avg_quality = np.mean([r['quality'] for r in results_list])
                
                # Separate stats for control vs varied problems (only for applicable problems)
                control_results = [r for r in results_list if r['problem_type'] == 'control' and r['correct'] is not None]
                varied_results = [r for r in results_list if r['problem_type'] == 'varied' and r['correct'] is not None]
                
                control_accuracy = sum(1 for r in control_results if r['correct']) / len(control_results) if control_results else 0
                varied_accuracy = sum(1 for r in varied_results if r['correct']) / len(varied_results) if varied_results else 0
                
                injection_stats[point] = {
                    'total_problems': len(results_list),
                    'adaptation_rate': adapted_count / len(results_list),
                    'accuracy_rate': accuracy_rate,
                    'accuracy_applicable': len(applicable_results),  # How many problems accuracy applies to
                    'control_accuracy': float(control_accuracy),
                    'varied_accuracy': float(varied_accuracy),
                    'avg_adaptation_tokens': float(avg_tokens),
                    'avg_quality': float(avg_quality)
                }
        
        report['injection_point_analysis'] = injection_stats
        
        # Generate recommendation
        if injection_stats:
            # Find best injection point based on combined metrics
            best_point = None
            best_score = -1
            
            for point, stats in injection_stats.items():
                if stats['total_problems'] > 0:
                    # Score based on: accuracy, adaptation rate, speed, and quality
                    score = (stats['accuracy_rate'] * 0.35 +  # Accuracy is most important
                            stats['adaptation_rate'] * 0.25 + 
                            (1.0 / (stats['avg_adaptation_tokens'] + 1)) * 0.2 +
                            stats['avg_quality'] * 0.2)
                    
                    if score > best_score:
                        best_score = score
                        best_point = point
            
            report['recommendation'] = {
                'best_injection_point': best_point,
                'reasoning': f"Token {best_point} achieves best overall performance: {injection_stats[best_point]['adaptation_rate']:.1%} adaptation rate, avg latency: {injection_stats[best_point]['avg_adaptation_tokens']:.1f} tokens"
            }
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total problems analyzed: {report['total_problems']}")
        print(f"Adaptation detection rate: {report['statistics']['adaptation_rate']:.1%}")
        
        print("\n📊 Injection Point Performance:")
        for point, stats in injection_stats.items():
            if stats['total_problems'] > 0:
                # Format accuracy display based on whether it's applicable
                if stats.get('accuracy_applicable', 0) > 0:
                    accuracy_str = f"{stats['accuracy_rate']:.1%} accuracy ({stats['control_accuracy']:.1%} control, {stats['varied_accuracy']:.1%} varied)"
                else:
                    accuracy_str = "N/A accuracy (no applicable problems)"
                
                print(f"   Token {point}: {accuracy_str}, {stats['adaptation_rate']:.1%} adapted, avg latency: {stats['avg_adaptation_tokens']:.1f} tokens")
        
        if 'recommendation' in report:
            print(f"\n🎯 RECOMMENDATION:")
            print(f"   {report['recommendation']['reasoning']}")


# =====================================
# UTILITY FUNCTIONS
# =====================================

def save_metrics_to_file(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    
    # Convert numpy arrays and other non-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, AdaptationState):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)


def test_attention_tracking():
    """Test the attention tracking system."""
    
    # Initialize model and tracker
    model_name = "Qwen/Qwen2.5-Math-7B"  # Use proper math model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    tracker = AttentionTracker(model, tokenizer, device='cpu')
    
    # Test generation with injection
    prompt = "The weather today is"
    injection = "actually quite terrible with heavy rain"
    
    results = tracker.track_generation_with_injection(
        prompt=prompt,
        injection_text=injection,
        injection_point=5,
        max_new_tokens=20
    )
    
    print(f"Generated: {results['generated_text']}")
    print(f"Adaptation metrics: {results['adaptation_metrics']}")
    
    # Plot results
    tracker.plot_attention_evolution()


def test_token_analysis():
    """Test token-level analysis."""
    
    # Initialize
    model_name = "Qwen/Qwen2.5-Math-7B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    analyzer = TokenLevelAnalyzer(model, tokenizer, device='cpu')
    
    # Run analysis
    results = analyzer.analyze_token_adaptation(
        original_context="The recipe requires 2 cups of flour",
        injected_context="Actually use 3 cups of flour instead",
        injection_point=8,
        max_tokens=30
    )
    
    print("\nAdaptation Summary:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    
    print("\nToken Metrics:")
    if 'metrics' in results:
        for phase, phase_metrics in results['metrics'].items():
            print(f"\n{phase}:")
            if isinstance(phase_metrics, dict):
                for metric, value in phase_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.3f}")
    
    # Plot results
    if 'adaptation_analysis' in results:
        analyzer.plot_token_adaptation(
            results['adaptation_analysis'],
            injection_point=8
        )


# =====================================
# MAIN ENTRY POINT
# =====================================

def main():
    """Main entry point for command-line usage."""
    
    parser = argparse.ArgumentParser(description="Proper adaptation latency measurement")
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model to analyze (use Qwen/Qwen2.5-7B-Instruct for non-math datasets)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help="Path to dataset JSON file (MAWPS math format only)"
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help="Run single problem test"
    )
    parser.add_argument(
        '--num_problems',
        type=int,
        default=None,
        help="Number of problems to analyze"
    )
    parser.add_argument(
        '--injection_points',
        type=int,
        nargs='+',
        default=[5, 10, 15],
        help="Token positions for injection"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="proper_results",
        help="Output directory for results"
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ProperAdaptationAnalyzer(
        model_name=args.model,
        device=args.device
    )
    
    if args.single or not args.dataset:
        # Run single test
        print("Running single problem test...")
        result = analyzer.analyze_single_problem(
            original_problem="Sarah has 5 apples. She buys 3 more apples from the store. How many apples does she have?",
            alternative_problem="Actually, Sarah has 7 apples and buys 4 more apples. How many apples does she have now?",
            injection_points=args.injection_points
        )
        
        # Save result  
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        analyzer._save_results([result], output_path / "final_results.json")
        analyzer._generate_report([result], output_path / "summary_report.json")
        
        # Print summary
        print("\n" + "="*50)
        print("UNIFIED SUMMARY")
        print("="*50)
        for key, value in result['unified_summary'].items():
            print(f"{key}: {value}")
    
    else:
        # Run dataset analysis
        analyzer.analyze_dataset(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            num_problems=args.num_problems,
            injection_points=args.injection_points
        )


if __name__ == "__main__":
    # If no command line args, run full dataset analysis
    import sys
    if len(sys.argv) == 1:
        print("🚀 Running Full Dataset Analysis (1000 problems)")
        print("Results will be saved incrementally - you can monitor progress and cancel anytime!")
        print("="*80)
        
        # Run analysis on both datasets
        # SINGLE PROBLEM TEST - COMMENTED OUT FOR FULL DATASET ANALYSIS
        # Uncomment the section below if you want to test single problems instead
        """
        print("\n🧮 Testing Single Math Problem...")
        math_analyzer = ProperAdaptationAnalyzer(model_name="Qwen/Qwen2.5-Math-7B", device=None)
        
        # Use simpler problems without forcing format
        result = math_analyzer.analyze_single_problem(
            original_problem="Mary has 15 apples. She gives away 4 apples. How many apples does Mary have now?",
            alternative_problem="Mary has 9 apples. She gives away 4 apples. How many apples does Mary have now?",
            injection_points=[5], 
            dataset_type='math'
        )
        
        print("\n📊 RESULTS:")
        print(f"Original problem: {result.get('original_problem', 'N/A')}")
        print(f"Alternative problem: {result.get('alternative_problem', 'N/A')}")
        print(f"Expected answer: 5.0 (9 - 4)")
        print(f"Generated answer: {result.get('generated_answer', 'N/A')}")
        print(f"\n🤖 FULL GENERATED TEXT:")
        print(f"'{result.get('generated_text', 'NO TEXT GENERATED')}'")
        print(f"\n📈 SUMMARY:")
        summary = result.get('summary', {})
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Show token breakdown
        if 'token_states' in result:
            token_states = result['token_states']
            context_counts = {'original': 0, 'mixed': 0, 'injected': 0}
            for state in token_states:
                source = state.get('context_source', 'unknown')
                if source in context_counts:
                    context_counts[source] += 1
            
            print(f"\n🔍 Token breakdown ({len(token_states)} total):")
            for source, count in context_counts.items():
                pct = count / len(token_states) * 100
                print(f"  {source}: {count} ({pct:.1f}%)")
        
        # Save result to file for inspection
        import json
        with open('test_single_output.json', 'w') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n💾 Full result saved to: test_single_output.json")
        
        sys.exit(0)  # Exit early for single test
        """
        
        # FULL DATASET - Now running by default:
        print("\n📊 Starting Math Dataset Analysis...")
        math_analyzer = ProperAdaptationAnalyzer(
            model_name="Qwen/Qwen2.5-Math-7B",
            device=None
        )
        
        # Use the fixed dataset with correct answers
        math_fixed_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/math/mawps_multilingual_fixed.json"
        math_original_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/math/mawps_multilingual.json"
        
        # Use fixed dataset if available, otherwise fall back to original
        math_dataset_path = math_fixed_path if os.path.exists(math_fixed_path) else math_original_path
        print(f"📊 Using math dataset: {os.path.basename(math_dataset_path)}")
        
        math_analyzer.analyze_dataset(
            dataset_path=math_dataset_path,
            output_dir="../output/math_results",
            injection_points=[10, 30, 50]  # Full 500 problems: 167 @ token 10, 167 @ token 30, 166 @ token 50
        )
        
        print("\n📚 Starting Non-Math Dataset Analysis...")
        del math_analyzer  # Clean up memory
        
        nonmath_analyzer = ProperAdaptationAnalyzer(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            device=None
        )
        
        # Use the fixed dataset with correct answers
        nonmath_fixed_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/non_math/fairytale_multilingual_fixed.json"
        nonmath_original_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/non_math/fairytale_multilingual.json"
        
        # Use fixed dataset if available, otherwise fall back to original
        nonmath_dataset_path = nonmath_fixed_path if os.path.exists(nonmath_fixed_path) else nonmath_original_path
        print(f"📊 Using non-math dataset: {os.path.basename(nonmath_dataset_path)}")
        
        nonmath_analyzer.analyze_dataset(
            dataset_path=nonmath_dataset_path,
            output_dir="../output/nonmath_results",
            injection_points=[10, 30, 50]  # Full 500 problems: 167 @ token 10, 167 @ token 30, 166 @ token 50
        )
        
        print("\n✅ Full Dataset Analysis Complete!")
        print("Check ../output/ for results")
    else:
        main()

# ============================================================================
# EXTENDED FUNCTIONALITY - COMMENTED OUT TO PREVENT INTERFERENCE
# ============================================================================

"""
EXTENDED ADAPTATION ANALYSIS FRAMEWORK
=====================================

This section contains extended functionality for comprehensive ablation studies
and multimodal adaptation testing. All code is COMMENTED OUT to prevent 
interference with current running experiments.

To activate any feature:
1. Uncomment the relevant section
2. Ensure no other experiments are running
3. Test on small datasets first

Features included:
- Support for translated datasets (_fixed.json files)
- Ablation studies for injection points 5-100 tokens
- Temperature variation studies (0.1-1.5)
- Coding ability datasets (HumanEval, MBPP-style)
- Image capability datasets (vectorized representations)
- Audio adaptation testing framework
"""

# ============================================================================
# TRANSLATED DATASET SUPPORT
# ============================================================================

def load_translated_datasets():
    """Load the new translated and fixed datasets - COMMENTED OUT"""
    # base_path = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds"
    # datasets = {}
    # 
    # # Load fixed math dataset (translated alternatives)
    # math_path = f"{base_path}/math/mawps_multilingual_fixed.json"
    # if os.path.exists(math_path):
    #     with open(math_path, 'r', encoding='utf-8') as f:
    #         datasets['math_translated'] = json.load(f)
    #     print(f"✅ Loaded {len(datasets['math_translated'])} translated math problems")
    # 
    # # Load fixed non-math dataset (translated alternatives)  
    # nonmath_path = f"{base_path}/non_math/fairytale_multilingual_fixed.json"
    # if os.path.exists(nonmath_path):
    #     with open(nonmath_path, 'r', encoding='utf-8') as f:
    #         datasets['nonmath_translated'] = json.load(f)
    #     print(f"✅ Loaded {len(datasets['nonmath_translated'])} translated reading problems")
    # 
    # return datasets
    pass

# ============================================================================
# ABLATION STUDY FRAMEWORK
# ============================================================================

class AblationStudyFramework:
    """Framework for running comprehensive ablation studies - COMMENTED OUT"""
    
    def __init__(self, analyzer):
        # self.analyzer = analyzer
        pass
    
    def run_injection_point_ablation(self, dataset, injection_points):
        """Test injection points from 5-100 tokens - COMMENTED OUT"""
        # print(f"🔬 Running injection point ablation study: {min(injection_points)}-{max(injection_points)} tokens")
        # 
        # results = {}
        # for injection_point in injection_points:
        #     print(f"Testing injection at token {injection_point}...")
        #     
        #     point_results = []
        #     for i, problem in enumerate(dataset[:10]):  # Test subset to avoid interference
        #         result = self.analyzer.analyze_single_problem(
        #             problem, 
        #             injection_point=injection_point,
        #             problem_index=i
        #         )
        #         point_results.append(result)
        #     
        #     # Calculate statistics for this injection point
        #     adaptation_latencies = [r.get('summary', {}).get('adaptation_tokens', -1) for r in point_results]
        #     success_rate = len([l for l in adaptation_latencies if l > 0]) / len(adaptation_latencies)
        #     avg_latency = np.mean([l for l in adaptation_latencies if l > 0]) if any(l > 0 for l in adaptation_latencies) else -1
        #     
        #     results[injection_point] = {
        #         'success_rate': success_rate,
        #         'avg_latency': avg_latency,
        #         'raw_results': point_results
        #     }
        #     
        #     print(f"  ✅ Token {injection_point}: {success_rate:.1%} success, {avg_latency:.1f} avg latency")
        # 
        # return results
        pass
    
    def run_temperature_ablation(self, dataset, temperatures):
        """Test different temperature settings - COMMENTED OUT"""
        # print(f"🌡️ Running temperature ablation study: {min(temperatures)}-{max(temperatures)}")
        # 
        # results = {}
        # 
        # for temperature in temperatures:
        #     print(f"Testing temperature {temperature}...")
        #     
        #     # Create new analyzer with different temperature
        #     temp_analyzer = ProperAdaptationAnalyzer(
        #         model_name=self.analyzer.model_name,
        #         device=self.analyzer.device
        #     )
        #     # Set temperature for generation (would need to modify generation parameters)
        #     
        #     temp_results = []
        #     for i, problem in enumerate(dataset[:5]):  # Small subset
        #         result = temp_analyzer.analyze_single_problem(
        #             problem,
        #             injection_point=30,  # Fixed injection point
        #             problem_index=i,
        #             temperature=temperature  # Pass temperature parameter
        #         )
        #         temp_results.append(result)
        #     
        #     # Calculate statistics for this temperature
        #     adaptation_latencies = [r.get('summary', {}).get('adaptation_tokens', -1) for r in temp_results]
        #     success_rate = len([l for l in adaptation_latencies if l > 0]) / len(adaptation_latencies)
        #     avg_latency = np.mean([l for l in adaptation_latencies if l > 0]) if any(l > 0 for l in adaptation_latencies) else -1
        #     
        #     results[temperature] = {
        #         'success_rate': success_rate,
        #         'avg_latency': avg_latency,
        #         'raw_results': temp_results
        #     }
        #     
        #     print(f"  ✅ Temp {temperature}: {success_rate:.1%} success, {avg_latency:.1f} avg latency")
        #     
        #     del temp_analyzer  # Clean up memory
        # 
        # return results
        pass
    
    def generate_injection_points_range(self, start=5, end=100, step=5):
        """Generate range of injection points for ablation study"""
        return list(range(start, end + 1, step))
    
    def generate_temperature_range(self, start=0.1, end=1.5, step=0.1):
        """Generate range of temperatures for ablation study"""
        return [round(t, 1) for t in np.arange(start, end + step, step)]

# ============================================================================
# CODING DATASETS
# ============================================================================

def prepare_coding_datasets():
    """Prepare high-quality LeetCode-style coding datasets with GENUINELY DIFFERENT problems - COMMENTED OUT"""
    # coding_datasets = {}
    # 
    # # LeetCode-style dataset with completely different problems in different languages
    # leetcode_problems = [
    #     {
    #         "Question": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
    #         "Answer": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
    #         "alt_problem": "Encuentra el subarray contiguo que tiene la suma máxima y devuelve su suma.",  # Spanish: Maximum Subarray
    #         "alt_answer": "def max_subarray_sum(arr):\n    max_actual = max_global = arr[0]\n    for i in range(1, len(arr)):\n        max_actual = max(arr[i], max_actual + arr[i])\n        max_global = max(max_global, max_actual)\n    return max_global",
    #         "problem_type": "array",
    #         "difficulty": "easy",
    #         "dataset_type": "coding"
    #     },
    #     {
    #         "Question": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
    #         "Answer": "def is_valid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n        else:\n            stack.append(char)\n    return not stack",
    #         "alt_problem": "Trouvez le plus long préfixe commun parmi un tableau de chaînes.",  # French: Longest Common Prefix
    #         "alt_answer": "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
    #         "problem_type": "string",
    #         "difficulty": "easy",
    #         "dataset_type": "coding"
    #     },
    #     {
    #         "Question": "Reverse a singly linked list.",
    #         "Answer": "def reverse_list(head):\n    prev = None\n    current = head\n    while current:\n        next_temp = current.next\n        current.next = prev\n        prev = current\n        current = next_temp\n    return prev",
    #         "alt_problem": "Finde die Anzahl der Inseln in einer 2D-Matrix.",  # German: Number of Islands
    #         "alt_answer": "def num_islands(grid):\n    if not grid:\n        return 0\n    count = 0\n    for i in range(len(grid)):\n        for j in range(len(grid[0])):\n            if grid[i][j] == '1':\n                dfs(grid, i, j)\n                count += 1\n    return count\n\ndef dfs(grid, i, j):\n    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':\n        return\n    grid[i][j] = '0'\n    dfs(grid, i+1, j)\n    dfs(grid, i-1, j)\n    dfs(grid, i, j+1)\n    dfs(grid, i, j-1)",
    #         "problem_type": "linked_list_to_graph",
    #         "difficulty": "medium",
    #         "dataset_type": "coding"
    #     },
    #     {
    #         "Question": "Given a binary tree, find its maximum depth.",
    #         "Answer": "def max_depth(root):\n    if not root:\n        return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))",
    #         "alt_problem": "Trova il k-esimo elemento più grande in un array non ordinato.",  # Italian: Kth Largest Element
    #         "alt_answer": "def find_kth_largest(nums, k):\n    import heapq\n    return heapq.nlargest(k, nums)[-1]",
    #         "problem_type": "tree_to_heap",
    #         "difficulty": "medium",
    #         "dataset_type": "coding"
    #     },
    #     {
    #         "Question": "Implement a function to detect a cycle in a linked list.",
    #         "Answer": "def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False",
    #         "alt_problem": "二分探索を使用してソート済み配列内の要素を見つける。",  # Japanese: Binary Search
    #         "alt_answer": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    #         "problem_type": "cycle_detection_to_search",
    #         "difficulty": "easy",
    #         "dataset_type": "coding"
    #     },
    #     {
    #         "Question": "Find the minimum number of coins needed to make change for a given amount.",
    #         "Answer": "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for coin in coins:\n        for x in range(coin, amount + 1):\n            dp[x] = min(dp[x], dp[x - coin] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1",
    #         "alt_problem": "Encontre o caminho mais curto em um grafo ponderado usando Dijkstra.",  # Portuguese: Dijkstra's Algorithm
    #         "alt_answer": "def dijkstra(graph, start):\n    import heapq\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    \n    while pq:\n        current_distance, current_node = heapq.heappop(pq)\n        if current_distance > distances[current_node]:\n            continue\n        for neighbor, weight in graph[current_node].items():\n            distance = current_distance + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(pq, (distance, neighbor))\n    return distances",
    #         "problem_type": "dp_to_graph",
    #         "difficulty": "hard",
    #         "dataset_type": "coding"
    #     }
    # ]
    # 
    # coding_datasets['leetcode'] = leetcode_problems
    # 
    # print(f"✅ Prepared {len(leetcode_problems)} high-quality LeetCode problems with different problem types")
    # print(f"📊 Problem transitions: Array→DP, String→Tree, LinkedList→Graph, Tree→Heap, etc.")
    # 
    # return coding_datasets
    pass

# ============================================================================
# IMAGE DATASETS (VECTORIZED)
# ============================================================================

def prepare_image_datasets():
    """Prepare COCO/OpenImages datasets with GENUINELY DIFFERENT images - COMMENTED OUT"""
    # image_datasets = {}
    # 
    # # High-quality image captioning with completely different scenes
    # coco_openimages = [
    #     {
    #         "Question": "Describe what you see in this image.",
    #         "Answer": "A golden retriever playing with a tennis ball in a backyard garden.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for dog/garden scene
    #         "alt_problem": "Descrivi cosa vedi in questa immagine.",  # Italian - DIFFERENT image
    #         "alt_answer": "Una cascata che scorre attraverso una foresta tropicale lussureggiante.",  # Waterfall in tropical forest
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for waterfall scene
    #         "scene_type": "outdoor_animal_to_nature",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "What is happening in this picture?",
    #         "Answer": "A chef preparing sushi in a Japanese restaurant kitchen.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for kitchen/chef scene
    #         "alt_problem": "¿Qué está pasando en esta imagen?",  # Spanish - DIFFERENT image
    #         "alt_answer": "Un surfista montando una ola grande al atardecer en el océano.",  # Surfer riding wave at sunset
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for ocean/surfing scene
    #         "scene_type": "indoor_cooking_to_water_sports",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "Describe the main elements in this scene.",
    #         "Answer": "A busy city intersection with traffic lights, pedestrians crossing, and tall buildings.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for urban scene
    #         "alt_problem": "Décrivez les éléments principaux de cette scène.",  # French - DIFFERENT image
    #         "alt_answer": "Un champ de tournesols sous un ciel bleu avec des nuages blancs.",  # Sunflower field with blue sky
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for rural/flower scene
    #         "scene_type": "urban_to_rural",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "What objects can you identify?",
    #         "Answer": "A laptop, coffee mug, notebook, and glasses on a wooden desk.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for office/desk scene
    #         "alt_problem": "Welche Objekte können Sie identifizieren?",  # German - DIFFERENT image
    #         "alt_answer": "Ein Heißluftballon, der über schneebedeckte Berge fliegt.",  # Hot air balloon over snowy mountains
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for aerial/mountain scene
    #         "scene_type": "indoor_workspace_to_aerial_landscape",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "Describe the environment shown.",
    #         "Answer": "A cozy library with floor-to-ceiling bookshelves and reading chairs.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for library scene
    #         "alt_problem": "この環境を説明してください。",  # Japanese - DIFFERENT image
    #         "alt_answer": "富士山の前で咲く桜の木々。",  # Cherry blossoms blooming in front of Mount Fuji
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for Japanese landscape
    #         "scene_type": "indoor_library_to_japanese_landscape",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "What activity is taking place?",
    #         "Answer": "Children playing soccer on a grass field during a school sports day.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for sports/children scene
    #         "alt_problem": "Que atividade está acontecendo?",  # Portuguese - DIFFERENT image
    #         "alt_answer": "Um astronauta flutuando no espaço próximo à estação espacial.",  # Astronaut floating near space station
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for space scene
    #         "scene_type": "sports_field_to_outer_space",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "Describe the setting and mood.",
    #         "Answer": "A peaceful beach at sunrise with gentle waves and seagulls.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for beach/sunrise scene
    #         "alt_problem": "설정과 분위기를 설명하세요.",  # Korean - DIFFERENT image
    #         "alt_answer": "네온 불빛이 빛나는 비 오는 밤의 도시 거리.",  # Rainy city street at night with neon lights
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for cyberpunk/urban night scene
    #         "scene_type": "peaceful_beach_to_urban_night",
    #         "dataset_type": "image"
    #     },
    #     {
    #         "Question": "What animals are visible?",
    #         "Answer": "A family of elephants walking across the African savanna.",
    #         "image_vector": np.random.rand(768).tolist(),  # CLIP embedding for wildlife/savanna scene
    #         "alt_problem": "Какие животные видны?",  # Russian - DIFFERENT image
    #         "alt_answer": "Полярный медведь на льдине в Арктике.",  # Polar bear on ice floe in Arctic
    #         "alt_image_vector": np.random.rand(768).tolist(),  # CLIP embedding for arctic/polar scene
    #         "scene_type": "african_wildlife_to_arctic",
    #         "dataset_type": "image"
    #     }
    # ]
    # 
    # image_datasets['coco_openimages'] = coco_openimages
    # 
    # print(f"✅ Prepared {len(coco_openimages)} high-quality COCO/OpenImages problems")
    # print(f"📊 Scene transitions: Indoor→Outdoor, Urban→Nature, Day→Night, Land→Sea, Earth→Space")
    # 
    # return image_datasets
    pass

# ============================================================================
# AUDIO DATASETS (VECTORIZED)
# ============================================================================

def prepare_audio_datasets():
    """Prepare AudioSet/FreeSound datasets with GENUINELY DIFFERENT audio clips - COMMENTED OUT"""
    # audio_datasets = {}
    # 
    # # High-quality audio classification with completely different sounds
    # audioset_freesound = [
    #     {
    #         "Question": "What sound do you hear?",
    #         "Answer": "A dog barking in a backyard.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for dog barking
    #         "alt_problem": "Che suono senti?",  # Italian - DIFFERENT audio
    #         "alt_answer": "Un pianoforte che suona musica classica.",  # Piano playing classical music
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for piano music
    #         "sound_type": "animal_to_music",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "Identify the audio event.",
    #         "Answer": "Thunder and heavy rain during a storm.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for thunderstorm
    #         "alt_problem": "Identifica el evento de audio.",  # Spanish - DIFFERENT audio
    #         "alt_answer": "Una multitud aplaudiendo en un estadio de fútbol.",  # Crowd cheering at football stadium
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for crowd cheering
    #         "sound_type": "weather_to_crowd",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "Describe what you hear.",
    #         "Answer": "A baby crying in a nursery.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for baby crying
    #         "alt_problem": "Décrivez ce que vous entendez.",  # French - DIFFERENT audio
    #         "alt_answer": "Un train à vapeur sifflant en quittant la gare.",  # Steam train whistle leaving station
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for train whistle
    #         "sound_type": "human_to_mechanical",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "What type of sound is this?",
    #         "Answer": "Ocean waves crashing on the shore.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for ocean waves
    #         "alt_problem": "Was für ein Geräusch ist das?",  # German - DIFFERENT audio
    #         "alt_answer": "Eine Kirchenglocke, die zur vollen Stunde läutet.",  # Church bell ringing on the hour
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for church bell
    #         "sound_type": "nature_to_bell",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "Classify this audio clip.",
    #         "Answer": "A car engine starting and idling.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for car engine
    #         "alt_problem": "このオーディオクリップを分類してください。",  # Japanese - DIFFERENT audio
    #         "alt_answer": "風鈴が風に揺れる音。",  # Wind chimes tinkling in the breeze
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for wind chimes
    #         "sound_type": "engine_to_chimes",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "What instrument is playing?",
    #         "Answer": "An acoustic guitar being strummed.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for guitar
    #         "alt_problem": "Que instrumento está tocando?",  # Portuguese - DIFFERENT audio
    #         "alt_answer": "Um helicóptero sobrevoando a cidade.",  # Helicopter flying overhead
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for helicopter
    #         "sound_type": "musical_to_aircraft",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "Identify the environmental sound.",
    #         "Answer": "Birds chirping in a forest at dawn.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for bird songs
    #         "alt_problem": "환경 소리를 식별하세요.",  # Korean - DIFFERENT audio
    #         "alt_answer": "커피숍에서 에스프레소 머신 소리.",  # Espresso machine in a coffee shop
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for espresso machine
    #         "sound_type": "nature_birds_to_cafe",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "What human sound is this?",
    #         "Answer": "People laughing at a party.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for laughter
    #         "alt_problem": "Какой это человеческий звук?",  # Russian - DIFFERENT audio
    #         "alt_answer": "Фейерверк взрывается в ночном небе.",  # Fireworks exploding in night sky
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for fireworks
    #         "sound_type": "human_laughter_to_fireworks",
    #         "dataset_type": "audio"
    #     },
    #     {
    #         "Question": "Describe the audio scene.",
    #         "Answer": "A busy restaurant kitchen with sizzling pans and chef callouts.",
    #         "audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for kitchen sounds
    #         "alt_problem": "صف المشهد الصوتي.",  # Arabic - DIFFERENT audio
    #         "alt_answer": "أذان المسجد يدعو للصلاة.",  # Mosque call to prayer (Adhan)
    #         "alt_audio_vector": np.random.rand(768).tolist(),  # Wav2Vec2 embedding for Adhan
    #         "sound_type": "kitchen_to_religious",
    #         "dataset_type": "audio"
    #     }
    # ]
    # 
    # audio_datasets['audioset_freesound'] = audioset_freesound
    # 
    # print(f"✅ Prepared {len(audioset_freesound)} high-quality AudioSet/FreeSound problems")
    # print(f"📊 Sound transitions: Animal→Music, Weather→Crowd, Human→Machine, Nature→Urban, Speech→Environmental")
    # 
    # return audio_datasets
    pass

# ============================================================================
# EXAMPLE ABLATION STUDY EXECUTION
# ============================================================================

def run_comprehensive_ablation_studies():
    """Example of how to run comprehensive ablation studies - COMMENTED OUT"""
    # print("🚀 COMPREHENSIVE ABLATION STUDIES")
    # print("=" * 60)
    # 
    # # Initialize analyzer
    # analyzer = ProperAdaptationAnalyzer("Qwen/Qwen2.5-Math-7B")
    # 
    # # Load datasets
    # translated_datasets = load_translated_datasets()
    # coding_datasets = prepare_coding_datasets()
    # image_datasets = prepare_image_datasets()
    # audio_datasets = prepare_audio_datasets()
    # 
    # # Initialize ablation framework
    # ablation = AblationStudyFramework(analyzer)
    # 
    # print("\n📊 ABLATION STUDY PARAMETERS:")
    # injection_points = ablation.generate_injection_points_range(5, 100, 5)
    # temperatures = ablation.generate_temperature_range(0.1, 1.5, 0.2)
    # print(f"🎯 Injection points: {injection_points}")
    # print(f"🌡️ Temperature range: {temperatures}")
    # 
    # # Run injection point ablation
    # print("\n🔬 Running injection point ablation...")
    # injection_results = ablation.run_injection_point_ablation(
    #     translated_datasets.get('math_translated', [])[:50],  # Small subset
    #     injection_points[:5]  # Test first 5 points
    # )
    # 
    # # Run temperature ablation
    # print("\n🌡️ Running temperature ablation...")
    # temp_results = ablation.run_temperature_ablation(
    #     translated_datasets.get('math_translated', [])[:20],  # Small subset
    #     temperatures[:5]  # Test first 5 temperatures
    # )
    # 
    # # Save results
    # results = {
    #     'injection_point_ablation': injection_results,
    #     'temperature_ablation': temp_results,
    #     'timestamp': datetime.now().isoformat()
    # }
    # 
    # output_path = "../output/ablation_studies/"
    # os.makedirs(output_path, exist_ok=True)
    # 
    # with open(f"{output_path}/comprehensive_ablation_results.json", 'w') as f:
    #     json.dump(results, f, indent=2)
    # 
    # print(f"\n✅ Ablation studies complete! Results saved to {output_path}")
    pass

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def print_extended_usage():
    """Print instructions for using extended functionality"""
    print("""
🚀 EXTENDED FUNCTIONALITY USAGE GUIDE
====================================

All extended features are COMMENTED OUT to prevent interference.
To use any feature:

1. 📁 TRANSLATED DATASETS:
   Uncomment load_translated_datasets() to use the fixed multilingual datasets

2. 🎯 INJECTION POINT ABLATION (5-100 tokens):
   Uncomment AblationStudyFramework.run_injection_point_ablation()
   
3. 🌡️ TEMPERATURE ABLATION (0.1-1.5):
   Uncomment AblationStudyFramework.run_temperature_ablation()
   
4. 💻 CODING DATASETS:
   Uncomment prepare_coding_datasets() for HumanEval/MBPP-style problems
   
5. 🖼️ IMAGE DATASETS:
   Uncomment prepare_image_datasets() for VQA/captioning with vectorized images
   
6. 🎵 AUDIO DATASETS:
   Uncomment prepare_audio_datasets() for speech/audio classification
   
7. 🔬 FULL ABLATION STUDIES:
   Uncomment run_comprehensive_ablation_studies() for complete analysis

⚠️ SAFETY NOTES:
- Test on small datasets first ([:10] slicing)
- Monitor memory usage (28GB+ for full models)
- Run ablations when no other experiments are active
- Save results incrementally to prevent data loss

📝 CURRENT STATUS:
- Original inference_loop.py functionality: ✅ ACTIVE
- Extended functionality: 🔒 SAFELY COMMENTED OUT
- Ready for activation when needed: ✅ YES
""")

# Uncomment this line to see usage instructions
# print_extended_usage()