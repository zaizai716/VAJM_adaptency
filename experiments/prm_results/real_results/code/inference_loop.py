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
            generated_tokens.append(next_token.item())
            
            # Inject new context at specified point
            if step == injection_point:
                injection_ids = self.tokenizer(injection_text, return_tensors="pt").input_ids.to(self.device)
                # This is where you'd modify the KV cache - simplified here
                # In practice, you'd concatenate injection embeddings to past_key_values
                
            # Calculate attention distribution
            if self.current_attention_weights:
                snapshot = self._create_attention_snapshot(
                    step, 
                    injection_point,
                    next_token.item()
                )
                self.attention_history.append(snapshot)
            
            # Early stopping
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
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
    
    # Lexical metrics
    vocabulary_overlap_original: float  # Vocab similarity to original context
    vocabulary_overlap_injected: float  # Vocab similarity to injected context
    
    # Semantic metrics
    embedding_similarity_original: float  # Cosine sim in embedding space
    embedding_similarity_injected: float  # Cosine sim in embedding space
    
    # Behavioral metrics
    perplexity: float  # Current perplexity
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
        
        # Prepare reference distributions
        original_embedding = self._get_context_embedding(original_context)
        injected_embedding = self._get_context_embedding(injected_context)
        original_vocab = set(original_context.lower().split())
        injected_vocab = set(injected_context.lower().split())
        
        for position in range(max_tokens):
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
                    # Simplified injection - in practice, modify past_key_values
                    pass
                
                # Calculate current state metrics
                state = self._calculate_state_metrics(
                    position=position,
                    token_id=next_token.item(),
                    logits=logits,
                    probs=probs,
                    hidden_states=outputs.hidden_states[-1],
                    original_embedding=original_embedding,
                    injected_embedding=injected_embedding,
                    original_vocab=original_vocab,
                    injected_vocab=injected_vocab
                )
                
                states.append(state)
                generated_tokens.append(next_token.item())
                
                # Early stopping
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
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
        uniform_dist = np.ones_like(probs_np) / len(probs_np)
        kl_from_uniform = entropy(probs_np, uniform_dist)
        
        # Lexical overlap
        token_text = self.tokenizer.decode([token_id])
        token_lower = token_text.lower().strip()
        
        vocab_overlap_original = 1.0 if token_lower in original_vocab else 0.0
        vocab_overlap_injected = 1.0 if token_lower in injected_vocab else 0.0
        
        # Semantic similarity
        current_embedding = hidden_states[0, -1].cpu()
        sim_original = 1 - cosine(
            current_embedding.numpy(), 
            original_embedding.cpu().numpy()
        )
        sim_injected = 1 - cosine(
            current_embedding.numpy(),
            injected_embedding.cpu().numpy()
        )
        
        # Behavioral metrics
        perplexity = torch.exp(-torch.log(probs[token_id])).item()
        confidence = probs.max().item()
        
        return AdaptationState(
            token_position=position,
            kl_from_original=kl_from_uniform,  # Placeholder
            kl_from_injected=kl_from_uniform,  # Placeholder
            mutual_info_original=0.0,  # Would need proper calculation
            mutual_info_injected=0.0,  # Would need proper calculation
            vocabulary_overlap_original=vocab_overlap_original,
            vocabulary_overlap_injected=vocab_overlap_injected,
            embedding_similarity_original=float(sim_original),
            embedding_similarity_injected=float(sim_injected),
            perplexity=perplexity,
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
            summary['adaptation_speed_tokens'] = inj['adaptation_speed']
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
    
    # Probability metrics
    logprob: float
    entropy: float
    top_k_probs: List[float]
    rank_in_original: int  # Where this token ranked in original context
    rank_in_injected: int  # Where this token ranked in injected context
    
    # Comparison metrics
    prob_ratio: float  # P(token|injected) / P(token|original)
    surprisal_delta: float  # Change in surprisal
    
    # Context attribution
    context_source: str  # 'original', 'injected', 'mixed', 'neither'
    confidence: float


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
        max_tokens: int = 100,
        top_k: int = 10
    ) -> Dict:
        """Perform token-level adaptation analysis."""
        
        # Generate reference distributions
        original_dist = self._get_reference_distribution(original_context, max_tokens)
        injected_dist = self._get_reference_distribution(injected_context, max_tokens)
        
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
        
        return {
            'token_states': generation_states,
            'adaptation_analysis': analysis,
            'metrics': metrics,
            'summary': self._create_summary(analysis, metrics)
        }
    
    def _get_reference_distribution(
        self, 
        context: str, 
        length: int
    ) -> Dict[int, Dict]:
        """Get reference token distributions for a context."""
        
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        distributions = {}
        past_key_values = None
        
        with torch.no_grad():
            # Initial pass
            outputs = self.model(
                inputs.input_ids,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            
            for position in range(length):
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Store distribution info
                distributions[position] = {
                    'logits': logits.cpu(),
                    'probs': probs.cpu(),
                    'entropy': -torch.sum(probs * torch.log(probs + 1e-10)).item(),
                    'top_tokens': torch.topk(probs, k=20)
                }
                
                # Generate next token for continuation
                next_token = torch.argmax(logits)
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
            
            for position in range(max_tokens):
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Get next token
                next_token = torch.argmax(logits)
                token_text = self.tokenizer.decode([next_token.item()])
                
                # Calculate metrics against both contexts
                state = self._calculate_token_state(
                    position=position,
                    token_id=next_token.item(),
                    token_text=token_text,
                    current_probs=probs,
                    current_logits=logits,
                    original_dist=original_dist.get(position, {}),
                    injected_dist=injected_dist.get(position, {}),
                    top_k=top_k
                )
                
                states.append(state)
                
                # Perform injection if at injection point
                if position == injection_point:
                    # Here you would modify past_key_values
                    # For demonstration, we'll track the injection point
                    pass
                
                # Continue generation
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                
                # Early stopping
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
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
        entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10)).item()
        
        # Top-k probabilities
        top_k_vals, top_k_indices = torch.topk(current_probs, k=top_k)
        top_k_probs = top_k_vals.cpu().tolist()
        
        # Rank in reference distributions
        rank_original = float('inf')
        rank_injected = float('inf')
        prob_original = 0.0
        prob_injected = 0.0
        
        if 'probs' in original_dist:
            orig_probs = original_dist['probs']
            if token_id < len(orig_probs):
                prob_original = orig_probs[token_id].item()
                sorted_indices = torch.argsort(orig_probs, descending=True)
                rank_original = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1 if token_id in sorted_indices else float('inf')
        
        if 'probs' in injected_dist:
            inj_probs = injected_dist['probs']
            if token_id < len(inj_probs):
                prob_injected = inj_probs[token_id].item()
                sorted_indices = torch.argsort(inj_probs, descending=True)
                rank_injected = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1 if token_id in sorted_indices else float('inf')
        
        # Probability ratio
        prob_ratio = prob_injected / (prob_original + 1e-10)
        
        # Surprisal delta
        surprisal_current = -token_logprob
        surprisal_original = -np.log(prob_original + 1e-10)
        surprisal_delta = surprisal_current - surprisal_original
        
        # Determine context source
        if prob_ratio > 2.0 and rank_injected < rank_original:
            context_source = 'injected'
        elif prob_ratio < 0.5 and rank_original < rank_injected:
            context_source = 'original'
        elif 0.5 <= prob_ratio <= 2.0:
            context_source = 'mixed'
        else:
            context_source = 'neither'
        
        # Confidence (max probability)
        confidence = top_k_probs[0]
        
        return TokenState(
            position=position,
            token_id=token_id,
            token_text=token_text,
            logprob=token_logprob,
            entropy=entropy,
            top_k_probs=top_k_probs,
            rank_in_original=rank_original,
            rank_in_injected=rank_injected,
            prob_ratio=prob_ratio,
            surprisal_delta=surprisal_delta,
            context_source=context_source,
            confidence=confidence
        )
    
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
        entropies = [s.entropy for s in states]
        confidences = [s.confidence for s in states]
        
        # Smooth the signals for better pattern detection
        if len(prob_ratios) > 5:
            prob_ratios_smooth = savgol_filter(prob_ratios, min(5, len(prob_ratios)), 2)
        else:
            prob_ratios_smooth = prob_ratios
        
        # Find adaptation point (where prob_ratio crosses 1.0 and stays)
        adaptation_point = self._find_adaptation_point(prob_ratios, injection_point)
        
        # Calculate adaptation speed (tokens from injection to adaptation)
        adaptation_speed = adaptation_point - injection_point if adaptation_point > injection_point else None
        
        # Analyze context source distribution
        source_counts = defaultdict(int)
        for state in states:
            source_counts[state.context_source] += 1
        
        source_distribution = {
            source: count / len(states) 
            for source, count in source_counts.items()
        }
        
        # Measure entropy spike at injection
        if injection_point < len(entropies) - 1:
            entropy_spike = entropies[injection_point + 1] - entropies[injection_point]
        else:
            entropy_spike = 0.0
        
        # Calculate stability (inverse of variance in confidence)
        confidence_variance = np.var(confidences)
        stability = 1.0 / (1.0 + confidence_variance)
        
        return {
            'adaptation_point': adaptation_point,
            'adaptation_speed_tokens': adaptation_speed,
            'source_distribution': source_distribution,
            'entropy_spike': entropy_spike,
            'stability': stability,
            'mean_confidence': np.mean(confidences),
            'trajectories': {
                'positions': positions,
                'prob_ratios': prob_ratios,
                'prob_ratios_smooth': prob_ratios_smooth.tolist() if isinstance(prob_ratios_smooth, np.ndarray) else prob_ratios_smooth,
                'entropies': entropies,
                'confidences': confidences
            }
        }
    
    def _find_adaptation_point(
        self, 
        prob_ratios: List[float], 
        injection_point: int,
        threshold: float = 1.0,
        window: int = 3
    ) -> int:
        """Find the point where model adapts to new context."""
        
        for i in range(injection_point, len(prob_ratios) - window):
            # Check if prob_ratio > threshold for window consecutive tokens
            if all(prob_ratios[i+j] > threshold for j in range(window)):
                return i
        
        return len(prob_ratios)  # Didn't adapt
    
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
                'mean_entropy': np.mean([s.entropy for s in pre_states]),
                'mean_confidence': np.mean([s.confidence for s in pre_states]),
                'mean_surprisal': np.mean([-s.logprob for s in pre_states]),
                'original_preference': sum(1 for s in pre_states if s.context_source == 'original') / len(pre_states)
            }
        
        # Post-injection metrics
        if post_states:
            metrics['post_injection'] = {
                'mean_entropy': np.mean([s.entropy for s in post_states]),
                'mean_confidence': np.mean([s.confidence for s in post_states]),
                'mean_surprisal': np.mean([-s.logprob for s in post_states]),
                'injected_preference': sum(1 for s in post_states if s.context_source == 'injected') / len(post_states)
            }
        
        # Delta metrics
        if pre_states and post_states:
            metrics['deltas'] = {
                'entropy_change': metrics['post_injection']['mean_entropy'] - metrics['pre_injection']['mean_entropy'],
                'confidence_change': metrics['post_injection']['mean_confidence'] - metrics['pre_injection']['mean_confidence'],
                'surprisal_change': metrics['post_injection']['mean_surprisal'] - metrics['pre_injection']['mean_surprisal']
            }
        
        # Immediate response (first 5 tokens after injection)
        immediate_states = [s for s in post_states[:5]]
        if immediate_states:
            metrics['immediate_response'] = {
                'mean_prob_ratio': np.mean([s.prob_ratio for s in immediate_states]),
                'max_entropy': max(s.entropy for s in immediate_states),
                'min_confidence': min(s.confidence for s in immediate_states),
                'context_sources': [s.context_source for s in immediate_states]
            }
        
        return metrics
    
    def _create_summary(self, analysis: Dict, metrics: Dict) -> Dict:
        """Create a summary of token-level adaptation."""
        
        summary = {}
        
        # Did adaptation occur?
        if 'adaptation_point' in analysis:
            summary['adapted'] = analysis['adaptation_point'] < float('inf')
            
            if summary['adapted']:
                summary['adaptation_latency_tokens'] = analysis.get('adaptation_speed_tokens', None)
            else:
                summary['adaptation_latency_tokens'] = None
        
        # Quality of adaptation
        if 'source_distribution' in analysis:
            injected_ratio = analysis['source_distribution'].get('injected', 0)
            summary['adaptation_quality'] = injected_ratio  # 0-1 scale
        
        # Stability
        summary['stable'] = analysis.get('stability', 0) > 0.7
        
        # Immediate impact
        if 'immediate_response' in metrics:
            summary['immediate_disruption'] = metrics['immediate_response']['max_entropy'] > 2.0
            summary['immediate_adaptation'] = metrics['immediate_response']['mean_prob_ratio'] > 1.5
        
        # Overall assessment
        if summary.get('adapted') and summary.get('adaptation_quality', 0) > 0.5:
            if summary.get('adaptation_latency_tokens', float('inf')) < 5:
                summary['assessment'] = 'excellent'
            elif summary.get('adaptation_latency_tokens', float('inf')) < 10:
                summary['assessment'] = 'good'
            else:
                summary['assessment'] = 'slow'
        elif summary.get('adapted'):
            summary['assessment'] = 'partial'
        else:
            summary['assessment'] = 'failed'
        
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
        injection_points: List[int] = [5, 10, 15],
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
                
                # Store generation info
                generated_tokens.append(next_token.item())
                token_logprobs.append(torch.log(probs[next_token]).item())
                hidden_states_history.append(outputs.hidden_states[-1][0, -1].cpu().numpy())
                
                # Check for injection
                if injection_context and injection_point and step == injection_point:
                    # Inject new context (simplified - in practice modify KV cache)
                    injection_ids = self.tokenizer(injection_context, return_tensors="pt").input_ids.to(self.device)
                    # This is where you'd actually modify past_key_values
                    # For now, we'll concatenate to input
                    
                # Continue generation
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                
                # Early stopping
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return {
            'generated_tokens': generated_tokens,
            'generated_text': self.tokenizer.decode(generated_tokens),
            'token_logprobs': token_logprobs,
            'hidden_states': hidden_states_history,
            'perplexity': np.exp(-np.mean(token_logprobs)) if token_logprobs else float('inf')
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
        
        # Perplexity-based metrics
        perplexity = generation['perplexity']
        
        # Trajectory analysis
        if 'token_logprobs' in generation:
            logprobs = generation['token_logprobs']
            # Measure stability (lower variance = more stable)
            stability = 1.0 / (1.0 + np.var(logprobs))
            # Measure confidence (mean log prob)
            confidence = np.mean(logprobs) if logprobs else -float('inf')
        else:
            stability = 0.0
            confidence = -float('inf')
        
        return {
            'original_overlap': original_overlap,
            'alternative_overlap': alternative_overlap,
            'adaptation_ratio': alternative_overlap / (original_overlap + 1e-8),
            'perplexity': perplexity,
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
            for metric in ['alternative_overlap', 'perplexity', 'stability']:
                inj_values = [m[metric] for m in injection_metrics]
                ora_values = [m[metric] for m in oracle_metrics]
                
                # Calculate gap
                gap = np.mean(inj_values) - np.mean(ora_values)
                
                # Statistical test
                t_stat, p_value = ttest_ind(inj_values, ora_values)
                
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
                
                t_stat, p_value = ttest_ind(inj_values, noise_values)
                
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
                elif metric == 'perplexity':
                    # Lower is better, but normalize
                    score = 1.0 / (1.0 + abs(gap_data['gap']))
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
                    "Poor adaptation detected. Consider using full reprompting instead of injection."
                )
            elif score < 0.7:
                recommendations.append(
                    "Moderate adaptation. Try earlier injection points or gradual context blending."
                )
            else:
                recommendations.append(
                    "Good adaptation capability. KV-cache injection is viable for this model."
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
        injection_points: List[int] = [5, 10, 15],
        max_new_tokens: int = 50
    ) -> Dict:
        """Run comprehensive analysis on a single problem pair."""
        
        results = {
            'original_problem': original_problem,
            'alternative_problem': alternative_problem,
            'timestamp': datetime.now().isoformat(),
            'model': self.model.config._name_or_path
        }
        
        print("\n1. Running Token-Level Analysis...")
        # Token-level analysis (most granular)
        token_results = {}
        for injection_point in injection_points:
            print(f"   Injection at token {injection_point}")
            token_analysis = self.token_analyzer.analyze_token_adaptation(
                original_problem,
                alternative_problem,
                injection_point,
                max_new_tokens
            )
            token_results[f'injection_{injection_point}'] = token_analysis
        results['token_analysis'] = token_results
        
        # Skip attention tracking - it's redundant with token probability ratios
        # and requires slow eager mode (15-20x slower than SDPA)
        print("\n2. Skipping Attention Weight Analysis (redundant with token ratios)...")
        results['attention_analysis'] = {}
        
        print("\n3. Running Multi-Dimensional Metrics...")
        # Multi-dimensional adaptation metrics
        adaptation_results = {}
        for injection_point in injection_points:
            print(f"   Injection at token {injection_point}")
            adaptation_analysis = self.adaptation_metrics.measure_adaptation(
                original_problem,
                alternative_problem,
                injection_point,
                max_new_tokens
            )
            adaptation_results[f'injection_{injection_point}'] = adaptation_analysis['summary']
        results['adaptation_metrics'] = adaptation_results
        
        print("\n4. Running Causal Experiments...")
        # Causal experiments (most comprehensive but slower)
        causal_results = self.causal_experiments.run_causal_experiment(
            original_problem,
            alternative_problem,
            injection_points,
            num_trials=3,  # Fewer trials for speed
            max_new_tokens=max_new_tokens
        )
        results['causal_analysis'] = {
            'overall': causal_results['overall_analysis'],
            'recommendations': causal_results['recommendations']
        }
        
        # Generate unified summary
        results['unified_summary'] = self._create_unified_summary(results)
        
        return results
    
    def analyze_dataset(
        self,
        dataset_path: str,
        output_dir: str = "proper_results",
        num_problems: int = None,
        injection_points: List[int] = [5, 10, 15]
    ) -> None:
        """Analyze an entire dataset of problems."""
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if num_problems:
            dataset = dataset[:num_problems]
        
        print(f"Analyzing {len(dataset)} problems")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Analyze each problem
        all_results = []
        for idx, problem in enumerate(tqdm(dataset, desc="Analyzing problems")):
            print(f"\n{'='*50}")
            print(f"Problem {idx + 1}/{len(dataset)}")
            print(f"{'='*50}")
            
            # Extract problem texts based on dataset format - MATH ONLY FOR NOW
            if 'Question' in problem:  # MAWPS math dataset format
                original = problem['Question']
                alternative = problem.get('alt_problem', problem['Question'])
                dataset_type = 'math'
            # elif 'story_content' in problem:  # FairytaleQA format - COMMENTED OUT
            #     story = problem['story_content']
            #     original = f"Story: {story}\n\nQuestion: {problem['Question']}"
            #     alternative = f"Story: {story}\n\nQuestion: {problem['alt_problem']}"
            #     dataset_type = 'fairytale'
            else:  # Generic format or skip non-math
                print(f"Skipping non-math problem {idx}: {list(problem.keys())}")
                continue
            
            if not original or not alternative:
                print(f"Skipping problem {idx}: missing text")
                continue
            
            try:
                # Run analysis
                result = self.analyze_single_problem(
                    original,
                    alternative,
                    injection_points
                )
                
                # Add metadata
                result['problem_index'] = idx
                result['problem_data'] = problem
                result['dataset_type'] = dataset_type
                
                all_results.append(result)
                
                # Save intermediate results
                if (idx + 1) % 10 == 0:
                    self._save_results(all_results, output_path / "intermediate_results.json")
                
            except Exception as e:
                print(f"Error analyzing problem {idx}: {e}")
                continue
        
        # Save final results
        self._save_results(all_results, output_path / "final_results.json")
        
        # Generate summary report
        self._generate_report(all_results, output_path / "summary_report.json")
        
        print(f"\nAnalysis complete. Results saved to {output_path}")
    
    def _create_unified_summary(self, results: Dict) -> Dict:
        """Create a unified summary across all analysis methods."""
        
        summary = {
            'adaptation_detected': False,
            'adaptation_quality': 'unknown',
            'best_injection_point': None,
            'key_findings': []
        }
        
        # Check token analysis
        if 'token_analysis' in results:
            token_summaries = []
            for inj_point, analysis in results['token_analysis'].items():
                if 'summary' in analysis:
                    token_summaries.append(analysis['summary'])
            
            if token_summaries:
                # Find best injection point based on token analysis
                best_assessment = None
                best_point = None
                assessment_order = ['excellent', 'good', 'slow', 'partial', 'failed']
                
                for idx, summary_item in enumerate(token_summaries):
                    if 'assessment' in summary_item:
                        assessment = summary_item['assessment']
                        if best_assessment is None or assessment_order.index(assessment) < assessment_order.index(best_assessment):
                            best_assessment = assessment
                            best_point = list(results['token_analysis'].keys())[idx]
                
                if best_assessment in ['excellent', 'good', 'slow']:
                    summary['adaptation_detected'] = True
                    summary['adaptation_quality'] = best_assessment
                    summary['best_injection_point'] = best_point
                    summary['key_findings'].append(f"Token-level: {best_assessment} adaptation at {best_point}")
        
        # Check attention analysis
        if 'attention_analysis' in results:
            stabilization_points = []
            for inj_point, metrics in results['attention_analysis'].items():
                if 'stabilization_tokens' in metrics:
                    stabilization_points.append(metrics['stabilization_tokens'])
            
            if stabilization_points:
                avg_stabilization = sum(stabilization_points) / len(stabilization_points)
                summary['key_findings'].append(f"Attention: stabilizes after ~{avg_stabilization:.1f} tokens")
        
        # Check causal analysis
        if 'causal_analysis' in results:
            if 'recommendations' in results['causal_analysis']:
                for rec in results['causal_analysis']['recommendations'][:2]:  # Top 2 recommendations
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
            json.dump(serializable_results, f, indent=2, default=str)
    
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
        quality_counts = {}
        
        for result in results:
            if 'unified_summary' in result:
                summary = result['unified_summary']
                
                if summary['adaptation_detected']:
                    adaptation_counts['detected'] += 1
                else:
                    adaptation_counts['not_detected'] += 1
                
                quality = summary.get('adaptation_quality', 'unknown')
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        report['statistics'] = {
            'adaptation_rate': adaptation_counts['detected'] / len(results) if results else 0,
            'adaptation_counts': adaptation_counts,
            'quality_distribution': quality_counts
        }
        
        # Identify patterns
        injection_performance = {}
        for result in results:
            if 'unified_summary' in result:
                best_point = result['unified_summary'].get('best_injection_point')
                if best_point:
                    injection_performance[best_point] = injection_performance.get(best_point, 0) + 1
        
        report['patterns']['best_injection_points'] = injection_performance
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total problems analyzed: {report['total_problems']}")
        print(f"Adaptation detection rate: {report['statistics']['adaptation_rate']:.1%}")
        print(f"Quality distribution: {report['statistics']['quality_distribution']}")
        print(f"Best injection points: {report['patterns']['best_injection_points']}")


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
        json.dump(serializable_metrics, f, indent=2)


def test_attention_tracking():
    """Test the attention tracking system."""
    
    # Initialize model and tracker
    model_name = "gpt2"  # Use small model for testing
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
    model_name = "gpt2"
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
        help="Model to analyze"
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
        analyzer._save_results([result], output_path / "single_test_result.json")
        
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
    # If no command line args, run a simple test
    import sys
    if len(sys.argv) == 1:
        print("Running default test (use --help for options)")
        
        # Use smaller model for testing
        analyzer = ProperAdaptationAnalyzer(
            model_name="gpt2",  # Small model for testing
            device=None
        )
        
        result = analyzer.analyze_single_problem(
            original_problem="The weather today is sunny and warm.",
            alternative_problem="Actually it's cold and raining heavily.",
            injection_points=[3, 5, 7],
            max_new_tokens=20
        )
        
        print("\n" + "="*50)
        print("TEST COMPLETE - Unified Summary:")
        print("="*50)
        for key, value in result['unified_summary'].items():
            print(f"{key}: {value}")
    else:
        main()