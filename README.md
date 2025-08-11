# Adaptation Latency Measurement via KV-Cache Injection

## Project Overview
This project measures **adaptation latency** in large language models (LLMs) by injecting new context mid-generation using KV-cache manipulation. We evaluate how quickly models can adapt to context switches without restarting generation, using **token-level analysis** and **mechanistic interpretation** to understand adaptation dynamics.

## Key Innovation
- **True KV-Cache Injection**: Direct manipulation of the model's attention cache during generation
- **No Reprompting**: Model maintains internal state while adapting to new context
- **Multi-Dimensional Measurement**: KL divergence, attention weights, token probability ratios, and embedding similarity
- **Oracle Comparison**: Measures adaptation gap against ideal performance

## Models Used
- **Generation Model**: Qwen2.5-Math-7B (base model, not instruct)
- **Analysis Methods**: Token-level probability analysis, attention weight tracking, embedding similarity
- **Device Support**: CUDA, MPS (Apple Metal), CPU

## Datasets
1. **Math Dataset** (MAWPS)
   - Location: `experiments/changed_ds/math/mawps_augmented.json`
   - **Size**: 500 problems with balanced variation distribution
   - **Structure**: 24% true controls (identical), 76% calculation changes
   - **Variations**: Single number change (25%), multiple numbers (25%), added steps (25%)
   - **Format**: Original problems with alternative versions, variation tracking, and equations
   
2. **Non-Math Dataset** (Semantic QA)
   - Location: `experiments/changed_ds/non_math/semantic_ds_adapt_latency_500.json`
   - Format: Q&A pairs with alternative questions for context switching

## Methodology

### 1. Generation Process
- Token-by-token generation tracking probability distributions
- Monitor attention weights across all layers and heads
- Maintain KV-cache state throughout injection process

### 2. Context Injection
- After N tokens (configurable: 5, 10, 15 tokens)
- Inject alternative problem/context directly into KV-cache
- Model must adapt without restarting generation

### 3. Multi-Dimensional Analysis
- **KL Divergence**: Measures distribution shift between original and injected contexts
- **Attention Tracking**: Monitors where model "looks" (pre vs post injection tokens)
- **Token Probability Ratios**: P(token|new_context) / P(token|old_context)
- **Embedding Similarity**: Semantic drift in representation space

### 4. Causal Experiments
- **Oracle**: Model with new context from start (gold standard)
- **Injection**: Our experimental condition (mid-generation switch)
- **Baseline**: Original context only
- **Noise Control**: Random injection to test specificity
- **Adaptation Gap**: Distance from injection performance to oracle performance

### 5. Key Metrics
- **Detection Latency**: Tokens until distribution shift detected
- **Adaptation Point**: When attention/probabilities switch to new context
- **Stabilization**: Tokens until adaptation behavior becomes consistent
- **Oracle Convergence**: How close injection gets to ideal oracle performance

## Project Structure
```
experiments/
├── prm_results/
│   ├── run_proper_analysis.py         # Main analysis with all measurement methods
│   └── real_results/
│       ├── inference_loop_math.py      # Original PRM-based approach (deprecated)
│       └── all_results.json           # Results from original experiments
├── changed_ds/
│   ├── math/
│   │   ├── mawps_augmented.json       # Math dataset with alternatives
│   │   └── mawps_augment.py           # Dataset augmentation script
│   └── non_math/
│       ├── semantic_ds_adapt_latency_500.json  # Non-math dataset
│       └── semantic_augment.py        # Dataset augmentation script
└── training/
    └── training.py                     # Training utilities
```

## Installation

```bash
# Install required packages
pip install transformers torch accelerate
pip install bitsandbytes  # For 4-bit quantization on CUDA (optional)

# Models will be auto-downloaded on first run (~14GB each)
# Cached at: ~/.cache/huggingface/hub
```

## Usage

### Run Proper Analysis
```bash
cd experiments/prm_results/
python3 run_proper_analysis.py --dataset ../changed_ds/math/mawps_augmented.json --num_problems 10
```

### Configuration Options
- `--model`: Generation model (default: Qwen2.5-Math-7B)
- `--dataset`: Path to dataset JSON file
- `--num_problems`: Number of problems to analyze (default: all)
- `--injection_points`: Token positions for injection (default: [5, 10, 15])
- `--output_dir`: Output directory for results (default: proper_results)
- `--device`: Device to use (cuda/mps/cpu, auto-detected)

### Single Problem Test
```bash
python3 run_proper_analysis.py --single
```

## Output Format

Results are saved with comprehensive multi-dimensional metrics:
```json
{
  "problem_index": 0,
  "original_problem": "Sarah has 5 apples...",
  "alternative_problem": "Sarah has 7 apples...",
  "timestamp": "2025-08-10T...",
  "model": "Qwen/Qwen2.5-Math-7B",
  
  "token_analysis": {
    "injection_5": {
      "summary": {
        "adapted": true,
        "adaptation_latency_tokens": 3,
        "adaptation_quality": 0.78,
        "assessment": "good"
      }
    }
  },
  
  "attention_analysis": {
    "injection_5": {
      "stabilization_tokens": 7,
      "attention_shift_magnitude": 0.45,
      "final_attention_ratio": 1.8
    }
  },
  
  "causal_analysis": {
    "overall": {
      "mean_adaptation": 0.73,
      "optimal_injection": 5
    },
    "recommendations": [
      "Good adaptation capability...",
      "Optimal injection point: after 5 tokens..."
    ]
  },
  
  "unified_summary": {
    "adaptation_detected": true,
    "adaptation_quality": "good",
    "best_injection_point": "injection_5",
    "key_findings": [...]
  }
}
```

## Key Features
- ✅ True KV-cache manipulation (no reprompting)  
- ✅ Local model execution (no rate limits)
- ✅ Token-level granularity analysis
- ✅ Multi-dimensional measurement (KL divergence, attention, embeddings)
- ✅ Causal experiments with proper controls (oracle, noise, baseline)
- ✅ Attention weight tracking across layers
- ✅ Statistical significance testing
- ✅ Comprehensive adaptation curves visualization
- ✅ Mechanistic interpretability
- ✅ Balanced dataset with controlled variations

## Current Status
- ✅ **Proper measurement system implemented** in `run_proper_analysis.py`
- ✅ Token-level adaptation tracking with probability ratios
- ✅ Attention weight redistribution measurement  
- ✅ KL divergence and embedding similarity analysis
- ✅ Causal experiments with oracle comparison
- ✅ Multi-dimensional adaptation metrics
- ✅ Unified analysis coordinating all measurement methods
- ⚠️ Legacy PRM-based approach deprecated but preserved for comparison

## Performance Notes
- Models require ~14GB disk space each (cached after first download)
- GPU recommended for reasonable inference speed
- MPS (Apple Metal) supported for Mac users
- Full precision uses ~28GB VRAM, quantization reduces to ~7GB

## Current Results
✅ **Working PRM System**: Fixed critical bug where PRM was returning all zeros
- **Results Location**: `experiments/prm_results/real_results/all_results.json`
- **Sample Results**: PRM scores now range from -1 to +1 with meaningful discrimination
- **Validation**: Good reasoning gets positive scores (0.89), bad reasoning gets negative (-0.55)
- **Experiments**: 5 completed experiments with unique problems (no repetition)

## Recent Fixes (2025-08-09)
1. **PRM Scoring Bug**: Fixed incorrect model loading and scoring logic
2. **Dataset Repetition**: Each injection experiment now uses unique problems
3. **Binary Classification**: Properly implemented P(good) - P(bad) scoring method
4. **Inference Scale**: Fixed problems_per_experiment from 2 to 167 to process all 500 problems
5. **Answer Extraction**: Added regex-based final answer extraction from model outputs
6. **Dataset Quality**: Complete rewrite of mawps_augment.py with focused calculation changes
7. **Accuracy Tracking**: Added comprehensive pre/post injection correctness measurement
8. **Variation Balance**: Achieved true 80/20 split with 3 balanced variation types
9. **Physical Constraints**: Fixed negative answer issues for problems involving physical objects

## Future Work
- Implement non-math inference loop
- Add visualization for adaptation curves  
- Experiment with different injection strategies
- Compare adaptation across model families
- Statistical analysis of adaptation patterns

## Contact
For questions or issues, please open an issue in the repository.