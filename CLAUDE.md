# CLAUDE.md - Session Context & Progress Tracking

## Last Updated: 2025-08-09

## Project Summary
Measuring **adaptation latency** in LLMs by injecting context mid-generation via KV-cache manipulation. Using **token-level analysis**, **attention weight tracking**, and **multi-dimensional metrics** (KL divergence, embedding similarity) to evaluate how quickly models adapt to context switches.

**Core Research Question**: How many tokens does it take for an LLM to adapt to a mid-generation context switch when using KV-cache injection, measured via mechanistic analysis rather than output quality?

## Critical Session Context

### Working Directory
**Project Root**: `/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/`
**Previous Session Working Dir**: `/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/prm_results/real_results/`

### Python Environment
- Python Version: 3.12.6
- Device: MPS (Apple Metal GPU) available
- Platform: macOS Darwin 24.6.0

### Key Files & Their Status
- ✅ `experiments/prm_results/run_proper_analysis.py` - COMPLETE multi-dimensional analysis system (92KB combined file)
- ✅ `experiments/prm_results/real_results/inference_loop_math.py` - LEGACY PRM-based approach (deprecated)
- ✅ `experiments/prm_results/real_results/all_results.json` - Results from legacy PRM experiments
- ✅ `experiments/changed_ds/math/mawps_augment.py` - COMPLETE dataset generator with 3 variation types + controls
- ✅ `experiments/changed_ds/math/mawps_augmented.json` - 500 problems with balanced variations (24% controls, 76% changes)
- ✅ `README.md` - Updated with proper measurement methodology
- ✅ `CLAUDE.md` - This file (updated to reflect new approach)

### Model Configuration
```python
GEN_MODEL = "Qwen/Qwen2.5-Math-7B"  # Base model (not instruct)
# No PRM needed - using mechanistic analysis instead
CACHE_DIR = "~/.cache/huggingface/hub"
USE_QUANTIZATION = False  # Full precision for accurate measurements
```

### Dataset Paths
```python
MAWPS_FILE = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/math/mawps_augmented.json"
SEMANTIC_FILE = "/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/experiments/changed_ds/non_math/semantic_ds_adapt_latency_500.json"
```

## Implementation Details

### Multi-Dimensional Analysis System (run_proper_analysis.py)
```python
class ProperAdaptationAnalyzer:
    # Key components:
    # 1. AttentionTracker - monitors attention weight redistribution
    # 2. TokenLevelAnalyzer - tracks probability ratios at each token
    # 3. AdaptationMetrics - KL divergence and embedding similarity
    # 4. CausalExperiments - oracle vs injection comparison
    # 5. Unified analysis coordinating all methods
```

### Token-Level Measurement
- **Probability Ratios**: P(token|injected_context) / P(token|original_context)  
- **Context Attribution**: Classifies each token as 'original', 'injected', 'mixed', 'neither'
- **Adaptation Point**: First token where ratio > 1.0 and stays consistent
- **Granularity**: Token-level (not sentence-level) for precision

### Attention Weight Tracking
- Registers hooks on all transformer layers to capture attention weights
- Measures attention ratio: post_injection_sum / pre_injection_sum
- Tracks stabilization point where variance drops below threshold
- Shows WHERE model is "looking" during adaptation

### Causal Isolation
- **Oracle**: Model with new context from start (gold standard)
- **Injection**: Mid-generation context switch (experimental condition) 
- **Adaptation Gap**: Statistical distance between injection and oracle performance
- **Controls**: Noise injection, baseline, repetition to isolate effects

### Dataset Generation (mawps_augment.py)
- **3 Variation Types**: Single number change (26.7%), multiple number change (26.7%), added step (26.7%)
- **True Controls**: 20% identical problems (variation_type = null)
- **Physical Constraints**: Ensures non-negative answers for physical objects
- **Equation Tracking**: Maintains separate equations for original and modified problems

### Key Metrics Calculated
- **Detection Latency**: Tokens until KL divergence threshold crossed
- **Adaptation Point**: Token where probability ratios favor new context consistently  
- **Stabilization**: Tokens until attention/probability patterns stabilize
- **Oracle Convergence**: How closely injection performance matches oracle
- **Multi-Scale Analysis**: Immediate (1-5 tokens), short-term (5-20), long-term (20+)
- **Statistical Validation**: T-tests comparing injection vs oracle vs controls

## Project Structure Overview

### Complete Directory Tree
```
VAJM_adaptency/
├── CLAUDE.md                           # Session tracking (moved from experiments/)
├── README.md                          # Main project documentation  
├── requirements.txt                   # Top-level dependencies
├── experiments/
│   ├── changed_ds/                    # Augmented datasets for experiments
│   │   ├── math/
│   │   │   ├── mawps_augment.py      # Script to create alternative problems
│   │   │   └── mawps_augmented.json  # Math dataset with alternatives (500 problems)
│   │   └── non_math/
│   │       ├── semantic_augment.py   # Script for non-math alternatives
│   │       ├── semantic_ds.json      # Original semantic dataset
│   │       └── semantic_ds_adapt_latency_500.json  # Semantic with alternatives
│   ├── prm_results/                   # Main inference experiments
│   │   ├── requirements.txt           # Experiment-specific deps
│   │   ├── proof_concept/
│   │   │   └── full_distribution_trace.json  # Earlier experiment results
│   │   └── real_results/              # Current working directory
│   │       ├── inference_loop_math.py     # COMPLETE (773 lines) - main math inference
│   │       ├── inference_loop_nonmath.py  # EMPTY - needs implementation
│   │       └── all_results.json           # Experiment outputs
│   └── training/
│       ├── training.py                # PRM training scripts
│       └── training_metrics/          # Training evaluation results
├── preprocessed/                      # Preprocessed training data
│   ├── helpsteer3/                   # HelpSteer 3.0 dataset preprocessing
│   │   ├── preprocess2.py
│   │   ├── preprocessed_helpsteer3_train.jsonl
│   │   └── preprocessed_helpsteer3_validation.jsonl
│   └── prm800k/                      # PRM800K dataset preprocessing
│       ├── preprocess.py
│       ├── phase1_test.jsonl
│       ├── phase1_train.jsonl
│       ├── phase2_test.jsonl
│       └── phase2_train.jsonl
└── training/                          # External PRM training code
    └── prm800k/                      # Original PRM800K repository
        ├── LICENSE
        ├── README.md
        └── prm800k/                  # Core training modules
            ├── data/                 # Training datasets
            ├── eval/                 # Evaluation scripts
            ├── grading/              # Automated grading
            └── instructions/         # Task instructions PDFs
```

## Recent Changes & Session History

### Session 1 (2025-08-09 Morning)
- ✅ Analyzed project structure and codebase
- ✅ Understood KV-cache injection methodology
- ✅ Reviewed PRM scoring implementation
- ✅ Created comprehensive README.md
- ✅ Created this CLAUDE.md for session continuity

### Session 2 (2025-08-09 Afternoon) 
- ✅ Read entire project structure and CLAUDE.md
- ✅ Moved CLAUDE.md to root directory for better accessibility
- ✅ Updated CLAUDE.md with comprehensive project overview
- ✅ Identified key files and their implementation status

### Session 3 (2025-08-09 Evening)
- ✅ **MAJOR FIX**: Completely fixed PRM scoring system that was returning all zeros
- ✅ **ROOT CAUSE**: PRM model was being loaded incorrectly + wrong scoring logic
- ✅ **SOLUTION**: Fixed model loading (AutoModel vs AutoModelForCausalLM) + implemented proper binary classification scoring
- ✅ **DATASET FIX**: Fixed experimental design to use unique problems per injection point (no more repetition)
- ✅ **RESULTS**: Generated working results in `all_results.json` with meaningful PRM scores (-1 to +1 range)
- ✅ **VALIDATION**: PRM now properly discriminates between good/bad reasoning (e.g., 0.89 for good reasoning, -0.55 for bad)
- ✅ **INFERENCE LOOP FIX**: Updated problems_per_experiment from 2 to 167 to process all 500 problems
- ✅ **ANSWER EXTRACTION**: Added final answer extraction from model outputs with regex patterns
- ✅ **ACCURACY TRACKING**: Added comprehensive accuracy statistics at experiment and summary levels
- ✅ **DATASET REDESIGN**: Complete rewrite of mawps_augment.py with focused calculation changes
- ✅ **VARIATION TYPES**: Final dataset has 3 balanced variation types (25% each) plus 24% true controls
- ✅ **PHYSICAL CONSTRAINTS**: Fixed negative answer issues for physical object problems

### Session 4 (2025-08-10 Evening)
- ✅ **PARADIGM SHIFT**: Moved away from PRM scoring to mechanistic measurement
- ✅ **BRUTAL HONESTY**: Realized PRM measures reasoning quality, not adaptation latency
- ✅ **PROPER METRICS**: Implemented KL divergence, attention tracking, token probability ratios
- ✅ **TOKEN-LEVEL ANALYSIS**: Switched from sentence-level to token-level granularity  
- ✅ **CAUSAL EXPERIMENTS**: Added oracle comparison, noise controls, statistical testing
- ✅ **COMBINED IMPLEMENTATION**: Created `run_proper_analysis.py` (92KB) with all measurement methods
- ✅ **DOCUMENTATION UPDATE**: Updated README.md and CLAUDE.md to reflect new approach
- ✅ **METHODOLOGY CLARIFICATION**: Defined detection vs correction vs stabilization latency
- ✅ **RESEARCH FOCUS**: Shifted to measuring information source switching, not output quality

### Session 5 (2025-08-11) - CURRENT
- ✅ **DATASET FIXES**: Updated FairytaleQA loader to handle local files, increased to 500 samples
- ✅ **ATTENTION FIX**: Resolved tensor dimension mismatch by using eager attention mode
- ⚠️ **PERFORMANCE ISSUE**: Eager attention mode is 15-20x slower than SDPA
- ✅ **METRICS FOCUS**: Using KL divergence, token probabilities, embeddings (skipping slow attention)
- ✅ **BASELINE CLARIFICATION**: Defined test conditions (oracle, injection, ignore, flush, partial)
- 🔄 **IN PROGRESS**: Testing with MAWPS math dataset using fast metrics only

## Current Issues & Bugs

### Recently Fixed ✅
1. **~~PRM Scoring All Zeros~~** - FIXED: Was using wrong model loading and scoring approach
2. **~~Dataset Repetition~~** - FIXED: Same problems were being repeated across different injection points

### Known Issues
1. **Attention Tracking Speed**: Eager mode required for attention weights is 15-20x slower than SDPA
2. **Empty Non-Math Loop**: `inference_loop_nonmath.py` is empty (0 bytes)
3. **Memory Usage**: Full precision models use ~28GB VRAM total

### Potential Improvements
1. Implement adaptive injection timing based on content
2. Add confidence intervals to adaptation metrics
3. Implement batch processing for efficiency

## TODO / Future Tasks

### Immediate (Priority 1)
- [ ] Run full MAWPS dataset with fast metrics (no attention tracking)
- [ ] Process FairytaleQA dataset with local file loader
- [ ] Implement baseline conditions (ignore, flush, partial recompute)
- [ ] Generate final results for both datasets

### Short-term (Priority 2)
- [ ] Add visualization scripts for adaptation curves
- [ ] Implement statistical significance testing
- [ ] Create summary statistics script
- [ ] Add error handling for model loading failures

### Long-term (Priority 3)
- [ ] Compare adaptation across different model families (Llama, Mistral, etc.)
- [ ] Experiment with different injection strategies (gradual, multiple points)
- [ ] Create web interface for real-time experimentation
- [ ] Publish results and methodology

## Commands & Quick Reference

### Run Experiments
```bash
# Math experiments
python3 inference_loop_math.py

# Non-math experiments (once implemented)
python3 inference_loop_nonmath.py

# Check results
cat all_results.json | python3 -m json.tool | less
```

### Debug Commands
```bash
# Check GPU availability
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Clear model cache if needed
rm -rf ~/.cache/huggingface/hub/models--Qwen*

# Monitor memory usage
top -o mem
```

### Git Status at Session Start
```
Branch: main
Modified: requirements.txt
Deleted: Several preprocessing files (moved to experiments/)
Untracked: .DS_Store, experiments/, preprocessed/, training/
```

## Important Code Snippets

### Loading Models (for reference)
```python
# Generation model
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR
).to(device)

# PRM model
prm_model = AutoModelForCausalLM.from_pretrained(
    PRM_MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR
).to(device)
```

### Dataset Format
```json
// Math (MAWPS) - Updated structure with variation tracking
{
    "Question": "Original problem",
    "Answer": 6.0,
    "alt_problem": "Modified problem", 
    "alt_answer": 8.0,
    "variation_type": "single_number" | "multi_number" | "added_step" | null,
    "Equation": "5 + 1",
    "alt_equation": "7 + 1" | null
}

// Non-Math (Semantic)
{
    "Question": "Original question",
    "Answer": "Original answer",
    "alt_problem": "Different question",
    "alt_answer": "Different answer"
}
```

## Session Notes

### Key Insights
1. **True Adaptation**: KV-cache injection maintains model state, forcing real adaptation
2. **Dual Metrics**: Cumulative vs individual scores capture different adaptation aspects
3. **CoT Reasoning**: Chain-of-thought prompting improves baseline quality
4. **Local Execution**: No API rate limits, full control over generation

### Architecture Decisions
- Using base models (not instruct) for cleaner adaptation measurement
- Separate models for generation and scoring to avoid bias
- JSON output for easy analysis and visualization
- Sentence-level granularity for precise latency measurement

## Contact & Resources
- Project Root: `/Users/justinyu/Desktop/CS/Algoverse/VAJM_adaptency/`
- Qwen Model Docs: https://github.com/QwenLM/Qwen2.5-Math
- PRM Paper: Process Reward Models for math reasoning

## Next Session Checklist
1. Check if models are still cached
2. Verify Python environment (3.12.6)
3. Review this file for context
4. Check git status for any uncommitted changes
5. Continue with TODO items

---
*This file should be updated at the end of each session to maintain continuity*