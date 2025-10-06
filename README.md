# IOH XSS Experiment - LLM Security Research

Experimental pipeline for studying **Insecure Output Handling (IOH)** vulnerabilities in Large Language Models, focusing on Cross-Site Scripting (XSS) risks.

## 🎯 Research Overview

Empirical study evaluating IOH vulnerabilities in two LLM architectures:
- **DeepSeek-R1 1.5B** (reasoning-optimized)
- **Llama 3 8B** (general-purpose)

Testing **500 prompts** across 3 risk tiers with multiple defense layers.

##  Key Findings

| Metric | DeepSeek-R1 | Llama 3 8B |
|--------|-------------|-------------|
| Baseline Exploit Rate | 4.2% | 27.6% |
| Pattern Prevalence | 87.8% | 74.2% |
| Defense Effectiveness | 100% | 100% |
## Experiment replicability
- **Seed value equals 42**
- **Used Ollama version 0.12.3**
## 🚀 Quick Start

```bash
git clone https://github.com/AAnchimiuk/IOH-xss-experiment.git
cd IOH-xss-experiment
pip install -r requirements.txt

# Run experiments
python xss_experiment_pipeline.py --n 500

