# CODA Experiment Scaffold

Reproducible experiments for validating the CODA (Cross-model Optimization through
Diagnostic Analysis) framework from the research paper.

## Overview

This scaffold implements three case studies from the paper:

- **Case Study A**: System prompt migration (GPT-4 -> GPT-4o)
- **Case Study B**: Tool-calling prompt adaptation (Claude 3 Sonnet -> Claude 3.5 Sonnet)
- **Case Study C**: Cross-provider CoT migration (GPT-4 -> Claude 3.5 Sonnet)

Each case study runs the full CODA pipeline: Diagnose -> Classify -> Optimize -> Validate.

## Project Structure

```
coda-experiments/
  config/
    models.yaml          # Model configurations and API endpoints
    thresholds.yaml      # Metric thresholds and PPI zone definitions
  prompts/
    case_a_original.txt  # Case A: original customer service system prompt
    case_a_optimized.txt # Case A: optimized prompt (generated during experiment)
    case_b_original.json # Case B: original tool-calling prompt + tool schemas
    case_b_optimized.json
    case_c_original.txt  # Case C: original financial analysis CoT prompt
    case_c_optimized.txt
  test_suites/
    case_a_tests.jsonl   # 50 customer service test cases
    case_b_tests.jsonl   # 50 tool-calling test cases
    case_c_tests.jsonl   # 50 financial analysis test cases
  evaluators/
    metrics.py           # Core metric computation (accuracy, format, adherence, etc.)
    classifier.py        # LLM-as-classifier for failure taxonomy mapping
    llm_judge.py         # LLM-as-judge for subjective quality evaluation
  scripts/
    run_diagnosis.py     # Phase 1: Run prompts on old + new model, compute metrics
    run_classification.py # Phase 2: Classify failures using taxonomy
    run_optimization.py  # Phase 3: Apply optimization strategies
    run_validation.py    # Phase 4: Validate optimized prompts
    run_full_pipeline.py # Run all four phases end-to-end
  results/               # Output directory for experiment results
  requirements.txt
  .env.example
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run a single case study

```bash
# Run Case A end-to-end
python scripts/run_full_pipeline.py --case a

# Or run individual phases
python scripts/run_diagnosis.py --case a
python scripts/run_classification.py --case a
python scripts/run_optimization.py --case a
python scripts/run_validation.py --case a
```

### 4. Run all case studies

```bash
python scripts/run_full_pipeline.py --case all
```

## Estimated Costs

Rough token estimates per full pipeline run (all 3 case studies):

| Component           | Approx. Tokens | Approx. Cost |
|---------------------|----------------|--------------|
| Diagnosis (old model) | ~200K input, ~100K output | $3-8 |
| Diagnosis (new model) | ~200K input, ~100K output | $3-8 |
| Classification        | ~150K input, ~50K output  | $2-5 |
| Optimization (ProTeGi iterations) | ~300K input, ~100K output | $5-12 |
| Validation            | ~200K input, ~100K output | $3-8 |
| **Total**            |                | **$16-41** |

Costs vary by model pricing. Run `--dry-run` to estimate without making API calls.

## Extending

To add a new case study:
1. Create a prompt file in `prompts/`
2. Create a test suite in `test_suites/` (JSONL format)
3. Add a case config entry in `config/models.yaml`
4. Run `python scripts/run_full_pipeline.py --case <your_case>`

## Output

Results are saved to `results/<case>/<timestamp>/`:
- `diagnosis_report.json` -- per-metric baseline vs. new model comparison
- `classification_report.json` -- failure taxonomy mapping with severity scores
- `optimization_log.json` -- optimization trajectory and prompt versions
- `validation_report.json` -- final metrics and PPI score
- `summary.json` -- high-level pass/fail and PPI
