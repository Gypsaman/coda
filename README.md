# CODA Experiment Scaffold

Reproducible experiments for validating the CODA (Cross-model Optimization through
Diagnostic Analysis) framework from the research paper.

## Overview

This scaffold implements seven case studies from the paper (see `coda.md`,
"Scenario design" and "Modernization methodology"):

- **Case A**: System prompt migration, within-provider upgrade (GPT-5.5-mini -> GPT-5.5)
- **Case B**: Tool-calling capability shift, within-provider downgrade (Claude Sonnet 5 -> Claude Haiku 4.5) --
  the framework's protected "optimization-resistant" scenario
- **Case C**: Cross-provider CoT migration (GPT-5.5 -> Claude Haiku 4.5) -- the "clean transfer" scenario
- **Case D**: Cross-provider format downgrade (Claude Sonnet 5 -> GPT-5.5-mini)
- **Case E**: Within-provider format downgrade (GPT-5.5 -> GPT-5.5-mini)
- **Case F**: Context-utilization migration (Gemini 3.1 Pro -> Gemini 3.5 Flash)
- **Case G**: Cross-provider safety-routing demonstration (Claude Haiku 4.5 -> Gemini 3.5 Flash)

Each case study runs the full CODA pipeline: Diagnose -> Classify -> Optimize -> Validate.

Model identifiers in `config/models.yaml` reflect current-generation models as of this
writing; verify them against live provider docs before a production run, since exact
model strings and pricing tiers move fast.

## Project Structure

```
coda/
  config/
    models.yaml          # Model configurations per case (old/new model, prompt, test suite, weights)
    thresholds.yaml       # Triage zones, taxonomy, optimizer hyperparameters, meta-model config
  prompts/
    case_{a-g}_original.*  # Original system prompts (+ tool schemas for Case B)
    case_{a-g}_optimized.* # Optimized prompts (generated during experiment runs)
  test_suites/
    case_{a-g}_tests.jsonl # Test cases with ground-truth expectations
  evaluators/
    metrics.py            # Core metric computation (accuracy, format, adherence, tool-calling, etc.)
    classifier.py          # LLM-as-classifier for failure taxonomy mapping
    llm_judge.py            # LLM-as-judge for subjective quality/tone/safety-proxy evaluation
    schema_validation.py    # jsonschema-based redundant structured-output validation
  optimizers/
    ape.py, opro.py, protegi.py, evoprompt.py  # Optimizer backends
    router.py              # Triage-zone + failure-category routing (incl. safety human-review short-circuit)
  scripts/
    llm_client.py          # Unified OpenAI/Anthropic/Gemini client
    run_diagnosis.py       # Phase 1: run prompts on old + new model, compute metrics
    run_classification.py  # Phase 2: classify failures using taxonomy
    run_optimization.py    # Phase 3: apply optimization strategies
    run_validation.py      # Phase 4: validate optimized prompts
    run_full_pipeline.py   # Run all four phases end-to-end
    aggregate_results.py   # Roll up all cases' results into paper-table CSV/LaTeX rows
  results/                 # Output directory for experiment results
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
#   GEMINI_API_KEY=...
```

### 3. Run a single case study

```bash
# Run Case A end-to-end
python scripts/run_full_pipeline.py --case a

# Or run individual phases
python scripts/run_diagnosis.py --case a
python scripts/run_classification.py --case a --results-dir results/case_a/<timestamp>
python scripts/run_optimization.py --case a --results-dir results/case_a/<timestamp>
python scripts/run_validation.py --case a --results-dir results/case_a/<timestamp>
```

### 4. Run all case studies

```bash
python scripts/run_full_pipeline.py --case all
```

## Estimated Costs

Rough, unreliable placeholder estimates per full pipeline run (all 7 case studies) --
replace with a real pilot-run cost (Case A first) before trusting these for budgeting.
Current-generation flagship pricing and longer (few-shot, XML-sectioned) prompts push
this well above the original 3-case estimate:

| Component                          | Approx. Cost |
|-------------------------------------|--------------|
| Diagnosis (old + new model, 7 cases)| $20-40       |
| Classification                      | $10-20       |
| Optimization (ProTeGi/OPRO/EvoPrompt/APE iterations) | $40-90 |
| Validation                          | $15-30       |
| **Total**                           | **~$80-180** |

Run `--dry-run` to estimate tokens without making API calls. Always pilot Case A alone
first, inspect its real cost, and get explicit sign-off before running the rest.

## Extending

To add a new case study:
1. Create a prompt file in `prompts/`
2. Create a test suite in `test_suites/` (JSONL format)
3. Add a case config entry in `config/models.yaml`
4. Add an `evaluate_case_<id>` function and register it in `EVALUATORS` in `scripts/run_diagnosis.py`
5. Add any new case-specific metric functions to `evaluators/metrics.py`
6. Run `python scripts/run_full_pipeline.py --case <your_case>`

`scripts/llm_client.py`'s `complete()` also exposes `tool_choice` (force tool use),
`response_schema` (native structured output, emulated via forced tool-call on Anthropic),
`reasoning_tier` (use `max_completion_tokens` for reasoning-class OpenAI models), and
`enable_prompt_cache` (Anthropic prompt caching) -- reach for these when a new case's
optimized prompt calls for a genuine API-level contract instead of a prose instruction.

## Output

Results are saved to `results/<case>/<timestamp>/`:
- `diagnosis_report.json` -- per-metric baseline vs. new model comparison
- `classification_report.json` -- failure taxonomy mapping with severity scores
- `optimization_log.json` -- optimization trajectory and prompt versions
- `validation_report.json` -- final metrics and PPI score
- `summary.json` -- high-level pass/fail and PPI

Run `python scripts/aggregate_results.py` after running all cases to produce a
consolidated CSV and LaTeX-table-row summary across every case's latest results.
