# CODA Modernization Plan

## Context

`coda.md` is a research paper describing CODA, a framework for diagnosing and repairing LLM prompt degradation across model migrations. It ships with a reference implementation (`/home/cesar/research/coda`): a 4-phase pipeline (Diagnose → Classify → Optimize → Validate) exercised by 5 demonstration scenarios (Cases A–E), each migrating a prompt from an "old" model to a "new" model (e.g. GPT-4 → GPT-4o, Claude Sonnet 4 → Claude Haiku 4.5).

The user's complaint: the models and prompting style are already dated relative to the paper's own July 2026 context (GPT-4 as a baseline, prose-only prompts with no structured-output/tool-forcing, etc.), and the reference implementation has accumulated real bugs on top of that staleness. Two deliverables are wanted: (1) a genuinely modernized system, and (2) a documented methodology — new material in `coda.md` itself — explaining the audit process and how the old prompts/models were migrated to the new ones, since this is a research paper and needs that rigor rather than a silent diff.

Three scope decisions were confirmed with the user:
- **Add Google Gemini as a third provider** (fulfills the paper's own "planned extension" note), which requires new scenario(s) to justify Gemini's inclusion rather than adding unused plumbing.
- **Actually execute the modernized pipeline live** against real provider APIs to regenerate the paper's PPI/MHS numbers — gated on the user supplying API keys (none exist in the repo today — no `.env` file present) and approving real spend.
- **Recalibrate scenario stressors** so the much-more-capable modern models still produce an illustrative spread of outcomes (some recoverable via optimization, one clean/no-op transfer, one genuinely optimization-resistant case) instead of trivially passing everything.

### Confirmed bugs to fix (found and verified by direct file reads, not just inference)

1. All 4 optimizers (`optimizers/{ape,opro,protegi,evoprompt}.py`) hardcode `provider="openai", model="gpt-4o"` for their own internal meta-prompt calls (verified in `protegi.py:63-64,92-93`) — they accept a `model_config` param but never read it, so "optimizer reasoning" always runs on gpt-4o regardless of the case's actual models.
2. `evaluators/classifier.py`'s LLM-as-classifier call is likewise hardcoded (invoked from `run_classification.py`, not shown here but confirmed by prior exploration).
3. `scripts/run_optimization.py`'s `quick_evaluate()` (line 50-57) and all of `scripts/run_validation.py`'s `client.complete(...)` calls (lines 64-71, 94-101) never pass `tools=` — verified by direct read. This silently drops tool schemas for Case B (the tool-calling scenario) during both optimization scoring and final validation, even though `run_diagnosis.py` passes tools correctly.
4. `run_optimization.py` only ever writes `case_{id}_optimized.txt` (line 163) even for Case B, whose original is `case_b_original.json` with a `tools` array — the tool schema is discarded once optimization runs (line 95-97 extracts only `system_prompt`).
5. `evaluators/llm_judge.py` (LLM-as-judge) exists but is never called from any `run_*.py` script — dead code.
6. No use of provider-native structured outputs / JSON-schema modes or `tool_choice` forcing anywhere — every output contract (Case C's JSON, Case D/E's report format, Case B's "always use a tool") is enforced purely through prose instructions, then parsed post-hoc via regex in `evaluators/metrics.py`.
7. **`config/models.yaml`'s Case D header comment (lines 18-21) is stale and wrong**: it describes Case D as "arithmetic cost estimation... unfixable," but `prompts/case_d_original.txt` is actually a support-ticket-response writer with no arithmetic, and `coda.md`'s own results table (Table 4) reports **D as recovered** (PPI 92.3 → 98.9). The vestigial `cost_estimation_*` functions in `metrics.py` (unused by any current case) are the fossil of that earlier Case D design. **Case B, not D, is the actual optimization-resistant/unfixable case** per the paper's own tables — this matters because scenario recalibration (below) must protect B's "unfixable" slot, not D's.
8. `requirements.txt` lists `jsonschema`/`pandas`/`numpy` but nothing in the read code actually uses them; `README.md` documents only 3 cases when there are 5; `.env.example` has no Gemini key; stray `output.txt` run-log at repo root.
9. `optimizers/router.py`'s `CATEGORY_OVERRIDES` has no path for `safety_refusal` — it would fall through to normal LLM-based optimization, contradicting `coda.md`'s own text ("Safety changes are flagged for human review") and the Limitations section's admission this isn't implemented yet.

## Guiding principle: two-tier modernization

Applying full "modern" API contract-enforcement (structured outputs, `tool_choice` forcing) to a scenario's **original** prompt can pre-empt the very failure the scenario exists to demonstrate — if Case C's original gets a native JSON schema, there's nothing left to diagnose when it migrates. So:

- **Original prompts** (`prompts/case_*_original.*`) get *clarity-only* modernization: XML/markdown sectioning of the existing prose rules, tightened wording, few-shot exemplars that clarify intent without deciding actual test-case answers. They deliberately keep "legacy" API usage (no `response_format`, no `tool_choice`) so the migration still has something real to break.
- **Optimized prompts** (`prompts/case_*_optimized.*`) and the optimizer/validation code path get full modern capability — these are framed as levers a human or the optimizer would reach for during a real repair, not baked into the starting point.
- Prompt caching applies everywhere (cost/latency only, no narrative effect) — but note honestly in the paper that these system prompts are likely below Anthropic's minimum cacheable-prefix length until few-shot exemplars are added.
- Case F's document-tagging is the one exception: XML document labels belong in the *original*, because context_utilization is a category about context organization itself, not an output contract — labeling documents doesn't tell the model which one is authoritative, so it doesn't erase the failure mode.

This principle is also the spine of the new methodology section in `coda.md` (Phase 6 below).

## Phase 1 — `scripts/llm_client.py` (foundation; everything else depends on it)

- Add a Gemini backend: new `_complete_gemini()` + lazy client property using the `google-genai` SDK. Normalize `system_prompt` → `system_instruction`, `tools` → `types.Tool(function_declarations=[...])` (reuse the same `input_schema`/`parameters` normalization already used for OpenAI/Anthropic), and parse response parts into the existing `{text, tool_calls, usage, latency_ms, raw}` shape. **Verify exact SDK import path, env var name (`GEMINI_API_KEY` vs `GOOGLE_API_KEY`), and field names against current `google-genai` docs before coding** — these move fast and my knowledge here is not fully current.
- Add a normalized `response_schema` param to `complete()`, translated per provider: OpenAI `response_format={"type":"json_schema",...,"strict":True}`; Anthropic via a forced single-tool trick (`tool_choice={"type":"tool","name":"emit_result"}` with the schema as its `input_schema`, then re-serialize the tool's `input` back into the `text` field so `metrics.py`'s existing `json.loads`-based parsers keep working unmodified); Gemini via `response_mime_type="application/json"` + `response_schema=`.
- Add a normalized `tool_choice` param (`None`/`"auto"`/`"required"`/`"<name>"`), translated to each provider's actual mechanism.
- Add `reasoning_tier: bool` (read from a new per-model `models.yaml` field) so `_complete_openai` uses `max_completion_tokens` instead of `max_tokens` and drops/clamps temperature for reasoning-class OpenAI models — needed before a current-generation OpenAI model can be safely plugged in.
- Add Anthropic prompt caching (`cache_control: {"type":"ephemeral"}` on the system block) behind an `enable_prompt_cache` flag.

## Phase 2 — Fix the bugs (before touching model identifiers or scenario content)

Do this before recalibration, or recalibration results get contaminated by bugs masquerading as scenario properties (e.g. Case B's tool-schema drop would look like "the model genuinely can't do this" instead of "the harness never gave it the schema").

- **Optimizer meta-model fix**: add `optimizer_meta_model: {provider, model}` to `config/thresholds.yaml` (default: the strongest available model, e.g. Claude Opus 4.8). In `run_optimization.py`, inject it into each optimizer's config dict the same way `failure_cases`/`classification_summary` are already injected (line 141-143). In each of `ape.py`/`opro.py`/`protegi.py`/`evoprompt.py`, replace the hardcoded `provider="openai", model="gpt-4o"` calls with `config["meta_model_config"]["provider"/"model"]`. `evoprompt.py` also reuses `ape.generate_candidate` for population seeding — thread the param through that call site too.
- **Classifier model fix**: same pattern in `run_classification.py` — read `thresholds.get("classifier_model", thresholds["optimizer_meta_model"])` instead of the current literal.
- **`tools=` threading fix**: factor a single `load_prompt_artifact(path) -> dict` helper (reuse/extract from `run_diagnosis.py`'s existing `.txt`/`.json` dispatch rather than writing a third copy) and use it in `run_optimization.py` and `run_validation.py`. Thread `tools=prompt_dict.get("tools")` through `quick_evaluate` and both `client.complete()` calls in `run_validation.py`.
- **Preserve Case B's tool schema through optimization**: in `run_optimization.py`, when the original `prompt_file` is `.json`, write the optimized artifact as `case_b_optimized.json` (`{"system_prompt": best_prompt, "tools": tools}`) instead of always `.txt`; update `run_validation.py`'s loader accordingly.
- **Wire `llm_judge.py`** into `run_diagnosis.py`'s Case A evaluator as a `tone_style` metric (add a small weight bucket in `models.yaml`), and into new Case G (below) as a documented proxy for human safety review — call both through the same `optimizer_meta_model` config, not a fresh hardcode.
- **DRY the JSON-fence-stripping** logic duplicated in `classifier.py::parse_classification` and `protegi.py::generate_gradient` (and about to appear a third time in the judge-wiring code) into one `strip_json_fences()` helper in `evaluators/__init__.py`.
- **Router safety-review path**: add a `human_review` short-circuit in `optimizers/router.py` — when `primary_category == "safety_refusal"`, return the prompt unchanged with `"optimizer": "human_review"` regardless of zone, mirroring the existing green-zone short-circuit (lines 111-121). This is what lets new Case G actually demonstrate the paper's existing (currently unimplemented) safety-routing claim.
- **Cleanup**: delete the vestigial `cost_estimation_*` functions from `metrics.py`; fix the stale Case D comment block in `models.yaml`; delete stray `output.txt`.

## Phase 3 — Config and docs

- `config/models.yaml`: bump all 5 pairs to current-generation model identifiers (see below), add `reasoning_tier`/`enable_prompt_cache` flags, add Cases F and G using the identical schema as A–E.
- `config/thresholds.yaml`: add `optimizer_meta_model` (+ optional `classifier_model` override).
- `requirements.txt`: add `google-genai`; either wire `jsonschema` into a real `evaluators/schema_validation.py` redundant-check for Case C/F's structured contracts, or drop it if unused; give `pandas`/`numpy` an actual job via the results-aggregation script (Phase 5) or drop them too.
- `.env.example`: add `GEMINI_API_KEY`.
- `README.md`: rewrite for 7 cases + Gemini setup + updated cost table; delete `output.txt` reference.

**Model identifiers** (best current knowledge — verify against live provider docs immediately before hardcoding, since these move fast and web search results for OpenAI/Google in particular were not fully reliable):
- Anthropic (high confidence, from environment config): Opus 4.8 = `claude-opus-4-8`, Sonnet 5 = `claude-sonnet-5`, Haiku 4.5 = `claude-haiku-4-5-20251001` (already in use).
- OpenAI (lower confidence): flagship is GPT-5.5 (established, ~April 2026); GPT-5.6 is in limited preview — treat GPT-5.5 as the stable target. Exact lighter-tier naming needs verification against OpenAI's models API before use.
- Google (lower confidence): `gemini-3.1-pro-preview` (reasoning-first, 1M context) and `gemini-3.5-flash` (GA) look like a plausible downgrade pair — verify against `ai.google.dev` docs before use.

## Phase 4 — Scenario recalibration + 2 new scenarios

Reuse `metrics.py`'s existing check-function style for all new metrics rather than inventing new evaluation abstractions.

- **Case A** (keep recoverable): add a genuinely competing instruction (an upsell rule that collides with the existing format/scope constraints) plus test cases combining escalation + competitor-mention + upsell in one message.
- **Case B** (protect the unfixable slot — this is the important one): add a third tool (`create_support_ticket` with a nested object parameter) that semantically overlaps `query_database` in some phrasings, plus test cases requiring 2-of-3 disambiguation and nested/conditionally-required field population. This is genuinely hard multi-tool judgment even for current-generation cheap-tier models, which is what preserves B as the "optimization can't fully close this" case. **Update `coda.md`'s Worked Example** (currently walks through the old 2-param `query_database` failure) to match.
- **Case C** (keep clean-transfer slot): add second-order/multi-hop calculations so the CoT task stays nontrivial for flagship-tier models on both sides. No `response_schema` upgrade here (per the two-tier principle) — stays fully prose-enforced.
- **Case D** (keep recoverable): harden priority classification with genuinely conflicting-signal tickets and a social-pressure variant (customer demands "P1 now" against contrary rubric data).
- **Case E** (keep recoverable): multi-state incident narratives (severity changes mid-incident) requiring the model to report the final correct state despite distractor intermediate states, plus duration arithmetic across windows.
- **New Case F** — Gemini within-provider downgrade, `context_utilization` primary (currently zero cases exercise this taxonomy category): a research-brief assistant given several source documents, some superseded by a later one, some irrelevant distractors; must cite the authoritative doc and avoid stale figures. `gemini-3.1-pro-preview → gemini-3.5-flash`. New metrics: `context_utilization_cites_correct_source`, `context_utilization_prefers_current_over_stale`, `context_utilization_no_hallucinated_claim`. Requires a new `"context_utilization"` bucket in `run_diagnosis.py`'s `metric_mapping`.
- **New Case G** (stretch, if budget allows) — cross-provider, `safety_refusal` primary (also zero cases today): a policy-boundary triage assistant over benign-vs-boundary dual-use requests, `claude-haiku-4-5-20251001 → gemini-3.5-flash`, designed outcome is "flagged for human review" via the Phase 2 router change — gives the paper a real example of the safety-routing claim it already makes.

Together F (and G) bring taxonomy coverage from ~5/7 to 7/7 implicit categories — this is the structural argument for why Gemini's addition isn't just unused plumbing, and becomes the backbone of the re-validation protocol described in the paper (Phase 6).

## Phase 5 — Live execution (explicitly gated, do last)

1. Prerequisite: user supplies `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` in `.env` (none exist today).
2. Verify Phases 1-4 pass the dry-run/mocked checks below before any live spend.
3. **Tiny smoke test first** (~$0.01-0.05): one real call per provider with the actual new model identifiers, to catch a bad model string before running a full suite into it.
4. **Staged rollout**: run Case A live first (cheapest), inspect actual cost, extrapolate to the remaining 6 cases.
5. **Explicit go/no-go checkpoint with the user** before running the rest — present Case A's real cost + full projection. Rough order-of-magnitude placeholder (not reliable, pricing for these specific models is not something I can verify from memory): $80-180 for all 7 cases, dominated by ProTeGi/OPRO's per-iteration full-suite re-scoring.
6. Build `scripts/aggregate_results.py` (uses `pandas`) to roll up every case's JSON artifacts into the paper's table rows (CSV + LaTeX row form) so `coda.md`'s tables are generated from real run output, not hand-transcribed.

## Phase 6 — `coda.md` methodology write-up (the paper's second deliverable)

Insert a new subsection **"Modernization methodology"** under "Reference implementation and experiment design," right after the existing "Scenario design" subsection and before "Measurement protocol." Cover:
- **Audit process**: the checklist of gaps found (hardcoded meta-model providers, missing `tools=` threading, dead `llm_judge.py`, absent structured-output/`tool_choice`/caching support, stale comments/dead code) — framed as a reusable audit for future migrations of this reference implementation itself.
- **Modernization technique + rationale**: state the two-tier principle explicitly, justify structured outputs / `tool_choice` / XML sectioning / few-shot / caching individually, including the honest caveat about caching's minimum-prefix limitation on these short prompts.
- **Scenario recalibration**: summarize the per-case stressor changes and the 5/7 → 7/7 taxonomy-coverage argument for Cases F/G.
- **Re-validation protocol**: the mocked-then-staged-live verification process (Phase 7 below) used to confirm the recalibrated pipeline still exercises all 7 categories before trusting final numbers.

Other required edits for internal consistency once code changes land:
- Table 3 (scenarios) and Table 4 (results): update model pairs, add F/G rows, regenerate Table 4 from `aggregate_results.py` once live numbers exist.
- "The optimization phase" section + Strategy Selection Matrix: update the "tool-calling routes through PPI-based path, deterministic translation planned" and safety-refusal rows now that the router's `human_review` path is implemented — say plainly what's implemented (routing) vs. still future work (full generate-variants-and-human-evaluate workflow), don't overclaim.
- "Implementation architecture": Gemini moves from "planned extension" to implemented.
- Worked Example: rewrite against Case B's new 3-tool/nested-schema stressor.
- Limitations/Future work: remove the Gemini bullet (shipped); keep the rest.
- Bibliography: add citations for the new models actually used; keep existing GPT-4/Claude Sonnet 4 citations since older-model behavior is still discussed in surrounding prose.

## Phase 7 — Verification before live spend

No test directory exists today; add a minimal one:
1. **Mocked unit checks** (`unittest.mock.patch` on `LLMClient.complete`, no API calls): Gemini response normalization round-trips correctly; `meta_model_config` correctly threads through the router into all 4 optimizers; `tool_choice`/`response_schema` translate to the right provider kwargs; Case B's `.json` artifact preserves `tools` end-to-end through optimize → validate.
2. **New-metric unit checks**: hand-constructed input/output pairs with known right answers for the new `context_utilization_*`, `safety_refusal_*`, and `tool_calling_nested_param_success` functions.
3. **Single-test-case dry run per scenario per phase**: trimmed 1-row test suites to confirm structured-output parsing succeeds (Anthropic forced-tool JSON trick, Gemini `response_mime_type`, OpenAI `response_format`) before spending on full suites.
4. Only after 1-3 pass does Phase 5's staged live rollout begin.

## Critical files

- `scripts/llm_client.py` — provider abstraction, needs Gemini + structured-output + tool_choice + caching
- `config/models.yaml`, `config/thresholds.yaml` — model pairs, meta-model config, new cases
- `optimizers/{ape,opro,protegi,evoprompt,router}.py` — meta-model threading, safety-review routing
- `scripts/{run_optimization,run_validation,run_classification}.py` — bug fixes (tools=, prompt loading, model hardcoding)
- `evaluators/{metrics,classifier,llm_judge,__init__}.py` — new metrics, dead-code wiring/removal, shared JSON-fence helper
- `prompts/case_{a-g}_{original,optimized}.*`, `test_suites/case_{a-g}_tests.jsonl` — the actual content rewrite
- `coda.md` — new methodology section + consistency edits across Tables 3/4, worked example, architecture sections, bibliography

## Verification (end-to-end)

1. Run Phase 7's mocked unit checks — all must pass with zero API calls before anything live happens.
2. Run the per-scenario single-test-case dry runs across all 4 pipeline phases for all 7 cases.
3. Run the tiny real smoke test (~$0.01-0.05) confirming all new model identifiers actually resolve on their respective provider APIs.
4. Run Case A live end-to-end, inspect real cost and output quality by hand, get explicit user go/no-go before the remaining 6 cases.
5. Run `scripts/aggregate_results.py` and manually spot-check its output against 1-2 cases' raw JSON before trusting it to regenerate `coda.md`'s tables.
6. Read through the updated `coda.md` end-to-end for internal consistency (model names, PPI numbers, table/prose agreement) before considering the paper deliverable done.
