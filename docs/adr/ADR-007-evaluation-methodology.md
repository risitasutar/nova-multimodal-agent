# ADR-007: Evaluation methodology

**Status:** Accepted · 2026-10-05

## Decision
- **Golden set:** `evaluation/dataset.json` has 42 cases in 12 categories, built on a **fictional** company
  report. A correct answer therefore shows retrieval worked, not that the model memorised the facts. Gold
  pages are 1-based.
- **One harness for both systems:** the baseline is a faithful reproduction of the pre-upgrade ReAct agent
  (`nova/agent/baseline.py`). Both systems use the same model, seed, market-data provider and web search.
- **Deterministic graders** (`evaluation/metrics.py`):
  - substring and number checks for objectively gradable cases;
  - tool-category sets for routing;
  - page-based Hit@k and MRR;
  - cited pages checked against retrieved and gold pages;
  - a lexical groundedness proxy.

  No LLM-as-judge feeds the headline numbers: a judge using the same model would be biased, and slow on CPU.
- **Component retrieval benchmark:** measures retriever quality separately from agent behaviour.
- **No tuning on the test set:** the relevance threshold is calibrated on a separate dev set.
- **Reproducibility:** every run is saved per case (`results/runs/*.jsonl`), and the report is regenerated
  from those records.

## Limitations
- 42 cases is a small sample, so confidence intervals are wide.
- Live web and finance answers change over time. Those cases are scored only on tool use, grounding and failures.
- The groundedness proxy is lexical and can miss support that is paraphrased.
