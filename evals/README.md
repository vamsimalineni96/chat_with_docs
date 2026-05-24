# evals/

Tooling for the three observability pillars from [docs/OBSERVABILITY.md](../docs/OBSERVABILITY.md). One folder per pillar — each folder contains the aggregators, dataset, and report-rendering code for that pillar.

| Pillar | Folder | What lives there |
|---|---|---|
| **Quality** | [`quality/`](quality/) | Eval harness — qa_set, retrieval metrics, LLM-as-judge, reporter, orchestrator. See [`quality/README.md`](quality/README.md). |
| **Cost** | [`cost/`](cost/) | Cost-per-task aggregator + pricing table. See [`cost/README.md`](cost/README.md). |
| **Latency** | [`latency/`](latency/) | p50/p95/p99 aggregator + slow-trace surfacing. See [`latency/README.md`](latency/README.md). |

Application-side instrumentation (token counting in [`src/utils/services/tokenizers.py`](../src/utils/services/tokenizers.py), heuristic checks in [`src/utils/services/heuristics.py`](../src/utils/services/heuristics.py), retry helpers in [`src/utils/services/retry.py`](../src/utils/services/retry.py)) stays in `src/` because it runs in the request path. This folder is for the *off-path* tooling that reads the resulting traces and produces reports.
