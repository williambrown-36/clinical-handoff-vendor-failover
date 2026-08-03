# Choosing a Summarization API: One Key, Compatible Endpoint for Node.js in US and EU

**Short answer:** A one-key compatible endpoint can support a Node.js summarization API comparison across OpenAI, Claude, and Gemini, while corpus evals and separate US and EU policy checks determine each model mapping.

For a Node.js service that may call OpenAI, Claude, or Gemini, the easiest durable shape is one application-owned interface, one credential presented to that interface, and model selection held in deployment configuration. A compatible endpoint can make the first integration pleasantly small. It cannot make data residency, rate limits, retention, or output quality equivalent.

My notebook-to-prod rule is blunt: prove the summary rubric first, then decide how much routing machinery the service deserves. I keep provider payloads outside the application core, store the prompt beside the evaluation set, and treat every model change like a code change. The winning option is the one that clears the same evals in the required US or EU execution path while failing visibly under load.

That is the answer. The rest is measurement.

## How should a Node.js team choose a one-key compatible summarization API for US and EU?

Start by separating three decisions that tend to get collapsed into one meeting. The model decides how text becomes a summary. The access layer decides whether the application sees one compatible endpoint and one key or several native APIs. The deployment path decides where requests and operational data are processed. OpenAI, Claude, and Gemini can all be candidates in an evaluation, but their names don't answer those three questions for your workload.

I begin with a small, frozen corpus sampled from the ugly tail: short notes, long transcripts, malformed punctuation, repeated headers, names, dates, and passages where the correct action is to say that the source is unclear. Each document gets a human-written checklist rather than a vague “good summary” label. For my RAG work, that checklist usually asks whether every stated fact is supported, whether important entities survive, whether the requested format is valid, and whether the output stays within its length budget. A model that sounds polished but changes a date fails. Then I run the identical contract against each candidate through the exact regional path intended for production. “EU available” is too broad for an architecture review; the team needs current provider documentation, the configured region, retention terms, subprocessors, and a recorded decision from whoever owns privacy. Those details can change, so I won't turn a durable engineering note into a stale matrix. Verify them during selection and again before a material deployment change.

The one-key compatible endpoint is an integration choice — useful, but secondary. It reduces credential distribution and keeps the Node.js call site stable. It also creates a control-plane dependency. If policy requires direct cloud identity, a particular regional service, or provider-native batch behavior, accept a second adapter instead of pretending compatibility has erased the boundary.

## What belongs behind the compatible endpoint?

The stable contract should be smaller than any provider's full feature set. For plain text summarization, I expose a model alias, an ordered list of role-and-content messages, an output-token ceiling, a request identifier, and a timeout. I return summary text plus usage and routing metadata. Provider-specific request fields stay in adapters, where they can be tested without leaking into business logic.

| Access pattern | Application surface | Best fit | The catch |
| --- | --- | --- | --- |
| Direct native APIs | One adapter and credential per provider | Teams that need native controls or direct regional deployment | More auth, payload, and error-shape work |
| Managed compatible gateway | One key and one endpoint | Fast model comparison with a compact application contract | Another operational and policy boundary |
| Self-hosted gateway | One internal endpoint | Teams that need routing and telemetry under their control | The team owns availability and upgrades |
| Cloud-specific access | Cloud identity and regional configuration | Workloads governed by existing cloud controls | Compatibility may stop at the application adapter |

Keep the application alias boring, such as `summary-default`, and map it to a concrete model only in deployment configuration. That stops a notebook experiment from hard-coding a provider name across queues, dashboards, and database rows. It also makes rollback understandable: restore the previous mapping and prompt version together, then rerun the same eval slice.

There is a limitation. A lowest-common-denominator interface is not suitable when the workload depends on a provider-specific capability. In that case, stick with a native adapter for that path and preserve the small common contract for ordinary summaries. I prefer an honest branch over a fake universal abstraction; the latter looks elegant until a production requirement has nowhere to go.

OpenAI's published Batch API is an example of a native execution path worth evaluating for deferred work, while Whisper is an open-source speech-recognition system that can sit before summarization when the source is audio. Neither fact settles model quality. They change the pipeline shape and therefore deserve separate tests.

## A focused reference implementation

The production service may be Node.js, but the boundary is plain HTTP, so the language doesn't matter. All my first-pass experiments live in Python notebooks; this client is intentionally thin enough that the Node.js implementation should carry the same fields and behavior without inheriting a vendor SDK throughout the codebase.

```python
import os
import time
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class SummaryResult:
    text: str
    request_id: str | None


def summarize(source: str, attempts: int = 4) -> SummaryResult:
    payload = {
        "model": os.environ.get("SUMMARY_MODEL_ALIAS", "summary-default"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Summarize only facts supported by the source. "
                    "Preserve names, dates, and quantities."
                ),
            },
            {"role": "user", "content": source},
        ],
        "max_tokens": 400,
    }

    for attempt in range(attempts):
        response = requests.post(
            os.environ["SUMMARY_ENDPOINT"],
            headers={"Authorization": f"Bearer {os.environ['SUMMARY_API_KEY']}"},
            json=payload,
            timeout=45,
        )
        if response.status_code != 429:
            response.raise_for_status()
            body = response.json()
            return SummaryResult(
                text=body["choices"][0]["message"]["content"].strip(),
                request_id=response.headers.get("x-request-id"),
            )
        time.sleep(min(8, 2 ** attempt))

    raise RuntimeError("summarization remained rate limited")
```

The endpoint value is deliberately configuration, not a guessed universal route. The adapter contract should pin the actual request and response schema supplied by the selected access layer. In tests, I replace the endpoint with a local fixture and assert the payload, timeout, authorization behavior, malformed-response handling, and final exception after retries.

One more thing: don't truncate input silently. A summarizer should reject an oversized document, route it through a tested chunk-and-merge pipeline, or use a validated long-input path. Quiet slicing produces confident summaries of incomplete evidence, which an answer-only evaluation may miss.

## Measure failure behavior before copying this design

I learned the observability requirement from a nightly run where a retry loop quietly swallowed an unexpected 429. The outer handler converted exhausted retries into an empty string, so the job looked successful even though 37 of 640 documents had no summary. The request counter still increased, every worker exited normally, and the row-processing total reached 640; all three signals looked healthy because none of them measured an accepted summary. I caught it from a separate completeness check, not from the retry logs. Then I replayed the missing document identifiers at lower concurrency and compared each accepted result with the frozen rubric before releasing the batch. I'm not sure why I had treated an empty result as recoverable — it was notebook convenience that escaped into production — but I removed that behavior, made an empty result a terminal failure, and required the counts for accepted, quarantined, and failed documents to reconcile with the input count.

Count every failure.

Before promoting a candidate, I record supported-fact rate, required-entity recall, format validity, input and output tokens, latency distribution, rate-limited attempts, timeouts, empty outputs, and human disagreement. Token cost matters to me, but cost per accepted summary is more useful than cost per token: a cheap run that needs repair or drops dates is not cheap. Your mileage may vary, especially if most inputs are repetitive or already structured.

The load test should exercise bounded concurrency, backoff, cancellation, and idempotent job handling. A 429 is a capacity signal, not a summary. Preserve its count even when a later retry succeeds, because a queue that completes today can still be approaching tomorrow's limit. Attach the prompt version, model alias, concrete routed model, region, and request identifier to each result — within the privacy policy — so a regression can be reconstructed rather than debated.

Finally, test the escape hatch. Switch the configured model, run the frozen corpus, compare it with the current baseline, and roll back. If that procedure requires editing application code, the compatible endpoint hasn't bought much. If a gateway conflicts with residency or native features, the catch is clear: keep the application interface but route that workload through a dedicated adapter. The goal isn't one key at any cost. It is a summarization system whose quality, location, and failures remain legible as providers change.

## References

- OpenAI Batch API guide: https://platform.openai.com/docs/guides/batch
- openai/whisper, open-source speech recognition: https://github.com/openai/whisper
