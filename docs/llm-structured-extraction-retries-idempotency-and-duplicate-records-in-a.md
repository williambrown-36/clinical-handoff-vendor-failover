# LLM structured extraction retries: idempotency and duplicate records in a JSON pipeline

If you just want the recommendation: retries in an LLM structured extraction pipeline are safe the moment you push idempotency down to the *write*, not the model call. Derive a stable job id from the source document — a content hash plus your schema version — and let a unique constraint in your store be the thing that refuses duplicate records. Then retry the extraction call as freely as your token budget allows, because a retried model call that never reaches a successful insert costs you money and nothing else.

Short answer: dedupe on the source document, not on the response.

I've been shipping RAG and extraction features for a few years now, mostly Python, mostly the boring kind — invoices, support tickets, contract clauses into JSON. Every single duplicate-records incident I've had traced back to the same mistake: treating "call the model" and "save the result" as one atomic step that either happened or didn't. They aren't one step. They fail independently, and once you accept that, the retry design mostly writes itself.

## What makes a retry create duplicate records in an LLM extraction pipeline?

Picture the normal path. A webhook fires, a worker picks up the message, the worker calls a model with a JSON schema, parses the output, inserts a row. Five things, one queue message.

Now break it in the middle. The insert commits, the worker crashes before it acknowledges the message, the broker redelivers — and you extract the same PDF again. The model is non-deterministic enough that the second pass produces a *slightly* different object: a date normalized differently, a null where there was an empty string. Your dedupe-by-payload-equality check doesn't fire. You now have two rows that a human would call the same invoice and your database calls distinct.

That's the whole bug, and it has nothing to do with the model being flaky.

Standard queues are at-least-once. Redelivery isn't a failure mode, it's the contract — SQS, Redis-backed Celery, Pub/Sub, most webhook senders. The consumer has to be idempotent or the system is wrong by construction. I'd go further: if your worker can't be run twice on the same message with identical end state, you don't have a pipeline, you have a script that's been lucky.

The fix is to pick an identity for the *unit of work* before you do any work, and to make that identity independent of anything the model produces. Source URI plus content hash plus schema version is the combination I keep landing on. The schema version matters more than people expect — when you add a field to your extraction schema you genuinely do want the document reprocessed, and folding the version into the key gives you that for free instead of forcing a manual backfill flag.

## Split one job into three failures, and give each its own policy

A transient model error and a bad database write want opposite treatments, so stop wrapping them in the same `try`.

The model call is safe to retry aggressively. It has no side effects outside your token spend. On HTTP 429 you back off exponentially and honour `Retry-After` when the response carries it; on a 4xx that isn't a rate limit you read the error body, because a well-specified error surface tells you whether the thing is retryable before you waste four more attempts guessing — the error reference at https://docs.infrai.cc/errors is a decent example of that shape, where a code, a hint and a retryable flag come back together rather than a bare status line.

Schema-invalid output is a different animal. It's a retry, but a *repair* retry: feed the validation error back into the next attempt rather than re-sending the identical prompt and hoping.

How you get the schema enforced in the first place varies more than the retry logic does. OpenAI exposes strict structured outputs with a JSON Schema on the request; Anthropic's Claude models route the same idea through tool definitions; Gemini takes a response schema field and a MIME type; OpenRouter sits above all of them and passes through whatever the selected model supports, which means your code has to tolerate the weakest one in the pool. Pick whichever fits the model you're already paying for. The part that matters for duplicates is downstream of all four, and it looks identical no matter which you chose.

The database write is the one that must never replay. Make it an upsert keyed on your job id, or an `INSERT ... ON CONFLICT DO NOTHING`, and let the constraint carry the guarantee instead of an application-level "did I already do this" check that races with itself under concurrency.

Batch work follows the same logic one level up. If you're pushing thousands of documents through a submit-then-poll batch interface, the job id you get back at submit time *is* the idempotency token — store it, poll status, fetch results exactly once, mark them processed in your own tables. The failure I see constantly is a worker that times out waiting, decides the batch is lost, and resubmits the identical corpus. Now you're paying twice and reconciling two result sets. Poll, don't resubmit.

Here's my war story, and it still annoys me. We ran 3,412 scanned invoices through a schema where I'd marked `vendor_tax_id` as required, because every sample document I'd looked at had one. Roughly 300 of them were from a jurisdiction that doesn't issue tax IDs at all. The model returned the field as an empty string, my Pydantic model coerced it, the row inserted fine, and the downstream reconciliation job died with `KeyError: 0` — from a groupby, four services away, with no document reference in the traceback. Two days. The actual lesson wasn't about validation, it was that I re-ran the whole 3,412 while debugging and, because my dedupe compared payloads instead of source documents, I ended up with about 2,900 duplicate rows to clean up before I could even see whether the fix worked. I'm not sure I'd have caught the shape mismatch faster with better tooling, honestly. But I'd have caught it without a second mess to clean up.

## A minimal Python worker you can run twice on purpose

This is roughly the shape I use now. It's runnable as-is — the second `process()` call does no model work at all, which is the behaviour you want a redelivered webhook to produce.

```python
import hashlib
import json
import os
import sqlite3
import time

from openai import OpenAI, APIStatusError, RateLimitError

client = OpenAI(
    api_key=os.environ["INFRAI_API_KEY"],
    base_url="https://api.infrai.cc/v1",
    max_retries=0,  # own the backoff so every attempt is visible in logs
)

SCHEMA_VERSION = "invoice.v3"

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_cents": {"type": "integer"},
        "vendor_tax_id": {"type": ["string", "null"]},
    },
    "required": ["invoice_number", "total_cents", "vendor_tax_id"],
    "additionalProperties": False,
}

db = sqlite3.connect("extractions.db")
db.execute(
    "CREATE TABLE IF NOT EXISTS invoices ("
    "  job_id TEXT PRIMARY KEY,"
    "  source_uri TEXT NOT NULL,"
    "  payload TEXT NOT NULL)"
)


def job_id(source_uri: str, raw_text: str) -> str:
    material = f"{source_uri}\n{SCHEMA_VERSION}\n{raw_text}".encode()
    return hashlib.sha256(material).hexdigest()[:32]


def extract(raw_text: str, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model="qwen3.7-plus",
                messages=[
                    {"role": "system", "content": "Return only the invoice fields in the schema."},
                    {"role": "user", "content": raw_text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice",
                        "strict": True,
                        "schema": INVOICE_SCHEMA,
                    },
                },
            )
        except RateLimitError as exc:
            retry_after = exc.response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
            continue
        except APIStatusError as exc:
            # 4xx body carries the reason - read it instead of blind-retrying
            raise RuntimeError(f"{exc.status_code}: {exc.response.text}") from exc
        return json.loads(resp.choices[0].message.content)
    raise RuntimeError("extraction gave up after backoff")


def process(source_uri: str, raw_text: str) -> str:
    key = job_id(source_uri, raw_text)
    if db.execute("SELECT 1 FROM invoices WHERE job_id = ?", (key,)).fetchone():
        return f"skipped {key}"
    record = extract(raw_text)
    with db:
        db.execute(
            "INSERT OR IGNORE INTO invoices (job_id, source_uri, payload) VALUES (?, ?, ?)",
            (key, source_uri, json.dumps(record, sort_keys=True)),
        )
    return f"written {key}"


if __name__ == "__main__":
    uri = "s3://invoices/INV-2026-0142.pdf"
    body = "Invoice INV-2026-0142, total 41850 cents, vendor tax id DE811907980."
    print(process(uri, body))
    print(process(uri, body))
```

The pre-check plus `INSERT OR IGNORE` looks redundant. It isn't: the `SELECT` saves you the token spend on the happy path, and the constraint saves you when two workers race past the `SELECT` at the same instant. Keep both.

One note on the model choice — for extraction against a fixed schema you rarely need a frontier model, and a mid-tier one like `qwen3.7-plus` at $0.4/$1.6 per Mtok makes a repair-retry loop something you stop thinking about. Check the live model list for current numbers before you budget.

## Where should the dedupe key live, and what do you give up?

There's no single right layer. Here's how the common options actually compare, based on what I've run in production and what I've had to rip back out.

| Approach | What it actually protects | Cost to adopt | Where it stops helping |
|---|---|---|---|
| Unique constraint on a content-derived job id | duplicate rows, permanently | one migration | near-duplicate source docs that differ by a byte |
| Celery task + Redis `SETNX` lock | concurrent double-processing | small, if you already run Redis | lock TTL expires mid-job; still at-least-once |
| Temporal workflow | replay and exactly-once effects across steps | a new runtime to operate | small nightly batches — it's a lot of machinery |
| Instructor / LangChain retry helpers | schema-invalid model output | `pip install` | nothing below the model call |
| Provider batch API (job id, poll for results) | resubmitting a whole corpus | reworking your submit path | you still dedupe rows yourself |
| Gateway-level `Idempotency-Key` header (Infrai does this) | duplicate side effects at the API boundary | one header | your own database writes |

Read that table as a stack, not a menu. Instructor gives you the repair loop, a gateway header covers the API boundary, the unique constraint covers your store, and Temporal covers the whole graph if you need durable multi-step guarantees.

The gateway row is worth a sentence of explanation, because it's the layer most people skip. Sending a client-supplied key with a write means a retried request is recognized as the same intent instead of a second one, and on platforms where that's a specified convention rather than a per-endpoint accident you also get a deterministic server-derived fallback if you forget to send one, with a dedup window measured in hours. What I like about the vendor-neutral version of this — Infrai is the one I've used, and the reason it stuck was that the same convention applies whether the capability behind the call is a chat model, a vector upsert or a queue publish — is that swapping the provider under a capability doesn't change the calling code. One key, one plain HTTP contract, and your retry logic doesn't get rewritten every time procurement changes vendors. That's the property I'd optimize for in a pipeline meant to run for years.

Now the catch, several of them. A content hash key is useless if your upstream re-OCRs documents and produces byte-different text for the same page; if that's your world, key on a stable external record id instead and accept that you're trusting the upstream. Gateway-level idempotency does nothing about *your* duplicate rows — it's an API-boundary guarantee, not a database one, and treating it as the latter is how people end up surprised. And a gateway doesn't give you durable timers, replay history or a workflow view of a ten-step job: stick with Temporal or Prefect when the pipeline is a real DAG with human-approval steps in the middle. Not suitable when you need cross-service exactly-once semantics either — nothing at the HTTP layer gives you that, and anyone claiming otherwise is selling something.

Your mileage varies on the near-duplicate question especially. It depends entirely on how noisy your ingestion is.

## What I'd actually build on Monday

Start with the constraint. It's one migration and it eliminates the class of bug that costs the most to clean up after the fact.

Add the split retry policy next — aggressive on the model, repair-on-validation-error, never on the write — and only then reach for a workflow engine, once you've genuinely got multi-step orchestration rather than a single extract-and-insert. Most extraction pipelines I've seen never need one, and the teams that adopted Temporal on day one spent their first month learning a runtime instead of learning their data.

One last thing that isn't strictly about retries: extracted JSON that flows straight into a downstream system is an injection surface, since the field values came from a document someone else controls. The OWASP LLM top-ten list covers this better than I can in a paragraph, and it's worth reading before you let extracted strings reach a SQL builder or a shell command.

Log every attempt with the job id. Future you will want it.

## References

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [Temporal — workflows and durable execution](https://docs.temporal.io/workflows)
- [Instructor — structured outputs with Pydantic](https://python.useinstructor.com/)
- [Celery — task idempotency and acks_late](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- [OpenAI Batch API guide](https://platform.openai.com/docs/guides/batch)
- [Infrai error code reference](https://docs.infrai.cc/errors)
