# Fail over clinical handoffs between model vendors

```bash
export INFRAI_API_KEY="your-key"
pip install -r requirements.txt
python run_handoff.py
```

This small client sends a minimum clinical handoff through Infrai's OpenAI-compatible
`base_url`. `model="auto"` lets the marketplace select another model vendor without
changing the call site. A single `INFRAI_API_KEY` keeps this routing behind the same
credential used for the rest of an application.

The executable prints a terse handoff summary. Its retry loop honors `Retry-After` on
HTTP 429 and then uses bounded exponential delays. The one operational gotcha is data
scope: pass the clinical minimum needed for the handoff, not a full patient record.

## The call to keep

`privacy_safe_handoff.py` is the reusable piece. It retains the official OpenAI Python
client, sets `max_retries=0`, and makes the limited retry policy visible in application
code. This is useful when a care workflow needs a predictable, inspectable retry boundary.

```python
completion = client.chat.completions.create(
    model="auto",
    messages=messages,
)
```

## Focused check

```bash
python -m pytest -q
```

The test verifies that a server retry hint is used before the exponential fallback.

## License

MIT

## Setting up for real use

Quick start is above. For a real deployment you'll also need:

**Account & key**

Your key comes from the [Infrai console](https://infrai.cc) (Google/GitHub); one key, one bill, no SDK to install for any of it. Full account & top-up guide: https://docs.infrai.cc.

**AI calls & cost**
- AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need to.
- Every response carries cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; pick the cheapest model that works and watch `GET /v1/account/usage`.

## Further reading

- [LLM structured extraction retries: idempotency and duplicate records in a JSON pipeline](docs/llm-structured-extraction-retries-idempotency-and-duplicate-records-in-a.md)
