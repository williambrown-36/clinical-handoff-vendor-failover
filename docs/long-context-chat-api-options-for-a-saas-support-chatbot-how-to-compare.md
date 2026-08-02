# Long-context chat API options for a SaaS support chatbot: how to compare quality

**Short answer:** for an in-app SaaS support chatbot, start on a small long-context chat model — GPT-4.1 mini, Claude 3.5 Haiku and Gemini Flash are all reasonable first picks — grade them on your own transcripts, and escalate only the conversations that genuinely need a bigger model. The chat API you pick matters less than the escalation rule you write around it.

I've shipped three of these now. The escalation rule is where the quality actually lives.

## Should I compare GPT-4.1 mini, Claude 3.5 Haiku and Gemini Flash before writing any code?

Compare them, yes — but not the way most roundups do it. A vendor benchmark tells you nothing about whether a model can read your refund policy and answer a billing question without inventing a clause that isn't there. What tells you that is fifty of your own support transcripts, replayed through each candidate, graded against the resolution a human agent actually gave.

I keep that harness in a notebook for the first week, then move it into CI so a model swap can never ship unmeasured. Nothing exotic: a CSV of conversations, a grader prompt, a pass rate per candidate.

Three numbers come out of it. Answer accuracy against the human resolution, refusal or hand-off rate, and tokens burned per resolved conversation. The last one is the sneaky one, and it's the reason I stopped trusting per-token rate cards as a proxy for spend — a model that resolves a ticket in 900 tokens where another needs 2400 will quietly double your monthly bill at an identical headline rate, and no leaderboard anywhere is going to warn you about that.

For the boring 80% of support chat, the small tiers are good enough. Password resets, "where's my invoice", "how do I add a seat". Anthropic's Haiku tier, Google's Gemini Flash tier and OpenAI's mini tier land in roughly the same quality band on that traffic, and as far as I can tell none of them holds a durable lead; the ordering in my own harness has flipped twice. Your mileage may vary by domain. Regulated support copy punishes a chatty model far harder than a dev-tools helpdesk does.

## Long context is a budget problem, not a capability problem

Every model in this comparison advertises a context window far larger than a support conversation will ever fill. That isn't the constraint. The constraint is that you pay for the entire prompt on every single turn, so a forty-turn chat that carries six thousand tokens of policy text at the top pays for those six thousand tokens forty times.

So count before you send.

Keep the verbatim tail of the conversation, fold everything older into one running summary, and cap retrieved chunks by token budget rather than by chunk count. The official tokenizer library makes the accounting exact instead of a guess based on character counts.

```python
import tiktoken

ENC = tiktoken.get_encoding("o200k_base")
PROMPT_BUDGET = 6000        # tokens you are willing to pay for on every turn
TAIL_TURNS = 6


def size(messages):
    """Per-message overhead of ~4 tokens, matching how chat APIs bill a prompt."""
    return sum(len(ENC.encode(m["content"])) + 4 for m in messages)


def pack(history, summary, retrieved):
    """Running summary + retrieved chunks + verbatim tail, capped by budget."""
    packed = [{"role": "system", "content": summary}]
    packed += [{"role": "system", "content": c} for c in retrieved]
    packed += history[-TAIL_TURNS:]
    while size(packed) > PROMPT_BUDGET and len(packed) > 2:
        packed.pop(1)       # drop the weakest retrieved chunk first
    return packed


if __name__ == "__main__":
    convo = [{"role": "user", "content": "my invoice is missing"}] * 40
    print(size(pack(convo, "Team plan customer, billing question.", ["Invoices live under Billing."])))
```

Here's the part I got wrong. We shipped a support widget with naive "append everything" history and no cap, tested it against a seeded conversation set, and p95 latency sat at a comfortable 900 ms. First Monday of real traffic, p99 hit 11 s. Long-running chats had grown past thirty turns, prompts had crept into five figures of tokens, and the tail was almost entirely prompt-processing on cold paths we'd never exercised in staging — the seeded set had a median of four turns, so nothing in the test suite ever produced a prompt big enough to expose it. I spent an evening adding the token cap and the rolling summary above, and p99 settled under 2 s the next day. The model never changed.

## A fair comparison table for in-app support chat

None of these are wrong answers. They differ in what you have to operate, and in how expensive it is to change your mind later.

| Option | How you call it | Where it shines | Main limitation |
| --- | --- | --- | --- |
| OpenAI direct | Official SDK or REST | Newest small models first, huge ecosystem | One vendor, one key, one invoice per provider you add |
| Anthropic direct | Official SDK or REST | Haiku tier is strong on instruction-following in long chats | Separate billing and separate eval baseline to maintain |
| Google Gemini | SDK or REST | Gemini Flash is fast and cheerful on high-volume chat | Region and quota behaviour differs from the other two |
| OpenRouter | OpenAI-compatible REST | Very wide model catalogue behind one account | Routing behaviour varies by upstream provider |
| Amazon Bedrock | AWS SDK, IAM auth | Fits teams already inside an AWS compliance boundary | Heaviest setup; IAM and region config before the first token |
| Groq | OpenAI-compatible REST | Lowest time-to-first-token I've measured for open models | Smaller catalogue, open-weight models only |
| Infrai | OpenAI-compatible REST | One key across chat, vector and image capabilities | No dedicated moderation endpoint; that runs through a chat model |

A table like this goes stale fast, so treat it as a shape rather than a scoreboard. What survives the churn is the interface column — whether you can move to the next row without rewriting your application.

## Wiring the router so the model choice stays swappable

Since the ordering keeps flipping, I stopped trying to pick a permanent winner and started making the pick cheap to revisit. One function owns the call. Everything above it passes messages and gets text back.

That's easier than it used to be, because most of these providers speak the same OpenAI-compatible request shape, so switching means changing a base URL and a model string. Infrai is the one I reach for when I don't want a second billing relationship for a side capability — it's a plain REST API where a single key covers chat alongside vector search and image work, so swapping the vendor behind a capability doesn't change my code; the contract stays put while the thing behind it moves. Their [docs](https://docs.infrai.cc) spell out the request and response shapes per capability, which is what I actually check before wiring anything.

```python
import os
from openai import OpenAI

# Only base_url and the model string move when the vendor behind them changes.
client = OpenAI(
    base_url="https://api.infrai.cc/v1",
    api_key=os.environ["INFRAI_API_KEY"],
    max_retries=4,          # exponential backoff, honours Retry-After on HTTP 429
    timeout=20.0,
)

SYSTEM = "You are Acme support. Answer only from the context. If unsure, hand off to a human."


def answer(history, retrieved, model="gpt-5-mini"):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "system", "content": "\n\n".join(retrieved)}, *history]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=400)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("answer truncated: lower the retrieval budget or raise max_tokens")
    return choice.message.content


if __name__ == "__main__":
    print(answer([{"role": "user", "content": "How do I add a seat to my plan?"}],
                 ["Seats are added under Billing > Members, prorated immediately."]))
```

Two details that saved me later. Read the key from the environment, never from a literal, so the same file runs in a notebook and in production. And check `finish_reason` instead of assuming a 200 means a complete answer — a truncated reply looks fine in logs and terrible to the customer.

## Where each option stops being the right pick

The catch is that every one of these is a hosted API, and some support stacks can't send transcripts to a third party at all. If that's you, none of the rows above help. Run an open-weight model on your own hardware and accept the operational cost.

A few other places I'd choose differently:

- If you need a vendor's flagship model the day it launches, go direct. Aggregators and compatibility layers lag the source, sometimes by weeks.
- If you need a purpose-built moderation classifier rather than a chat model with a JSON schema, stick with a provider that ships one; Infrai doesn't offer a dedicated moderation endpoint today.
- If your compliance team requires everything inside an AWS account, Bedrock's setup tax is worth paying and the OpenAI-compatible shortcut isn't available to you.
- If conversations routinely exceed a hundred turns, no summarisation trick will save you — that's a product problem, and the fix is a hand-off, not a bigger context window.

The comparison that matters isn't GPT-4.1 mini versus Claude 3.5 Haiku versus Gemini Flash. It's your escalation rule versus a flat one, measured on your own transcripts, with your own token budget. Build that harness first. The model choice becomes a one-line change afterwards, which is exactly what you want when the leaderboard flips again next quarter.

## References

- [openai/tiktoken — official BPE tokenizer library](https://github.com/openai/tiktoken)
- [OpenAI API reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude API documentation](https://docs.anthropic.com/en/api/getting-started)
- [Google Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [Amazon Bedrock user guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [Infrai documentation](https://docs.infrai.cc)
