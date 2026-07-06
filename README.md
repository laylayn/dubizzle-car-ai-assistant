# dubizzle Cars AI Assistant

A conversational prototype for searching a provided car inventory, asking
follow-up questions, comparing listings, remembering returning users, qualifying
leads, and booking viewings.

## Quick setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
# Add your Google AI Studio key to GEMINI_API_KEY in .env
uv sync
```

Start the FastAPI backend:

```bash
uv run uvicorn main:app --reload
```

In a second terminal, start the Streamlit interface:

```bash
uv run streamlit run app.py
```

The interface uses `http://127.0.0.1:8000` by default. 

## Why this setup

I chose Streamlit because it provides a simple interactive chat experience and
keeps the client separate from the FastAPI backend. Instead of a large agent
framework, the project uses a small custom orchestration layer with the Gemini
SDK so routing, grounding, and fallbacks remain explicit. Inventory retrieval
combines query planning, pandas filters, listing-text matching, and reusable
price/mileage/feature extractors. Short-term state is stored per session in
memory, while SQLite stores returning-user preferences, meaningful interactions,
and an LLM-assisted summary with deterministic fallback; leads and confirmed
bookings are written to CSV.

## Implementation

Rule-based guardrails and deterministic extraction run first. Gemini is optional
and helps extract intent and filters when needed, while the backend executes
searches against the supplied dataset and gives the model only retrieved listing
facts. Session memory resolves references such as “the first one” or “it,” while
guardrails decline unrelated requests. Booking rules, lead qualification,
comparisons, and long-term memory updates are handled by dedicated backend
services.

Some features were intentionally left outside the scope of this take-home prototype. I considered adding hybrid RAG retrieval to make the search more semantic, but for a small dataset, I kept the retrieval mostly structured because it made the system easier to test, debug, and trust. 

In a fuller version, I would add production authentication, database-backed inventory management, CRM integration, automated WhatsApp or email confirmations, an admin dashboard, personalised recommendations, top-deal suggestions, price comparisons, finance estimates, and suggested conversation starters to make the assistant easier to use.


## Examples

### Multi-turn inventory conversation

[Watch the multi-turn inventory conversation](screenshots/multi_turn_chat.mp4)

### Returning-user memory

[Watch the returning-user memory demo](screenshots/returning_user_memory.mp4)
