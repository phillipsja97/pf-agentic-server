# Morning Briefing Workflow — Design Spec

**Date:** 2026-06-10  
**Status:** Approved

## Overview

A fire-and-forget async workflow that produces a structured morning briefing covering weather, curated news, and stock performance. Triggered via `POST /workflows/briefing`, returns a `job_id` immediately. The caller (hermes agent) polls `GET /workflows/{job_id}` until the job completes and reads the structured result. All user preferences are stored in a JSON config file (`data/briefing_config.json`) and managed via a dedicated config router.

---

## File Structure

New files:

```
workflows/effgen/briefing.py       # agent runner — mirrors research.py
core/storage/briefing_config.py    # read/write data/briefing_config.json
routers/briefing_config.py         # config management endpoints
```

Modified files:

```
schemas/models.py                  # add briefing schemas
routers/workflows.py               # add POST /briefing trigger
config.py                          # add briefing_config_path setting
.env.example                       # document BRIEFING_CONFIG_PATH
```

---

## Config File

Path: `data/briefing_config.json` (configurable via `BRIEFING_CONFIG_PATH` in `.env`).

Created automatically with defaults on first run if missing.

```json
{
  "weather_location": "Nashville, Tennessee",
  "weather_units": "fahrenheit",
  "news_subjects": [],
  "rss_feeds": [],
  "stocks": [],
  "stock_mode": "ticker-view"
}
```

Each stock entry:

```json
{ "ticker": "AAPL", "name": "Apple Inc.", "notes": "optional context for the agent" }
```

---

## Config Management Endpoints

Mounted at `/briefing/config`. All write endpoints return the full updated config.

```
GET    /briefing/config                         # read full config
PUT    /briefing/config/weather                 # set weather_location
PUT    /briefing/config/stock-mode              # switch "ticker-view" / "portfolio-view"

POST   /briefing/config/news-subjects           # add one or more subjects
DELETE /briefing/config/news-subjects/{subject} # remove a subject

POST   /briefing/config/rss-feeds               # add one or more feed URLs
DELETE /briefing/config/rss-feeds               # remove a feed URL (passed in body)

POST   /briefing/config/stocks                  # add one or more stock entries
DELETE /briefing/config/stocks/{ticker}         # remove a stock by ticker
```

---

## Workflow & Agent Design

### Trigger

`POST /workflows/briefing` — no request body required. Creates a job row (`status=pending`), enqueues `run_briefing` as a `BackgroundTask`, returns `job_id`.

### Runner (`workflows/effgen/briefing.py`)

Follows the `research.py` pattern:

- Lazy model singleton with `asyncio.Lock`
- `asyncio.to_thread` wraps the synchronous effgen agent call
- `update_job` manages `pending → running → completed/failed`
- Reads `briefing_config.json` fresh at the start of each job

### Tool Selection (dynamic, based on config)

| Tool | Included when |
|---|---|
| `WeatherTool` | Always |
| `NewsTool` | `news_subjects` is non-empty |
| `HackerNewsTool` | `news_subjects` is non-empty |
| `RSSFeedTool` | `rss_feeds` is non-empty |
| `StockPriceTool` | `stocks` is non-empty |

### Agent Prompt Phases

1. **Gather** — fetch weather for configured location; for each news subject, query all applicable tools requesting 2 results each; fetch price data for all configured tickers
2. **Synthesize** — per news subject, merge results from all tools and deduplicate: if two stories cover the same event, combine into one entry keeping the richer content; for stocks, produce per-ticker snapshots (`ticker-view`) or a single portfolio narrative (`portfolio-view`)
3. **Format** — structure into `BriefingOutput` Pydantic model

Weather prompt specifies: full day forecast, Fahrenheit, broken into morning/afternoon/evening buckets plus current conditions.

---

## Output Structure (`BriefingOutput`)

Stored as the job `result`. Sections absent from config are omitted entirely from the output.

```json
{
  "date": "2026-06-10",
  "weather": {
    "location": "Nashville, Tennessee",
    "current": {
      "temp_f": 72,
      "condition": "Partly cloudy",
      "humidity": "65%",
      "wind": "8 mph SW"
    },
    "forecast": {
      "morning":   { "temp_f": 68, "condition": "Clear",            "summary": "Cool and clear start." },
      "afternoon": { "temp_f": 85, "condition": "Partly cloudy",    "summary": "Warming up with cloud cover." },
      "evening":   { "temp_f": 76, "condition": "Chance of storms", "summary": "Isolated storms possible after 7pm." }
    }
  },
  "news": [
    {
      "subject": "AI",
      "stories": [
        { "title": "...", "summary": "...", "source": "The Verge",   "url": "https://..." },
        { "title": "...", "summary": "...", "source": "Hacker News", "url": "https://..." }
      ]
    }
  ],
  "stocks": {
    "mode": "ticker-view",
    "tickers": [
      { "ticker": "AAPL", "name": "Apple Inc.", "close": 189.45, "change_pct": -2.1, "summary": "Dropped 2.1% alongside broader tech selloff." }
    ]
  }
}
```

`portfolio-view` alternative for `stocks`:

```json
{
  "mode": "portfolio-view",
  "narrative": "Your watchlist had a mixed session. SPY held steady while AAPL and NVDA pulled back on macro pressure..."
}
```

---

## Schemas (additions to `schemas/models.py`)

```python
class StockEntry(BaseModel):
    ticker: str
    name: str
    notes: Optional[str] = None

class BriefingConfig(BaseModel):
    weather_location: str = "Nashville, Tennessee"
    weather_units: str = "fahrenheit"
    news_subjects: list[str] = []
    rss_feeds: list[str] = []
    stocks: list[StockEntry] = []
    stock_mode: str = "ticker-view"  # "ticker-view" | "portfolio-view"

# --- request/response models ---

class AddNewsSubjectsRequest(BaseModel):
    subjects: list[str]

class AddRSSFeedsRequest(BaseModel):
    feeds: list[str]

class RemoveRSSFeedRequest(BaseModel):
    feed: str

class AddStocksRequest(BaseModel):
    stocks: list[StockEntry]

class UpdateWeatherRequest(BaseModel):
    location: str

class UpdateStockModeRequest(BaseModel):
    mode: Literal["ticker-view", "portfolio-view"]

# --- output models ---

class WeatherPeriod(BaseModel):
    temp_f: float
    condition: str
    summary: str

class WeatherCurrent(BaseModel):
    temp_f: float
    condition: str
    humidity: str
    wind: str

class WeatherOutput(BaseModel):
    location: str
    current: WeatherCurrent
    forecast: dict[str, WeatherPeriod]  # keys: morning, afternoon, evening

class NewsStory(BaseModel):
    title: str
    summary: str
    source: str
    url: str

class NewsSubjectOutput(BaseModel):
    subject: str
    stories: list[NewsStory]

class TickerOutput(BaseModel):
    ticker: str
    name: str
    close: float
    change_pct: float
    summary: str

class StocksOutput(BaseModel):
    mode: Literal["ticker-view", "portfolio-view"]
    tickers: Optional[list[TickerOutput]] = None   # ticker-view
    narrative: Optional[str] = None                # portfolio-view

class BriefingOutput(BaseModel):
    date: str
    weather: WeatherOutput                         # always present — WeatherTool always runs
    news: Optional[list[NewsSubjectOutput]] = None # omitted if news_subjects is empty
    stocks: Optional[StocksOutput] = None          # omitted if stocks is empty
```

---

## Error Handling

- If the agent fails, job transitions to `failed` with the error message stored on the job row — same pattern as `research.py`.
- If `briefing_config.json` is missing at job start, it is created with defaults and the job continues.
- Tools that return no results for a given subject (e.g. `HackerNewsTool` for a non-tech subject) are silently skipped — the agent is prompted to treat empty tool results as "no stories available from this source."
