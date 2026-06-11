from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ResearchRequest(BaseModel):
    query: str
    depth: str = "standard"


class CodingRequest(BaseModel):
    idea: str
    slug: Optional[str] = None


class JobCreatedResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    input: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str


# --- briefing config models ---

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
    stock_mode: Literal["ticker-view", "portfolio-view"] = "ticker-view"


# --- briefing config request models ---

class UpdateWeatherRequest(BaseModel):
    location: str


class UpdateStockModeRequest(BaseModel):
    mode: Literal["ticker-view", "portfolio-view"]


class AddNewsSubjectsRequest(BaseModel):
    subjects: list[str]


class AddRSSFeedsRequest(BaseModel):
    feeds: list[str]


class RemoveRSSFeedRequest(BaseModel):
    feed: str


class AddStocksRequest(BaseModel):
    stocks: list[StockEntry]


# --- briefing output models ---

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
    forecast: dict[str, WeatherPeriod]


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
    tickers: Optional[list[TickerOutput]] = None
    narrative: Optional[str] = None


class BriefingOutput(BaseModel):
    date: str
    weather: WeatherOutput
    news: Optional[list[NewsSubjectOutput]] = None
    stocks: Optional[StocksOutput] = None
