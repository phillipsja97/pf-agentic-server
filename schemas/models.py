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
    user_id: Optional[str] = None
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

class WeatherCurrent(BaseModel):
    temp_f: float
    condition: str
    humidity: str
    wind: str


class WeatherForecast(BaseModel):
    temp_high_f: float
    temp_low_f: float
    condition: str
    precipitation_chance: Optional[int] = None
    wind_speed_max: Optional[float] = None
    summary: str


class WeatherOutput(BaseModel):
    location: str
    current: WeatherCurrent
    forecast: WeatherForecast


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


# --- learning plan workflow models ---

class LearningPlanChild(BaseModel):
    id: int
    name: str
    notes: str


class LearningPlanGapNode(BaseModel):
    id: str
    name: Optional[str] = None
    description: str


class LearningPlanLog(BaseModel):
    date: str
    description: str


class LearningPlanRequest(BaseModel):
    child: LearningPlanChild
    week_start: str
    focus_note: str = ""
    gap_nodes: list[LearningPlanGapNode] = []
    recent_logs: list[LearningPlanLog] = []


# --- RAG workflow models ---

class RagIngestRequest(BaseModel):
    source: str          # path to file or directory to ingest
    collection_id: str   # slug identifying this knowledge base, e.g. "my-docs"


class RagChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RagChatRequest(BaseModel):
    collection_id: str
    message: str
    history: list[RagChatTurn] = []


# --- soccer analyst workflow models ---

class SoccerQueryPlan(BaseModel):
    sql: str
    chart_type: str
    x_column: str
    y_column: str
    title: str


class SoccerAnalystRequest(BaseModel):
    question: str


class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]
    chart: dict
    row_count: int
