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
    series_column: Optional[str] = None
    title: str


class ComparisonQueryPlan(BaseModel):
    is_comparison: Literal[True]
    player_a: str
    season_a: int
    player_b: str
    season_b: int
    title: str


class SoccerAnalystRequest(BaseModel):
    question: str
    league: Optional[str] = None


class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]
    chart: dict
    row_count: int
    comparison: Optional[dict] = None


class CfbAnalystRequest(BaseModel):
    question: str
    power4_only: bool = True


# --- cfb matchup predictor models ---

class CfbMatchupRequest(BaseModel):
    home_team: str
    away_team: str
    neutral_site: bool = False
    season: Optional[int] = None        # default: latest season in cfb_team_profile
    through_week: Optional[int] = None  # default: latest fetched week of that season


class CfbTeamRecord(BaseModel):
    team: str
    conference: Optional[str] = None
    wins: int
    losses: int
    games_played: int
    # Rendered brand colour, already paired against the opponent and lifted
    # to stay legible on the dark chart background. See workflows/cfb/colors.py.
    color: str
    # CORE efficiency as of the same week, in points per 100 plays. Offence is
    # points created above average, defence is points *allowed* above average
    # (so lower is better), overall is offence minus defence. Null when the
    # ratings table has not been built.
    core_off: Optional[float] = None
    core_def: Optional[float] = None
    core_overall: Optional[float] = None
    # Percentile among FBS teams at the same week, outward-is-better for all
    # three, so the UI never has to know which way a metric points.
    core_off_pct: Optional[float] = None
    core_def_pct: Optional[float] = None
    core_overall_pct: Optional[float] = None
    # What SP+ (Connelly) and FPI (ESPN) said about this team as of the same
    # week. Null until the weekly snapshot has run for that week.
    sp_rating: Optional[float] = None
    sp_rank: Optional[float] = None
    fpi: Optional[float] = None
    fpi_rank: Optional[float] = None


class CfbPrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    projected_home_points: float
    projected_away_points: float
    expected_margin: float


class CfbCalibrationBin(BaseModel):
    lo: float
    hi: float
    n: int
    predicted: Optional[float] = None
    actual: Optional[float] = None


class CfbModelRecord(BaseModel):
    accuracy: float
    brier: float
    log_loss: float
    elo_baseline_accuracy: float
    elo_baseline_brier: float
    calibration_bins: list[CfbCalibrationBin]
    trained_through_season: int
    shrink_k: int
    # Weeks 1-4 alone, where the model leans hardest on last season.
    early_accuracy: Optional[float] = None
    early_elo_accuracy: Optional[float] = None
    # Closing-line accuracy on the same held-out games. The strongest public
    # predictor; null when betting lines have not been ingested.
    market_accuracy: Optional[float] = None
    # SP+ and FPI are published only as end-of-season numbers, so each is a
    # bracket: "prior" uses last season's final all season (leak-free, never
    # updates, understates them); "final" uses this season's final (has seen
    # every result, overstates them). In-season truth sits between.
    sp_prior_accuracy: Optional[float] = None
    sp_final_accuracy: Optional[float] = None
    fpi_prior_accuracy: Optional[float] = None
    fpi_final_accuracy: Optional[float] = None


class CfbEdge(BaseModel):
    stat: str
    home_value: Optional[float]
    away_value: Optional[float]
    diff: Optional[float]           # home - away, in the stat's own unit
    favors: Literal["home", "away", "even"]
    unit: str


class CfbH2HGame(BaseModel):
    season: int
    week: int
    season_type: str
    date: Optional[str] = None
    home_team: str
    away_team: str
    home_points: int
    away_points: int


class CfbMatchupResponse(BaseModel):
    season: int
    through_week: int
    data_through_date: Optional[str]
    neutral_site: bool
    home: CfbTeamRecord
    away: CfbTeamRecord
    prediction: CfbPrediction
    model: CfbModelRecord
    comparison: dict                # RadarFigure: {title, axes, series[{name, values}]}
    edges: list[CfbEdge]
    head_to_head: list[CfbH2HGame]


class CfbSlateGame(BaseModel):
    game_id: int
    week: int
    start_date: Optional[str] = None
    home_team: str
    away_team: str
    neutral_site: bool
    home_win_prob: float
    projected_home_points: float
    projected_away_points: float
    home_color: str
    away_color: str


class CfbSlateResponse(BaseModel):
    season: int
    week: int
    through_week: int
    data_through_date: Optional[str]
    games: list[CfbSlateGame]


class CfbBacktestGame(BaseModel):
    """One scored game: what was predicted before kickoff against the final."""
    game_id: int
    week: int
    date: Optional[str] = None
    home_team: str
    away_team: str
    neutral_site: bool
    # "stored" = read back from cfb_predictions, i.e. what the slate served.
    # "replayed" = rebuilt from the profile snapshot as of through_week.
    source: Literal["stored", "replayed"]
    through_week: int
    # The model that made it: cfb_model.fitted_at.
    fitted_at: Optional[str] = None
    home_win_prob: float
    projected_home_points: float
    projected_away_points: float
    expected_margin: float
    actual_home_points: int
    actual_away_points: int
    actual_margin: int
    actual_total: int
    margin_error: float
    total_error: float
    straight_up_correct: bool


class CfbBacktestMetrics(BaseModel):
    games: int
    straight_up_accuracy: Optional[float] = None
    brier: Optional[float] = None
    margin_mae: Optional[float] = None
    margin_median_abs_error: Optional[float] = None
    margin_bias: Optional[float] = None
    within_3: Optional[float] = None
    within_7: Optional[float] = None
    within_10: Optional[float] = None
    within_14: Optional[float] = None
    total_mae: Optional[float] = None
    total_bias: Optional[float] = None
    home_score_mae: Optional[float] = None
    home_score_bias: Optional[float] = None
    away_score_mae: Optional[float] = None
    away_score_bias: Optional[float] = None


class CfbBacktestWeek(CfbBacktestMetrics):
    week: int


class CfbBacktestResponse(BaseModel):
    season: int
    week: Optional[int] = None
    # The model the replayed rows were computed with: cfb_model.fitted_at.
    fitted_at: Optional[str] = None
    trained_through_season: Optional[int] = None
    data_through_date: Optional[str] = None
    stored_games: int
    replayed_games: int
    # Completed games with no as-of profile snapshot to predict from.
    skipped_games: int
    overall: CfbBacktestMetrics
    by_week: list[CfbBacktestWeek]
    games: list[CfbBacktestGame]


class CfbTeam(BaseModel):
    team: str
    conference: Optional[str] = None


class CfbIngestRequest(BaseModel):
    year: Optional[int] = None


class CfbIngestStatus(BaseModel):
    state: Literal["idle", "running", "completed", "failed"] = "idle"
    year: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    stage: Optional[str] = None
    budget: Optional[str] = None
    exit_code: Optional[int] = None
    log_tail: list[str] = []
