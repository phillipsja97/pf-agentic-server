from fastapi import APIRouter

from config import settings
from core.storage.briefing_config import load_config, save_config
from schemas.models import (
    AddNewsSubjectsRequest,
    AddRSSFeedsRequest,
    AddStocksRequest,
    BriefingConfig,
    RemoveRSSFeedRequest,
    UpdateStockModeRequest,
    UpdateWeatherRequest,
)

router = APIRouter(prefix="/briefing", tags=["briefing-config"])


@router.get("/config", response_model=BriefingConfig)
async def get_briefing_config() -> BriefingConfig:
    return load_config(settings.briefing_config_path)


@router.put("/config/weather", response_model=BriefingConfig)
async def update_weather(request: UpdateWeatherRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.weather_location = request.location
    save_config(settings.briefing_config_path, config)
    return config


@router.put("/config/stock-mode", response_model=BriefingConfig)
async def update_stock_mode(request: UpdateStockModeRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.stock_mode = request.mode
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/news-subjects", response_model=BriefingConfig)
async def add_news_subjects(request: AddNewsSubjectsRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = set(config.news_subjects)
    for s in request.subjects:
        if s not in existing:
            config.news_subjects.append(s)
            existing.add(s)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/news-subjects/{subject}", response_model=BriefingConfig)
async def remove_news_subject(subject: str) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.news_subjects = [s for s in config.news_subjects if s != subject]
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/rss-feeds", response_model=BriefingConfig)
async def add_rss_feeds(request: AddRSSFeedsRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = set(config.rss_feeds)
    for f in request.feeds:
        if f not in existing:
            config.rss_feeds.append(f)
            existing.add(f)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/rss-feeds", response_model=BriefingConfig)
async def remove_rss_feed(request: RemoveRSSFeedRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.rss_feeds = [f for f in config.rss_feeds if f != request.feed]
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/stocks", response_model=BriefingConfig)
async def add_stocks(request: AddStocksRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = {s.ticker.upper() for s in config.stocks}
    for stock in request.stocks:
        upper = stock.ticker.upper()
        if upper not in existing:
            config.stocks.append(stock.model_copy(update={"ticker": upper}))
            existing.add(upper)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/stocks/{ticker}", response_model=BriefingConfig)
async def remove_stock(ticker: str) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.stocks = [s for s in config.stocks if s.ticker.upper() != ticker.upper()]
    save_config(settings.briefing_config_path, config)
    return config
