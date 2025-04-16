import pandas as pd
import numpy as np
import spacy
import requests
import logging
from .base_strategy import BaseStrategy

# Configure logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set as needed; INFO or DEBUG for more detail

#------------------------------------------------------------------------------------------------------------------
# LLM News Strategy with Automated Headline Fetching, Vector Similarity, and Sophisticated Thresholds
# This strategy fetches the top headlines from a news API (NewsAPI) for the Financial Times,
# then analyzes the headlines using vector-based similarity (via spaCy embeddings) to gauge overall
# market sentiment. The sentiment is classified into five categories:
#
#   - "Strongly Bullish": average_diff > 0.10
#   - "Bullish": 0.05 < average_diff <= 0.10
#   - "Neutral": -0.05 <= average_diff <= 0.05
#   - "Bearish": -0.10 <= average_diff < -0.05
#   - "Strongly Bearish": average_diff < -0.10
#
# Trading Decisions:
#   - For both "Strongly Bullish" and "Bullish": if there is no current position, generate a BUY signal.
#   - For both "Bearish" and "Strongly Bearish": if a position exists, generate a SELL signal.
#   - Neutral sentiment results in no trading action.
#------------------------------------------------------------------------------------------------------------------
class LLMNewsStrategy(BaseStrategy):
    def __init__(self, tickers=None, news_count=15, news_api_key="78a9ce1b9eba4a5fba164b235977b0f2"):
        """
        Initialize the LLM News strategy with vector similarity, automated headline fetching,
        enhanced error logging, and sophisticated sentiment thresholds.

        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols to trade. Defaults to ["SPY"] if not provided.
        news_count : int, optional
            Number of top headlines to fetch (default is 15).
        news_api_key : str, optional
            API key for NewsAPI.
        """
        if tickers is None or not tickers:
            tickers = ["SPY"]
        super().__init__(tickers)
        self.news_count = news_count
        self.last_sentiment = None
        self.news_api_key = news_api_key

        # Load the spaCy medium English model with vectors.
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            logger.error("Error loading spaCy model: %s", e)
            raise

        # Reference keywords for bullish and bearish sentiments.
        bullish_keywords = ["bullish", "optimistic", "growth", "record", "soaring", "surge", "rally"]
        bearish_keywords = ["bearish", "pessimistic", "decline", "loss", "fall", "drop", "uncertainty"]

        # Compute the reference centroids based on the keyword vectors.
        try:
            bullish_tokens = [self.nlp(word) for word in bullish_keywords]
            bearish_tokens = [self.nlp(word) for word in bearish_keywords]
            self.bullish_centroid = np.mean([token.vector for token in bullish_tokens], axis=0)
            self.bearish_centroid = np.mean([token.vector for token in bearish_tokens], axis=0)
        except Exception as e:
            logger.error("Error computing reference centroids: %s", e)
            raise

    def _cosine_similarity(self, vec1, vec2):
        """
        Compute the cosine similarity between two vectors.

        Parameters:
        -----------
        vec1 : numpy.array
            First vector.
        vec2 : numpy.array
            Second vector.

        Returns:
        --------
        float
            Cosine similarity between vec1 and vec2.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def fetch_top_headlines(self, source="financial-times"):
        """
        Fetch the top headlines from a specified news source using NewsAPI.

        Parameters:
        -----------
        source : str, optional
            News source identifier (default is "financial-times").

        Returns:
        --------
        list
            A list of news headline strings.
        """
        if not self.news_api_key:
            logger.error("News API key not provided.")
            return []

        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "sources": source,
            "pageSize": self.news_count,
            "apiKey": self.news_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                headlines = [article["title"] for article in data.get("articles", [])]
                return headlines[:self.news_count]
            else:
                logger.error("Failed to fetch headlines, status code: %s", response.status_code)
                return []
        except Exception as e:
            logger.error("Exception while fetching headlines: %s", e)
            return []

    def read_top_news(self):
        """
        Retrieve the top headlines using the NewsAPI integration.

        Returns:
        --------
        list
            A list of news headline strings.
        """
        return self.fetch_top_headlines()

    def analyze_news_sentiment(self, headlines):
        """
        Analyze a list of headlines using vector similarity to determine overall market sentiment.

        For each headline, compute its cosine similarity with bullish and bearish reference centroids.
        The overall sentiment is determined by averaging the differences (bullish similarity minus bearish similarity)
        across all headlines and classifying it into one of five categories.

        Parameters:
        -----------
        headlines : list
            List of news headlines.

        Returns:
        --------
        str
            Overall market sentiment: one of "Strongly Bullish", "Bullish", "Neutral", "Bearish",
            or "Strongly Bearish".
        """
        total_diff = 0
        count = 0

        for headline in headlines:
            try:
                doc = self.nlp(headline)
                sim_bullish = self._cosine_similarity(doc.vector, self.bullish_centroid)
                sim_bearish = self._cosine_similarity(doc.vector, self.bearish_centroid)
                diff = sim_bullish - sim_bearish
                total_diff += diff
                count += 1
            except Exception as e:
                logger.error("Error processing headline '%s': %s", headline, e)
                continue

        average_diff = total_diff / count if count else 0

        # Define sophisticated thresholds for sentiment classification.
        strong_bullish_threshold = 0.10
        bullish_threshold = 0.05
        bearish_threshold = -0.05
        strong_bearish_threshold = -0.10

        if average_diff > strong_bullish_threshold:
            sentiment = "Strongly Bullish"
        elif average_diff > bullish_threshold:
            sentiment = "Bullish"
        elif average_diff < strong_bearish_threshold:
            sentiment = "Strongly Bearish"
        elif average_diff < bearish_threshold:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        logger.info("Average diff: %.4f, classified sentiment: %s", average_diff, sentiment)
        return sentiment

    def calculate_available_capital(self, current_data, total_capital=100000.0):
        """
        Calculate available capital based on current positions.

        Parameters:
        -----------
        current_data : pandas.Series
            Current market data.
        total_capital : float, optional
            Total capital available.

        Returns:
        --------
        float
            Available capital.
        """
        positions_value = 0
        for ticker, quantity in self.positions.items():
            try:
                if ticker in current_data:
                    price = current_data[ticker]
                else:
                    price = current_data[ticker]['Close']
                positions_value += price * quantity
            except Exception as e:
                logger.error("Error calculating position value for ticker %s: %s", ticker, e)
                continue
        available_capital = total_capital - positions_value
        return max(0, available_capital)

    def generate_signals(self, date, data, **kwargs):
        """
        Generate trading signals based on market sentiment derived from live news headlines.

        Parameters:
        -----------
        date : datetime
            Current date.
        data : pandas.DataFrame
            Market data up to the current date.
        kwargs : dict
            Additional parameters (if any).

        Returns:
        --------
        list
            List of signal dictionaries.
        """
        signals = []
        try:
            current_data = data.loc[date]
        except Exception as e:
            logger.error("Error retrieving market data for date %s: %s", date, e)
            return signals

        # Fetch headlines and analyze sentiment.
        headlines = self.read_top_news()
        if not headlines:
            logger.error("No headlines fetched for date %s", date)
            return signals

        sentiment = self.analyze_news_sentiment(headlines)
        self.last_sentiment = sentiment

        ticker = self.tickers[0]
        try:
            if ticker in current_data:
                price = current_data[ticker]
                if isinstance(price, pd.Series):
                    price = pd.to_numeric(price, errors='coerce')
                    price = float(price) if not pd.isna(price) else None
            else:
                price = float(current_data[ticker]['Close'])
        except Exception as e:
            logger.error("Error retrieving price for ticker %s: %s", ticker, e)
            price = None

        if price is None or price <= 0:
            logger.error("Invalid price for ticker %s on date %s", ticker, date)
            return signals

        # Trading decision based on sophisticated sentiment:
        # For both "Strongly Bullish" and "Bullish": execute BUY if no position exists.
        # For both "Bearish" and "Strongly Bearish": execute SELL if position exists.
        if sentiment in ("Strongly Bullish", "Bullish"):
            if ticker not in self.positions or self.positions[ticker] == 0:
                available_capital = self.calculate_available_capital(current_data)
                quantity = self.calculate_position_size(available_capital, price)
                if quantity > 0:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'BUY',
                        'quantity': quantity
                    })
                    logger.info("Generated BUY signal for %s: quantity %d", ticker, quantity)
        elif sentiment in ("Bearish", "Strongly Bearish"):
            if ticker in self.positions and self.positions[ticker] > 0:
                signals.append({
                    'ticker': ticker,
                    'order_type': 'SELL',
                    'quantity': self.positions[ticker]
                })
                logger.info("Generated SELL signal for %s: quantity %d", ticker, self.positions[ticker])
        else:
            logger.info("Neutral sentiment; no action taken for ticker %s", ticker)

        return signals