    import os
    import requests
    import pandas as pd
    import numpy as np
    from flask import Flask, request, jsonify
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from nltk.sentiment import SentimentIntensityAnalyzer
    from bs4 import BeautifulSoup
    import joblib
    import logging
    import time
    from datetime import datetime, timedelta
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from dotenv import load_dotenv
    from celery import Celery
    import redis

    # Load environment variables
    load_dotenv()

    # Initialize Flask app
    app = Flask(__name__)

    # Initialize Celery
    celery = Celery(app.name, broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
    celery.conf.update(app.config)

    # Initialize Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # API Keys
    FOOTBALL_DATA_API_KEY = '94b2f4c13d024e0586c69618ddf224fc'
    OPENWEATHER_API_KEY = '39b7c165ce8d223945af9c442e78d5b7'
    ODDS_API_KEY = '35170d264caf410d466ef753d4430c34'

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Helper Functions

    def fetch_sports_data():
        """Fetch football matches data from the Football API."""
        api_url = "https://api.football-data.org/v2/matches"
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        try:
            logging.info("Fetching sports data from Football API")
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch sports data: {e}")
            raise

    def fetch_odds_data():
        """Fetch odds data from OddsAPI."""
        api_url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us,uk',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }
        try:
            logging.info("Fetching odds data from OddsAPI")
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch odds data: {e}")
            raise

    def fetch_weather_data(city, date):
        """Fetch weather data from OpenWeather API."""
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}"
        try:
            logging.info(f"Fetching weather data for {city} on {date}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            for forecast in weather_data['list']:
                if date in forecast['dt_txt']:
                    return {
                        'temp': forecast['main']['temp'],
                        'humidity': forecast['main']['humidity'],
                        'weather_condition': forecast['weather'][0]['description']
                    }
        except requests.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            raise

    def clean_match_data(data):
        """Clean and format football match data."""
        match_data = []
        for match in data['matches']:
            match_data.append({
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_score': match['score']['fullTime']['homeTeam'],
                'away_score': match['score']['fullTime']['awayTeam'],
                'date': match['utcDate']
            })
        return pd.DataFrame(match_data)

    def clean_odds_data(data):
        """Clean and format odds data."""
        odds_data = []
        for game in data:
            odds_data.append({
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_odds': game['bookmakers'][0]['markets'][0]['outcomes'][0]['price'],
                'away_odds': game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
            })
        return pd.DataFrame(odds_data)

    def fetch_team_stats(team_name):
        """Fetch team stats, injuries, and other relevant data."""
        url = f"https://api.football-data.org/v2/teams/{team_name}"
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        try:
            logging.info(f"Fetching team stats for {team_name}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            team_stats = response.json()
            injuries = team_stats['squad']  # Example: list of players and injuries
            return injuries
        except requests.RequestException as e:
            logging.error(f"Failed to fetch team stats: {e}")
            raise

    def feature_engineering(data, odds_data):
        """Perform feature engineering on match data, including odds."""
        try:
            logging.info("Starting feature engineering")
            # Merge match data with odds data
            data = pd.merge(data, odds_data, on=['home_team', 'away_team'], how='left')

            # Example features
            data['home_advantage'] = (data['home_score'] - data['away_score']).apply(lambda x: 1 if x > 0 else 0)
            
            # Recent form over the last 5 games (rolling window)
            data['recent_form_home'] = data.groupby('home_team')['home_advantage'].rolling(5).sum().reset_index(0, drop=True)
            data['recent_form_away'] = data.groupby('away_team')['home_advantage'].rolling(5).sum().reset_index(0, drop=True)
            
            # Head-to-head feature
            data['head_to_head'] = data.groupby(['home_team', 'away_team'])['home_advantage'].cumsum().reset_index(0, drop=True)
            
            # Fetch weather data for each match (for home and away)
            weather_data = data.apply(lambda row: fetch_weather_data(row['home_team'], row['date']), axis=1)
            data['temp'] = weather_data.apply(lambda x: x['temp'])
            data['humidity'] = weather_data.apply(lambda x: x['humidity'])
            data['weather_condition'] = weather_data.apply(lambda x: x['weather_condition'])
            
            # Fetch team sentiment and player injuries
            data['sentiment_home'] = data['home_team'].apply(lambda x: fetch_sentiment_from_web(x))
            data['sentiment_away'] = data['away_team'].apply(lambda x: fetch_sentiment_from_web(x))
            
            # Injuries (a simple example of including injuries count for each team)
            data['home_injuries'] = data['home_team'].apply(lambda x: len(fetch_team_stats(x)))
            data['away_injuries'] = data['away_team'].apply(lambda x: len(fetch_team_stats(x)))
            
            # Add odds-related features
            data['odds_difference'] = data['home_odds'] - data['away_odds']
            
            logging.info("Feature engineering completed")
            return data
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}")
            raise

    def fetch_sentiment_from_web(team_name):
        """Scrape web data and analyze sentiment using BeautifulSoup and VADER."""
        cached_sentiment = redis_client.get(f"sentiment_{team_name}")
        if cached_sentiment:
            return float(cached_sentiment)

        url = f"https://www.google.com/search?q={team_name}+news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            logging.info(f"Fetching sentiment for {team_name}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.text for p in soup.find_all('p')])
            sentiment = sia.polarity_scores(text)['compound']
            
            # Cache the sentiment for 24 hours
            redis_client.setex(f"sentiment_{team_name}", 86400, str(sentiment))
            
            return sentiment
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch sentiment for {team_name}: {e}")
            return 0  # Default to neutral if there's an issue

    def prepare_data_for_model(data):
        """Prepare data for model training."""
        X = data[['recent_form_home', 'recent_form_away', 'head_to_head', 'temp', 'humidity', 'weather_condition', 
                'sentiment_home', 'sentiment_away', 'home_injuries', 'away_injuries', 'odds_difference']]
        y = data['home_advantage']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(X_train, y_train):
        """Train the RandomForest model."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def save_model(model, filename='model.pkl'):
        """Save the trained model to a file with versioning."""
        version = int(time.time())
        filename_with_version = f"model_v{version}.pkl"
        with open(filename_with_version, 'wb') as f:
            joblib.dump(model, f)
        logging.info(f"Model saved as {filename_with_version}")

    def load_model(filename='model.pkl'):
        """Load a trained model from a file."""
        with open(filename, 'rb') as f:
            return joblib.load(f)

    def prepare_lstm_data(df, time_steps):
        """Prepare the data for LSTM by structuring it into a time-series format."""
        features = ['recent_form_home', 'recent_form_away', 'player_rating_home', 'player_rating_away', 'odds_difference']
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i+time_steps])
            y.append(df['outcome'].iloc[i+time_steps])
        
        return np.array(X), np.array(y)

    def build_lstm_model(input_shape):
        """Build and compile an LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Celery Tasks

    @celery.task
    def update_sentiment_data():
        """Background task to update sentiment data for all teams."""
        teams = set(pd.concat([data['home_team'], data['away_team']]))
        for team in teams:
            fetch_sentiment_from_web(team)
        logging.info("Sentiment data updated for all teams")

    # Flask Routes

    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict the outcome of a football match."""
        try:
            input_data = request.json
            model = load_model('sports_prediction_lstm.h5')
            team_data = pd.DataFrame([input_data])

            odds_data = fetch_odds_data()
            odds_df = clean_odds_data(odds_data)
            processed_data = feature_engineering(team_data, odds_df)

            # Prepare data for LSTM
            X, _ = prepare_lstm_data(processed_data, time_steps=5)

            prediction = model.predict(X)
            return jsonify({'prediction': float(prediction[0][0])})

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500

    @app.route('/train', methods=['POST'])
    def train():
        """Train the model with new data."""
        try:
            input_data = request.json
            new_data = pd.DataFrame(input_data)

            odds_data = fetch_odds_data()
            odds_df = clean_odds_data(odds_data)
            engineered_data = feature_engineering(new_data, odds_df)

            X, y = prepare_lstm_data(engineered_data, time_steps=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            model.save('sports_prediction_lstm.h5')

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Mean CV score: {np.mean(cv_scores)}")

            return jsonify({'message': 'Model trained successfully!', 'mean_cv_score': float(np.mean(cv_scores))})

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return jsonify({'error': 'Training failed'}), 500

    if __name__ == "__main__":
        try:
            # Initial data preparation and model training
            raw_data = fetch_sports_data()
            cleaned_data = clean_match_data(raw_data)
            
            odds_data = fetch_odds_data()
            cleaned_odds_data = clean_odds_data(odds_data)
            
            engineered_data = feature_engineering(cleaned_data, cleaned_odds_data)
            X, y = prepare_lstm_data(engineered_data, time_steps=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
            model.save('sports_prediction_lstm.h5')

            # Schedule sentiment update task
            update_sentiment_data.apply_

            