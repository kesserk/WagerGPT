 WagerGPT

WagerGPT is a machine learning-powered web application designed for predictive analysis and sentiment-based insights, particularly suited for financial forecasting and betting-related predictions. This project integrates machine learning models, web scraping capabilities, sentiment analysis, and a background task queue to deliver real-time insights and predictions.

 Features

- Prediction Engine: Uses a Random Forest Classifier and LSTM model for predictive analytics.
- Sentiment Analysis: Analyzes sentiment from text data using NLTK's Sentiment Intensity Analyzer.
- Web Scraping: Extracts data using BeautifulSoup for real-time data analysis.
- Task Queue: Utilizes Celery and Redis to manage background tasks, ensuring scalable and efficient processing.
- Flask API: Provides an API endpoint for prediction and other functionalities.

 Installation

 Prerequisites

- Python 3.8+
- Redis server
- Docker (optional, for containerized setup)

 Step-by-Step Guide

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/WagerGPT.git
    cd WagerGPT
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    Create a `.env` file in the root directory and add the required environment variables:
    ```env
    CELERY_BROKER_URL=redis://localhost:6379/0
    FLASK_APP=WagerGPT.py
    ```

4. Run Redis (if not using Docker):
    ```bash
    redis-server
    ```

5. Start the Flask Application:
    ```bash
    flask run
    ```

6. Run Celery Worker:
    ```bash
    celery -A WagerGPT.celery worker --loglevel=info
    ```

 Usage

After setting up the server, you can access the application at `http://localhost:5000`. The application provides several endpoints:

- /predict - Generates predictions based on input data.
- /sentiment - Returns sentiment analysis for provided text.
- /scrape - Performs web scraping based on provided parameters.

Refer to the API documentation (if provided) for details on request formats and expected responses.

 Technologies Used

- Flask: Web framework for building APIs.
- Celery & Redis: For managing background tasks.
- scikit-learn: Machine learning library for training and evaluating models.
- TensorFlow: Deep learning framework, specifically for LSTM-based models.
- NLTK: Library for sentiment analysis.
- BeautifulSoup: Web scraping library for data extraction.

 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

