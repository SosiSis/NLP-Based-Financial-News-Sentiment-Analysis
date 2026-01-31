# ADDIS ABABA UNIVERSITY

## COLLEGE OF TECHNOLOGY AND BUILT ENVIRONMENT  
## SCHOOL OF INFORMATION TECHNOLOGY AND ENGINEERING  

---

**Project Title:**  
**NLP-Based Financial News Sentiment Analysis for Predicting Stock Market Trends**

**Group Members:**

1. Roman Kebede — UGR/0448/14  
2. Sosina Sisay — UGR/0131/14  
3. Loti Yadeta — UGR/2782/14  

---

## 1. Abstract

This project investigates the correlation between financial news sentiment and stock market price movements. By implementing a pipeline that integrates Natural Language Processing (NLP) and Machine Learning, we developed a system to classify news headlines and predict short-day price directions. Using domain-specific models like FinBERT alongside traditional technical indicators, the study demonstrates that sentiment analysis provides a significant *informational edge* in forecasting market trends.

---

## 2. Project Overview

Financial markets are highly sensitive to public information. Traditional quantitative analysis often overlooks the *human element*—the sentiment and psychology reflected in news articles. This project aims to bridge this gap by converting unstructured text data into quantitative sentiment scores to predict whether a specific ticker's price will move **UP** or **DOWN** on the following trading day.

---

## 3. Objectives

- **Text Preprocessing:** Clean and tokenize financial headlines using NLTK to remove noise (URLs, tickers, and symbols).  
- **Sentiment Extraction:** Utilize both lexicon-based (VADER) and transformer-based (FinBERT) models to generate sentiment features.  
- **Feature Integration:** Combine text-derived sentiment with historical price data (OHLCV) and technical indicators.  
- **Trend Prediction:** Train and evaluate machine learning models (Random Forest and LSTM) to predict market direction.  
- **Visualization:** Illustrate the relationship between sentiment shifts and price volatility.  

---

## 4. Methodology and Implementation

### 4.1 Data Pipeline

The project utilizes a dataset of 81 high-quality financial records containing headlines and corresponding price metrics.

### 4.2 Text Preprocessing

We implemented a robust cleaning function using regular expressions and NLTK:

- **Regex Cleaning:** Removed URLs and specific financial patterns such as `$TICKER` and currency symbols.  
- **Tokenization and Lemmatization:** Used the `WordNetLemmatizer` to normalize words to their root forms.  
- **Sentiment-Aware Stopwords:** Customized the standard English stopword list to **retain** critical market-direction terms such as *“up”*, *“down”*, *“growth”*, and *“high”*.  

### 4.3 Feature Engineering

The final feature set includes 27 unique columns:

- **Sentiment Features:**  
  - `finbert_positive`  
  - `finbert_negative`  
  - `finbert_neutral`  
  - `vader_compound`  

- **Technical Indicators:**  
  - 5-day and 20-day Simple Moving Averages (SMA)  
  - Relative Strength Index (RSI)  
  - MACD  
  - Bollinger Bands (`BB_upper`, `BB_lower`)  

---

## 5. Results and Evaluation

Based on results from `Evaluation.ipynb`, a traditional Random Forest classifier was compared with a deep learning LSTM model.

### 5.1 Model Performance

| Metric | Random Forest | LSTM (Deep Learning) |
|------|---------------|---------------------|
| **Accuracy** | *0.471* | *0.800* |
| **F1-Score (DOWN)** | *0.000* | *0.000* |
| **F1-Score (UP)** | *0.640* | *0.889* |

### 5.2 Key Findings

- **FinBERT vs. VADER:** FinBERT demonstrated superior performance in identifying neutral financial news that VADER often misclassified as positive.  
- **Sentiment Impact:** Sudden spikes in `finbert_negative` scores showed strong correlation with price declines within a 24-hour window.  

---

## 6. Tools and Technologies

- **Programming Language:** Python 3.12  

- **Libraries and Frameworks:**  
  - `Pandas`, `NumPy` — Data manipulation  
  - `NLTK` — Text preprocessing  
  - `Scikit-learn` — Model training and evaluation  
  - `TensorFlow / Keras` — LSTM implementation  
  - `Matplotlib`, `Seaborn` — Visualization  

- **Development Environment:**  
  - Visual Studio Code  
  - Jupyter Notebooks  

---

## 7. Conclusion

The project successfully demonstrates that NLP-based sentiment analysis, when integrated with technical market indicators, improves the accuracy of short-term stock trend prediction. Although market volatility remains a challenge, FinBERT-derived sentiment features proved particularly effective for directional forecasting.

---

## 8. Future Recommendations

1. **Real-Time Deployment:** Implement the system using a **FastAPI** backend and **Streamlit** frontend for live news analysis.  
2. **Extended Sentiment Sources:** Integrate social media sentiment from platforms such as StockTwits or Reddit.  
3. **Cross-Ticker Expansion:** Train models on multiple sectors (Technology, Energy, Healthcare) to capture sector-specific sentiment effects.  

---

**Date:** January 2026  

**Department:** School of Information Technology and Engineering  

**Addis Ababa University**
