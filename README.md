# Bitcoin Trading Bot with Machine Learning

## Introduction

### Problem Statement:
In the rapidly evolving world of cryptocurrency, Bitcoin remains a dominant player, and its upcoming halving event this year is poised to have significant effects on its market dynamics. University students, keen to invest in emerging technologies but often with limited financial experience and resources, face heightened risks due to this volatility. To start their crypto trading journey, we are developing a Bitcoin trading bot that uses machine learning to predict the price movement of bitcoin in the short-run. This bot aims to help students make informed, confident trading decisions, minimizing their risks and maximizing potential gains, thus providing a safer entry into the world of cryptocurrency trading.

### Dataset:
For our dataset, we obtained our data from Kaggle, which includes multiple indicators relating to the price of Bitcoin. This is the latest Bitcoin dataset as it includes data in the year 2024 ranging all the way back to 2015, which is the perfect size for our method of Machine Learning. [Dataset Source](https://www.kaggle.com/datasets/aspillai/bitcoin-price-trends-with-indicators-8-years/data)

## Exploratory Data Analysis (EDA)

In the EDA phase, we first set the date as our index and plotted a candlestick graph to analyze the general trend of Bitcoin’s prices throughout the years. We observed significant price movements, particularly following the halving event in 2020. By understanding the nature of Bitcoin's long-term prices and the various factors affecting them, we aimed to predict the short-term price movements using specific market indicators.

We created three new variables to aid in predicting the growth rate of Bitcoin: short moving average, long moving average, and a signal. The signal column was derived from comparing the short moving average to the long moving average using a well-known investment strategy called the ‘golden cross’. We analyzed correlations between various indicators and the signal to identify significant predictors.

## Machine Learning Models

We implemented three machine learning models: Logistic Regression, Gradient Boosting Classifier, and Random Forest Classifier. These models were chosen due to their ability to predict binary variables effectively.

- **Logistic Regression**: Utilizes a mathematical formula to model the relationship between predictor variables and the probability of the binary outcome. [Learn more](https://www.ibm.com/topics/logistic-regression)
- **Gradient Boosting Classifier**: Captures complex nonlinear relationships between predictor variables and the binary outcome by sequentially fitting new models to the residuals of previous models. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- **Random Forest Classifier**: Handles large datasets and reduces the risk of overfitting through ensemble learning. [Tutorial](https://www.datacamp.com/tutorial/random-forests-classifier-python)

We split our dataset into training and test sets (80:20) and applied k-fold cross-validation to evaluate model performance. Based on precision scores, we selected the Random Forest Classifier for hyperparameter tuning, achieving an optimal accuracy score of 0.936 on the test set. [Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

## Conclusion

Through meticulous data cleaning, preparation, and fine-tuning of machine learning models, we achieved a consistent and high accuracy in predicting buy/sell signals based on the Golden Cross investment strategy. Feature importance analysis revealed that MACD significantly affects the model's predictions, emphasizing its importance for traders. However, we caution against relying solely on the model due to the potential impact of sentiment values on Bitcoin's movement.

## References:
- Kaggle Dataset: [Bitcoin Price Trends with Indicators - 8 Years](https://www.kaggle.com/datasets/aspillai/bitcoin-price-trends-with-indicators-8-years/data)
- Canva Design: [Link to Design](https://www.canva.com/design/DAGDT6k5w7s/hNjkh6q51VQXo9xNVmw11w/view?utm_content=DAGDT6k5w7s&utm_campaign=designshare&utm_medium=link&utm_source=editor)
- Golden Cross Strategy: [Investopedia](https://www.investopedia.com/terms/g/goldencross.asp)
- K-fold Cross Validation: [Machine Learning Mastery](https://machinelearningmastery.com/k-fold-cross-validation/)
- Bitcoin Market Cap: [Companies Market Cap](https://companiesmarketcap.com/assets-by-market-cap/)
- Logistic Regression: [IBM](https://www.ibm.com/topics/logistic-regression)
- Random Forest Classifier: [DataCamp](https://www.datacamp.com/tutorial/random-forests-classifier-python)
- Hyperparameter Tuning: [Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

