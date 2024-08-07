# DriveFeelings: What Do Twitter Users Think About BMW, Renault, and Tesla? #

This project analyzes the text of tweets from Twitter users containing keywords related to three randomly selected major automotive brands:

1. BMW
2. Renault
3. Tesla

The goal of the project is to understand user opinions about these cars, determine which is the most or least appreciated, examine the topics referenced, and ultimately implement a sentiment classifier using the pre-trained RoBERTa model.

-------------------------

### Analysis to Be Performed ###

The project consists of 4 phases, each corresponding to a dedicated branch:

1. TwitterScraper: Extraction of 10,000 tweets for each automotive brand, totaling 30,000 tweets.
2. PreProcessing: Cleaning the tweets to prepare them for sentiment and topic analysis.
3. TopicModeling: Grouping the extracted tweets into 5 categories or topics.
4. RoBERTa SentimentAnalysis: Performing sentiment analysis using the RoBERTa model, evaluating each label (positive/neutral/negative) for each automotive brand, and comparing the accuracy between pre-trained and fine-tuned models to label the extracted data using the most effective model.

-------------------------
[![Top Programming Language](https://img.shields.io/github/languages/top/chrisputzu/DriveFeelings-RoBERTa-SentimentAnalysis)](https://github.com/chrisputzu/DriveFeelings-RoBERTa-SentimentAnalysis)
