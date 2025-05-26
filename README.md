# 🍃 Climate Sentiment Text Classification
This is the project repository of **Machine Unlearning** in partial fulfillment of their project for CS 180 during the 2nd Semester of A.Y. 2024-2025. **Machine Unlearning** is composed of [Riana Bejarin](https://github.com/rianadb), [Nicholas Reyes](https://github.com/Njlr41), and [Jopeth Seda](https://github.com/pingu204).

The goal of this project is to perform sentiment analysis on an expert-annotated dataset containing climate-related paragraphs in corporate disclosures in order to mitigate the negative effects of climate change.

## Data Source 🌐
To be filled out

## Methodology 📚
The team employed both traditional machine learning and deep learning-based approaches in solving the problem.

### Data Preprocessing
To handle potential data inconsistencies, the dataset was first pre-processed, in accordance to the following pipeline:

1. Only retain alphanumeric characters and whitespaces
2. Convert characters into lowercase
3. Remove extra whitespaces
4. Handle negations by merging them into one token (e.g. `not good` becomes `not_good`)

### Sentiment Score

The VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool, designed to analyize the polarity of words and assign scores to them based on their emotional value, was used to improve the accuracy of the model.

### Traditional

For the traditional approach, the team made use of a weighted ensemble of a multinomial Naive Bayes model trained on `tf-idf`-tokenized dataset and a logistic regression model of sentiment scores obtained on the same dataset.

The dataset was first preprocessed and the compound sentiment score for each row was computed.

Afterwards, the sentences are tokenized using a `tf-idf` tokenizer before fed to the predictor, which returns the predicted labels as well as the probability of each prediction.

Initially, the traditional only made use of a multinomial Naive Bayes model. However, grid search and the implementation of the weighted ensemble was explored to improve the accuracy of the model.

## Dependencies 🗃️
The project is deployed on [Streamlit](https://streamlit.io/). The dependencies needed for the web app to work is listed in `requirements.txt`.

Nonetheless, the project in entirety made use of the following libraries:
- `pandas`
- `scikit-learn`
- `vaderSentiment`
- `pickle`
...

## Code Structure 🖥️
```
cs180-nlp/
├─ web.py
├─ etc/
├─ share/
├─ models/
│  ├─ bert/
│  ├─ lr_model.sav
│  ├─ trad_model.sav
│  ├─ vectorizer.sav
├─ predictions/
│  ├─ bert_predictions.csv
│  ├─ trad_predictions.csv
├─ src/
│  ├─ bert_predictions.ipynb
│  ├─ bert_train.ipynb
│  ├─ trad_demo.ipynb
│  ├─ trad_predictions.ipynb
│  ├─ trad_train.ipynb
├─ .gitignore
├─ requirements.txt
├─ README.md
```
