# Sentiment Analysis on Amazon Reviews

## Project Description

This project involves sentiment analysis on Amazon reviews to classify reviews as positive or negative. 

## Current Status

As of now, the data preprocessing has been almost completed. The dataset has been cleaned and prepared for further analysis. 
The current steps include applying a Naive Bayes model for sentiment classification. Future plans involve experimenting with deep learning algorithms to potentially improve performance and accuracy.I have developed two models, ComplementNB and MultinomialNB, both of which perform exactly the same with an accuracy of 87.69%.

## Dataset

Download the datasets:
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/

## Setup

1. Clone this repository:

	```bash
	git clone https://github.com/trippynix/Sentiment-Analysis-on-Amazon-Reviews.git

2. Navigate to the project directory:
	```bash
	cd Sentiment-Analysis-on-Amazon-Reviews

3. Install required dependencies:
	```bash
	pip install scikit-learn
	pip install joblib

## Steps involved

1. Data Preparation:
	- Changed column names to 'Polarity', 'Title', and 'Text'.
	- Converted the 'Polarity' column values from integers to strings. (mapped 1 -> 'Negative' and 2 -> 'Positive')

2. Feature Engineering:
	- Utilized ColumnTransformer to apply TF-IDF on the train data features, which included 'Title' and 'Text' columns.

3. Model Training:
	- Trained two models: ComplementNB and MultinomialNB.

4. Model Evaluation:
	- Created a helper function called 'evaluationMetrics' to evaluate the models on test data.
	- The evaluationMetrics function provides output in the form of a dictionary with the following metrics:
		- Accuracy
		- Precision
		- Recall
		- F1-Score

### 'evaluationMetrics()'

	from sklearn.metrics import accuracy_score, precision_recall_fscore_support

	def evaluationMetrics(y_true, y_pred):
	    model_accuracy = accuracy_score(y_true, y_pred) * 100
	    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred)
	    
	    model_result = {
	       	"Accuracy": model_accuracy,
	       	"Precision": model_precision,
	       	"Recall": model_recall,
	       	"F1-Score": model_f1
		}
	    
	    return model_result

## Model Performance (Accuracy):

1. ComplementNB : 87.69%
2. MultinomialNB : 87.69%

## Contributing

Feel free to fork the repository and submit a pull request. Contributions and feedback are welcome!

