# Sentiment Analysis on Amazon Reviews

In this project, neural networks and Naive Bayes models are used to analyze sentiment in Amazon reviews. Sorting reviews into positive and negative categories according to their substance is the main objective.

## Project Overview

- Data: Amazon reviews dataset
- Models: Naive Bayes models (ComplementNB, MultinomialNB) and a neural network model (RNN)
- Metrics: Accuracy, Precision, Recall, F1-Score

## Files

- "Sentiment_analysis_Amazon_Reviews_Naive_Bayesian_Models.ipynb": Jupyter notebook containing the implementation and evaluation of Naive Bayes models. 
- "Sentiment Analysis on Amazon Reviews (Deep Neural Networks).ipynb": Jupyter notebook containing the implementation and evaluation of the Recurrent Neural Network (RNN) model stacked with Long short-term memory (LSTM) model.
- "Saved_models": Contains all the saved trained models.

## Dataset

Download the datasets:
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/

## Setup

1. Clone this repository:

	```bash
	git clone https://github.com/trippynix/Sentiment-Analysis-on-Amazon-Reviews.git
        ```

2. Navigate to the project directory:
	
	```bash
	cd Sentiment-Analysis-on-Amazon-Reviews
	```

3. Install required dependencies:
	
	```bash
	pip install numpy
	pip install pandas
	pip install tensorflow
	pip install scikit-learn
	pip install joblib
	```

## Steps involved in training Naive Bayesian models

1. Data Preparation:
	- Changed column names to 'Polarity', 'Title', and 'Text'.
	- Converted the 'Polarity' column values from integers to strings. (mapped 1 -> 'Negative' and 2 -> 'Positive')
	- Saved the preprocessed data

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

## Steps involved in training Neural Networks

1. Data Preprocessing:
	- Loaded the previously preprocessed data
	- Mapped string values in 'Polarity' column i.e. 'Positive' and 'Negative' to 1 and 0 respectively
	- Sliced the data to 10% of it's original size as my local system was not capable of processing such vast amount of data
	- Split the data into 'train_features', 'train_target' and 'test_features', 'test_target' and 'val_features', 'val_target' sets	

2. Feature Engineering:
	- Initialized 'int' type vectorizor and 'TF-IDF' type vectorizer.
	- Adapted the vectorizors to train_features. 
	- Created embedding

3. Model training:
	- Trained GRU stacked with LSTM layers model (Both belong to the RNN family)
	- Compiled the model and fit the model on 'train_features' and 'train_target'

### evaluationMetrics()

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

## Model Performance (Accuracy on test data):

1. ComplementNB : 87.69%
2. MultinomialNB : 87.69%
3. RNN (GRU and LSTM) : 93.17%

## Using the saved models

To use the saved RNN model for predictions, you can load it using the following code:

```python
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the model
loaded_model = load_model('RNN_model')

# Predict using the loaded model
RNN_model_intVect_pred = loaded_model.predict(test_features)

# Squeeze the predicted data to 0 and 1
RNN_model_intVect_pred = tf.squeeze(tf.round(RNN_model_intVect_pred))

# Evaluate the model's performance
RNN1_eval = evaluationMetrics(test_target, RNN_model_intVect_pred)
print(f"Result of the model on unseen data: {RNN1_eval}")
```

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

Ensure that the predictions are binary before passing them to the 'evaluationMetrics' function.

## Contributing

Feel free to fork the repository and submit a pull request. For any issues or suggestions, please open an issue on the GitHub repository.

## **LICENSE**

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.
