import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# The column names are actually a row let's extract it and change the column names and put it into the the actual data
train_df_newRow = pd.DataFrame([train_df.columns.tolist()], columns = train_df.columns)
test_df_newRow = pd.DataFrame([test_df.columns.tolist()], columns = test_df.columns)
train_df = pd.concat([train_df_newRow, train_df], ignore_index = True)
test_df = pd.concat([test_df_newRow, test_df], ignore_index = True)

# Renaming the data columns
train_df.columns = ['Polarity', 'Title', 'Text']
test_df.columns = ['Polarity', 'Title', 'Text']

# Clean Polarity / Labels
train_df['Polarity'] = train_df['Polarity'].astype(str).str.strip().astype(int)
test_df['Polarity'] = test_df['Polarity'].astype(str).str.strip().astype(int)

# Removing Missing / Duplicate Values
train_df.dropna(subset=['Text', 'Title'], inplace=True)
train_df.drop_duplicates(subset=['Text'], inplace=True)

test_df.dropna(subset=['Text', 'Title'], inplace=True)
test_df.drop_duplicates(subset=['Text'], inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\<.*?\>', '', text)  # remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

train_df['Clean_Text'] = train_df['Text'].apply(clean_text)
test_df['Clean_Text'] = test_df['Text'].apply(clean_text)

# Combine Title and Text
train_df['Input_Text'] = train_df['Title'].fillna('') + ' ' + train_df['Clean_Text']
test_df['Input_Text'] = test_df['Title'].fillna('') + ' ' + test_df['Clean_Text']

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # can adjust later
X_train_vec = vectorizer.fit_transform(train_df['Input_Text'])
X_test_vec = vectorizer.transform(test_df['Input_Text'])

y_train = train_df['Polarity']
y_test = test_df['Polarity']

# Save preprocessed data
import joblib
joblib.dump(X_train_vec, 'data/X_train_vec.pkl')
joblib.dump(X_test_vec, 'data/X_test_vec.pkl')
joblib.dump(y_train, 'data/y_train.pkl')
joblib.dump(y_test, 'data/y_test.pkl')
joblib.dump(vectorizer, 'data/tfidf_vectorizer.pkl')
