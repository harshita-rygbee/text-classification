
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
import numpy as np

data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

categories = ['alt.atheism', 'sci.med'] # label 1 means it's sci-med

train_df = pd.read_csv(os.path.join(data_dir, "newsgroup_train.csv"), sep='\t')
test_df = pd.read_csv(os.path.join(data_dir, "newsgroup_test.csv"), sep='\t')

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(train_df["text"], train_df["label"])

docs_test = test_df["text"]
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == test_df["label"]))

