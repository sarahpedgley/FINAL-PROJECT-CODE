from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def fit_transform(self, text: List[str]):
        return self.vectorizer.fit_transform(text)
    
    def transform(self, text: List[str]):
        return self.vectorizer.transform(text) #i am now getting an error here 'NoneType' object has no attribute 'lower'