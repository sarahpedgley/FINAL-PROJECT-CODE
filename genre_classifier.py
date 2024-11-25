from models import EnsembleModel
from vectorizer import Vectorizer
from models import Model
from sklearn.metrics import accuracy_score
from typing import List

class GenreClassifier:
    def __init__(self, model: Model, vectorizer: Vectorizer, genre_labels: List[str]):
        self.model = model
        self.vectorizer = vectorizer
        self.genre_labels = genre_labels
    
    def load_data(self, file_path: str) -> List[str]:
        # load and return data from a file
        file = open(file_path, "r") ####use numpy to read from local file??????
        pass
    
    def preprocess(self, text: str) -> str:
        # preprocess the text (tokenisation etc)
        return text
    
    def train(self, X: List[str], y: List[str]) -> None:
        X_vectorized = self.vectorizer.fit_transform(X)
        self.model.fit(X_vectorized, y)
        ##
    
    def predict(self, text: str) -> str:
        text_vectorized = self.vectorizer.transform([self.preprocess(text)])
        prediction = self.model.predict(text_vectorized)
        ##
        return prediction[0]
    
    def evaluate(self, X: List[str], y: List[str]) -> float:
        X_vectorized = self.vectorizer.transform(X)
        predictions = self.model.predict(X_vectorized)
        return accuracy_score(y, predictions)
    #on second thoughts is this method needed? it's not currently part of the plan