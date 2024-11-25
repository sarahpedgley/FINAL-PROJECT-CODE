from abc import ABC, abstractmethod
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
    
class Model(ABC):
    @abstractmethod
    def fit(self, X: List[str], y: List[str]) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def score(self, X: List[str], y: List[str]) -> float:
        pass

class NaiveBayesModel(Model):
    def __init__(self):
        self.model = MultinomialNB()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    def predict(self, X) -> List[str]:
        return self.model.predict(X)
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)

class LogisticRegressionModel(Model):
    def __init__(self):
        self.model = LogisticRegression()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    def predict(self, X) -> List[str]:
        return self.model.predict(X)
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)
        
class SVMModel(Model):
    def __init__(self):
        self.model = SVC()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    def predict(self, X) -> List[str]:
        return self.model.predict(X)
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)
    
class EnsembleModel(Model):
    def __init__(self):
        self.naive_bayes = MultinomialNB()
        self.logistic_regression = LogisticRegression()
        self.svm = SVC(probability=True)  # probability estimates for voting

    def fit(self, X, y) -> None:
        # fit all models on the same data
        self.naive_bayes.fit(X, y)
        self.logistic_regression.fit(X, y)
        self.svm.fit(X, y)

    def predict(self, X) -> List[str]:
        # get predictions from each model
        nb_preds = self.naive_bayes.predict(X)
        lr_preds = self.logistic_regression.predict(X)
        svm_preds = self.svm.predict(X)
        
        # majority vote across models
        final_predictions = []
        for nb, lr, svm in zip(nb_preds, lr_preds, svm_preds):
            votes = [nb, lr, svm]
            final_prediction = max(set(votes), key=votes.count)
            final_predictions.append(final_prediction)
        
        return final_predictions

    def score(self, X, y) -> float:
        predictions = self.predict(X)
        return accuracy_score(y, predictions)