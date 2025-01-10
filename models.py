from abc import ABC, abstractmethod
from typing import List, Dict
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
    
class DictionaryAlgorithm(): #new addition - create own dictionary based algorithm
    #note - should the preprocessing for this be different? more of it?
    def __init__(self, genre_keywords: Dict[str, List[str]]):
        
        genre_keywords = {
    "fantasy": ["dragon", "magic", "wizard", "castle", "sorcery", "witch", "curse", "hex", "castle", "royal", "princess", "prince", "palace", "country"],
    "sci-fi": ["spaceship", "mars", "martian", "moon", "space", "alien", "robot", "future", "quantum", "equation", "formula", "lunar", "solar", "world", "earth"],
    "horror": ["ghost", "haunted", "vampire", "zombie", "fear", "blood", "decapitated", "head", "body", "kill", "terror", "afraid"],
    "thriller": ["murder", "spy", "conspiracy", "detective", "chase", "dark", "quiet", "suspicious", "shadow", "gloom", "night", "race"],
    "mystery": ["murder","clue", "investigation", "detective", "crime", "scene", "body", "mystery", "police", "evidence", "robbery", "business", "affair", "puzzle", "knife", "gun"],
    "romance": ["love", "marry", "romance", "passion", "kiss", "heart", "husband", "wife", "beauty", "inheritance", "partner", "estate"]
}
        self.genre_keywords = genre_keywords
    
    def count_keywords(self, text: str) -> Dict[str, int]:
        genre_scores = {}  
        for genre, keywords in self.genre_keywords.items():
            genre_scores[genre] = 0  
        for keyword in keywords:
            genre_scores[genre] += text.count(keyword)  
        #print(genre_scores)
        return genre_scores
    
    def fit(self, X: List[str], y: List[str]) -> None:
        #not needed for this algorithm but implementing for consistency
        pass
    
    def predict(self, X) -> List[str]:
        predictions = []
        for text in X:
            genre_scores = self.count_keywords(text)
            best_genre = max(genre_scores, key=genre_scores.get)
            predictions.append(best_genre)
        return predictions
    
    def score(self, X, y) -> float:
        predictions = self.predict(X)
        return accuracy_score(y, predictions)