from genre_classifier import GenreClassifier
from models import EnsembleModel, NaiveBayesModel, LogisticRegressionModel, SVMModel
from vectorizer import Vectorizer
import os

def main():
    
    print("Literary genre classifier.")
    
    # initialize components
    vectorizer = Vectorizer()
    model = EnsembleModel()  # can replace with LogisticRegressionModel or SVMModel or naivebayes while developing
    genre_labels = ["fantasy", "sci-fi", "horror", "thriller", "mystery", "romance"]
    
    classifier = GenreClassifier(model, vectorizer, genre_labels)
    
    # example training data (load from a file) 
    file1 = open("C:\Users\pedgl\OneDrive\Documents\Uni\Final Year Project\FINAL PROJECT CODE\training_data\carmilla.txt", 'r', encoding='utf-8')
    
    texts = [file1, "A detective solving a murder mystery"]
    #print("training data loaded")
    
    labels = ["horror", "mystery"]
    
    # train the classifier
    classifier.train(texts, labels)

    filename = input("Please enter the file name/location of the literary sample you would like to classify: ")
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            sample = file.read()  
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)
    
    prediction = classifier.predict(sample.lower())   
    print("Prediction:", prediction)
    
    # evaluate the classifier
    accuracy = classifier.evaluate(texts, labels)
    print("Accuracy:", accuracy)
    #(not sure if this is needed)

if __name__ == "__main__":
    main()