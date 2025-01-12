from genre_classifier import GenreClassifier
from models import EnsembleModel, NaiveBayesModel, LogisticRegressionModel, SVMModel, DictionaryAlgorithm
from models import genre_keywords
from vectorizer import Vectorizer
import os

def main():
    
    print("Literary genre classifier.")

    vectorizer = Vectorizer()
    genre_labels = ["Fantasy", "Sci-fi", "Horror", "Thriller", "Mystery", "Romance"]
    #model = EnsembleModel()
    
    from models import genre_keywords 
    dictionary_algorithm = DictionaryAlgorithm(genre_keywords=genre_keywords)
    
    models = {
        "Naive Bayes": NaiveBayesModel(),
        "Logistic Regression": LogisticRegressionModel(),
        "SVM": SVMModel(),
        "Ensemble": EnsembleModel(),
        "Dictionary": dictionary_algorithm
    }
    
    training_dir = "C:\\Users\\pedgl\\OneDrive\\Documents\\Uni\\Final Year Project\\FINAL PROJECT CODE\\training_data"

    # Load training data
    texts, labels = [], []
    try:
        print(f"Training model: {model_name}")
        if isinstance(model, DictionaryAlgorithm):
            # Skip vectorization for DictionaryAlgorithm
            model.fit(texts, labels)
        else:
            classifier = GenreClassifier(model, vectorizer, genre_labels)
            classifier.train(texts, labels)
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit(1)

    if not texts or not labels:
        print("Training data is empty. Exiting...")
        exit(1)
    
    # train the classifier
    texts = [text if text is not None else "" for text in texts]
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
    
    # loop through each model so we get results for each one individually 
    print("\nResults of each model:")
    for model_name, model in models.items():
        try:
            if isinstance(model, DictionaryAlgorithm):
                # skips vectorization
                prediction = model.predict([sample.lower()])[0]
            else:
                classifier = GenreClassifier(model, vectorizer, genre_labels)
                prediction = classifier.predict(sample.lower())

            print(f"\n{model_name} Prediction:")
            print(f"  {prediction}")
        except Exception as e:
            print(f"Error with model '{model_name}': {e}")

if __name__ == "__main__":
    main()