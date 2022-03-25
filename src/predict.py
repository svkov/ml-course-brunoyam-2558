import sys
import pickle
from src.train import preprocess_sentence


def load_model(path):
    with open(path, 'rb') as file:
        content = file.read()
    return pickle.loads(content)

def predict(model, inp_string):
    model_input = preprocess_sentence(inp_string)
    return model.predict([model_input])

if __name__ == '__main__':
    # Используем так: 
    # python src/predict.py random_forest.pickle "random string to predict"
    path_to_model, input_string = sys.argv[1], sys.argv[2]
    model = load_model(path_to_model)
    answer = predict(model, input_string)[0]
    print(f'"{input_string}" - {answer}')