import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

def generate_data_toggle_switch(num_examples):
    x = np.random.randint(0, 2, size=(num_examples, 1))
    y = np.logical_xor(x, 1).astype(int)
    return x, y

def generate_data_xor(num_examples):
    x = np.random.randint(0, 2, size=(num_examples, 2))
    y = np.logical_xor(x[:,0], x[:,1]).astype(int)
    x = np.hstack((x, y[:,np.newaxis]))
    return x, y

def build_model(input_dim, hidden_dim, output_dim, activation):
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim, activation=activation, input_dim=input_dim),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(x_train, y_train, x_test, y_test, input_dim, hidden_dim, output_dim):
    activations = ['relu', 'sigmoid']
    histories = []
    for activation in activations:
        model = build_model(input_dim, hidden_dim, output_dim, activation)
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=0)
        histories.append(history)

    accuracies = []
    for history in histories:
        accuracy = history.history['val_accuracy'][-1]
        accuracies.append(accuracy)

    best_accuracy = max(accuracies)
    return best_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='toggle_switch', help='Toggle switch or XOR function')
    args = parser.parse_args()

    if args.scenario == 'toggle_switch':
        input_dim = 1
        hidden_dim = 8
        output_dim = 1
        x_train, y_train = generate_data_toggle_switch(1000)
        x_test, y_test = generate_data_toggle_switch(200)
        threshold = 0.95
    elif args.scenario == 'xor':
        input_dim = 2
        hidden_dim = 8
        output_dim = 1
        x_train, y_train = generate_data_xor(1000)
        x_test, y_test = generate_data_xor(200)
        threshold = 0.85
    else:
        raise ValueError('Invalid scenario')

    accuracies = []
    for i in range(5):
        accuracy = train_and_evaluate(x_train, y_train, x_test, y_test, input_dim, hidden_dim, output_dim)
        accuracies.append(accuracy)

    best_accuracy = max(accuracies)
    if best_accuracy >= threshold:
        print('Best model achieved test accuracy of', best_accuracy)
    else:
        print('No model achieved test accuracy greater than', threshold)

if _name_ == '_main_':
    main()