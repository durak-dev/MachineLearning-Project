import numpy as np
import pandas as pd
np.random.seed(42)

class Sigmoid:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((output_dim, 1))
        self.input = None
        self.arr_Z = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        output = self.sigmoid(x)
        return output * (1 - output)

    def forward(self, input):
        self.input = input
        self.arr_Z = np.dot(self.weights.transpose(), self.input) + self.bias
        return self.sigmoid(self.arr_Z)

    def backward(self, delta_from_next_layer, learning_rate):
        del_Z = delta_from_next_layer * self.sigmoid_derivative(self.arr_Z)
        dB = np.sum(del_Z, axis=1, keepdims=True)
        dW_for_update = np.dot(self.input, del_Z.transpose())
        self.weights -= learning_rate * dW_for_update
        self.bias -= learning_rate * dB
        error_to_previous_layer = np.dot(self.weights, del_Z)
        return error_to_previous_layer

class SoftMAX:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros((output_dim, 1))
        self.input = None
        self.arr_Z = None

    def softmax(self, x):
        arr = []
        for val in x:
            arr.append([np.exp(val) / np.sum(np.exp(x))])
        return np.array(arr).reshape(2, 1)

    def softmax_derivative(self, x):
        output = self.softmax(x)
        return output * (1 - output)

    def forward(self, input):
        self.input = input
        self.arr_Z = np.dot(self.weights.transpose(), self.input) + self.bias
        return self.softmax(self.arr_Z)

    def backward(self, delta_from_next_layer, learning_rate):
        del_Z = delta_from_next_layer * self.softmax_derivative(self.arr_Z)
        dB = np.sum(del_Z, axis=1, keepdims=True)
        dW_for_update = np.dot(self.input, del_Z.transpose())
        self.weights -= learning_rate * dW_for_update
        self.bias -= learning_rate * dB
        error_to_previous_layer = np.dot(self.weights, del_Z)
        return error_to_previous_layer

class Trinity:
    def __init__(self, size, epochs, learning_rate=0.1, momentum=None):
        self.momentum = momentum
        self.v_input_w = None
        self.v_input_b = None
        self.v_mid_w = None
        self.v_mid_b = None
        self.v_out_w = None
        self.v_out_b = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.del_in = None
        self.del_out = None
        # For binary inputs, we can simplify: directly map to hidden layer
        self.input_layer = Sigmoid(size, size // 2 + 1)
        self.mid_layer = Sigmoid(size // 2 + 1, size // 4 + 1)
        self.output_layer = SoftMAX(size // 4 + 1, 2)  # 2 classes: OK, KO

    def forward_pass(self, data):
        X = self.input_layer.forward(data)
        Y = self.mid_layer.forward(X)
        Z = self.output_layer.forward(Y)
        return Z

    def backward_pass(self, delta):
        error_to_mid_layer = self.output_layer.backward(delta, self.learning_rate)
        error_to_input_layer = self.mid_layer.backward(error_to_mid_layer, self.learning_rate)
        self.input_layer.backward(error_to_input_layer, self.learning_rate)

    def update_with_momentum(self, error):
        # 1. Run backward passes (your original layers unchanged)
        error_to_mid = self.output_layer.backward(error, self.learning_rate)
        error_to_input = self.mid_layer.backward(error_to_mid, self.learning_rate)
        self.input_layer.backward(error_to_input, self.learning_rate)  # Runs but gradients not applied yet


        # INPUT LAYER
        del_Z_input = error_to_input * self.input_layer.sigmoid_derivative(self.input_layer.arr_Z)
        dW_input = np.dot(self.input_layer.input, del_Z_input.transpose())
        dB_input = np.sum(del_Z_input, axis=1, keepdims=True)
        if self.v_input_w is None:
            self.v_input_w = np.zeros_like(self.input_layer.weights)
            self.v_input_b = np.zeros_like(self.input_layer.bias)
        self.v_input_w = self.momentum * self.v_input_w - self.learning_rate * dW_input
        self.v_input_b = self.momentum * self.v_input_b - self.learning_rate * dB_input
        self.input_layer.weights += self.v_input_w
        self.input_layer.bias += self.v_input_b

        # MID LAYER (repeat exact pattern)
        del_Z_mid = error_to_mid * self.mid_layer.sigmoid_derivative(self.mid_layer.arr_Z)
        dW_mid = np.dot(self.mid_layer.input, del_Z_mid.transpose())
        dB_mid = np.sum(del_Z_mid, axis=1, keepdims=True)
        if self.v_mid_w is None:
            self.v_mid_w = np.zeros_like(self.mid_layer.weights)
            self.v_mid_b = np.zeros_like(self.mid_layer.bias)
        self.v_mid_w = self.momentum * self.v_mid_w - self.learning_rate * dW_mid
        self.v_mid_b = self.momentum * self.v_mid_b - self.learning_rate * dB_mid
        self.mid_layer.weights += self.v_mid_w
        self.mid_layer.bias += self.v_mid_b

        # OUTPUT LAYER (repeat exact pattern - adjust for SoftMAX derivative)
        del_Z_out = error * self.output_layer.softmax_derivative(self.output_layer.arr_Z)  # Use your SoftMAX deriv
        dW_out = np.dot(self.output_layer.input, del_Z_out.transpose())
        dB_out = np.sum(del_Z_out, axis=1, keepdims=True)
        if self.v_out_w is None:
            self.v_out_w = np.zeros_like(self.output_layer.weights)
            self.v_out_b = np.zeros_like(self.output_layer.bias)
        self.v_out_w = self.momentum * self.v_out_w - self.learning_rate * dW_out
        self.v_out_b = self.momentum * self.v_out_b - self.learning_rate * dB_out
        self.output_layer.weights += self.v_out_w
        self.output_layer.bias += self.v_out_b

    def train(self, training_data, training_result):
        cnt = 0
        while cnt < self.epochs:
            epoch_loss = 0
            for i in range(len(training_data)):
                input_sample = training_data[i]
                target_sample = training_result[i].reshape(-1, 1)
                predicted_output = self.forward_pass(input_sample)
                error = predicted_output - target_sample
                epoch_loss += np.mean(np.abs(error))
                if self.momentum:
                    self.update_with_momentum(error)
                else:
                    self.backward_pass(error)
            if cnt % 100 == 0:
                avg_error = epoch_loss / len(training_data)
                print(f"Epoch: {cnt:4d} | Avg Error: {avg_error:.6f} | Learning Rate: {self.learning_rate:.6f}")
            cnt += 1
            self.learning_rate *= 0.995

        print(f"\nTraining completed. Final learning rate: {self.learning_rate:.6f}")

    def run(self, vector):
        """
        Classify a binary input vector as OK or KO

        Args:
            vector: binary input vector (containing only 0s and 1s)

        Returns:
            Classification result as string
        """
        results = self.forward_pass(vector)

        # Get the class with highest probability
        max_value = np.max(results)

        if results[0] == max_value:
            classification = "KO"
            confidence = results[0]
        else:
            classification = "OK"
            confidence = results[1]

        return results, classification

class StackingTrinity(Trinity):
    def __init__(self, models, epochs=500, learning_rate=0.1, momentum=None):
        super().__init__(len(models), epochs, learning_rate, momentum)
        self.models = models
        self.size = len(models)
        self.train_set_trinity = []

    def train(self, X, y):
        model_results = []

        for i, model in enumerate(self.models):
            model.fit(X, y)
            proba = model.predict_proba(X)[:, 1]
            model_results.append(proba)
            print(f"Model {i}: {proba}")
        model_results = np.array(model_results)
        training_trinity = [
        model_results[:, i:i+1]
        for i in range(X.shape[0])]

        y_trinity = pd.get_dummies(y, dtype=int).to_numpy()
        super().train(training_trinity, y_trinity)

    def predict(self, X):
        model_results = []

        for i, model in enumerate(self.models):
            model_results.append(model.predict_proba(X)[:, 1])

        model_results = np.array(model_results)
        testing_trinity = [model_results[:, i:i+1] for i in range(X.shape[0])]

        results_trinity = []
        for res in testing_trinity:
            _, result = super().run(res)
            results_trinity.append(result)

        return pd.Series(results_trinity)

    def score(self, X, y):
        predicted = self.predict(X)
        nr_records = len(y)
        correct = sum(predicted[i] == y.values[i] for i in range(nr_records))
        return correct / nr_records

class TrinityClassifier(Trinity):
    def __init__(self, n_features, epochs=500, learning_rate=0.1, momentum=None):
        super().__init__(n_features, epochs, learning_rate, momentum)
        self.n_features = n_features
        self.train_set_trinity = []

    def train(self, X, y):
        training_trinity = np.array([[np.array([val]) for val in row] for row in X.values])
        y_trinity = pd.get_dummies(y, dtype=int).to_numpy()
        super().train(training_trinity, y_trinity)

    def predict(self, X):
        testing_trinity = np.array([[np.array([val]) for val in row] for row in X.values])

        results_trinity = []
        for res in testing_trinity:
            _, result = super().run(res)
            results_trinity.append(result)

        return pd.Series(results_trinity, index = X.index)

    def score(self, X, y):
        predicted = self.predict(X)
        # Flatten to ensure 1D arrays and compare positionally
        y_flat = y.values.ravel()
        pred_flat = predicted.values.ravel()
        return np.mean(y_flat == pred_flat)
