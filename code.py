import numpy as np
import math
import pandas as pd

# It is point to be noted that this is for educational purposes mainly to understand dense layers, back propagatio, gradient descent and loss functions etc.
# I have not included the batches and normalization concept in the code which are really responsible for neywork to learn.

# Dense layer weights and biases intialzation, foward and backward passes
class Layers:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output  # returns output for activation function of a layer

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs  # loss that has to be propagated to the previous layer


# Activation Functions and their derivative that has to be propagated backwards

class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs>0, dvalues, 0)
        return self.dinputs

class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1.0/(1+np.exp(-inputs))
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.outputs * (1-self.outputs)
        return self.dinputs

class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))    
        return self.outputs

    def backward(self, devalues):
        self.dinputs = devalues * (1 - (self.outputs)**2)
        return self.dinputs

class SoftMax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_value = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_value/np.sum(exp_value, axis=1, keepdims=True)   
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (out, dval) in enumerate(zip(self.output, dvalues)):
            out = out.reshape(-1, 1)
            jacobian = np.diagflat(out) - np.dot(out, out.T)
            self.dinputs[i] = np.dot(jacobian, dval)
        return self.dinputs    

ACTIVATIONS = {'relu': Relu, 'sigmoid': Sigmoid, 'tanh': Tanh, 'softmax': SoftMax}

# Loss Functions and their derivative to propagate to previous layer  

class MSE:
    def forward(self, y_true, y_pred):
        self.outputs = np.mean((y_pred - y_true)**2)
        return self.outputs   

    def backward(self, y_true, y_pred):
        samples = len(y_pred)
        outputs = y_pred.shape[1]
        self.dinput = 2 * (y_pred - y_true)/ samples/ outputs
        return self.dinput

class CategoricalCrossEntropy:
    """y_true can be integer class labels (shape (n,)) or one-hot (shape (n, classes))."""

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_c = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if y_true.ndim == 1:
            confidences = y_pred_c[range(samples), y_true]
        else:
            confidences = np.sum(y_pred_c * y_true, axis=1)
        self.output = np.mean(-np.log(confidences))
        return self.output

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        labels = y_pred.shape[1]
        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]
        y_pred_c = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = (-y_true / y_pred_c) / samples
        return self.dinputs

LOSSES = {
    'mse': MSE,
    'categorical_crossentropy': CategoricalCrossEntropy,
}  # I only used two loss functions because this if foor educational purposses but I practise we have many options



class Model:
    def __init__(self, train_X, train_Y):  
        self.train_X = train_X
        self.train_Y = train_Y
        self.n_inputs = train_X.shape[1]
        self.layer = []
        self.activation = []
        self.loss = None

    # adds layers according to the developer's needs to self.layer list
    def add(self, n_neurons, activation=Relu):
        input_size = self.layer[-1].weights.shape[1] if self.layer else self.n_inputs
        if activation not in ACTIVATIONS:
            print("Please choose activations from", ACTIVATIONS)
        self.layer.append(Layers(input_size, n_neurons))
        self.activation.append(ACTIVATIONS[activation]())
        return self    

    # just declares which loss function to use
    def compile(self, loss):
        if loss not in LOSSES:
            print("Type of loss not found") 
        self.loss = LOSSES[loss]()
        self.loss_name = loss

    # it is a full forward pass trough the network
    def forward(self, X):
        out = X
        for layer, act in zip(self.layer, self.activation):
            out = layer.forward(out)
            out = act.forward(out)
        return out

    # calculates the gradient of wach weigth and bias
    def backward(self, y_pred, y_true):
        dvalue = self.loss.backward(y_pred, y_true)
        for layer, act in zip(reversed(self.layer), reversed(self.activation)):
            dvalue = act.backward(dvalue)
            dvalue = layer.backward(dvalue)    

    # update the weights and biases according to the learning rate
    def update(self, lr):
        for layer in self.layer:
            layer.weights = layer.weights - lr*layer.dweights    
            layer.biases = layer.biases - lr*layer.dbiases

    # it is the calling sequance of classes and methods
    def train(self, x, y, epochs=1000, lr=0.001, print_every=1):
        self.history = {'loss': [], 'accuracy': []} # stores the loss and accuracy throughout the training
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss_val = self.loss.forward(y_pred, y)
            acc = self._accuracy(y_pred, y) if self.loss_name != 'mse' else None

            self.backward(y_pred, y)
            self.update(lr)

            self.history['loss'].append(loss_val)
            self.history['accuracy'].append(acc)

            if print_every and epoch % print_every == 0:
                msg = f"epoch {epoch:5d}  loss {loss_val:.4f}"
                if acc is not None:
                    msg += f"  acc {acc:.2f}"
                print(msg)
        return y_pred    

    @staticmethod
    def _accuracy(y_pred, y_true):
        preds = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        if y_true.ndim == 1:
            truth = y_true # integer class labels
        elif y_true.shape[1] > 1:
            truth = np.argmax(y_true, axis=1) # one-hot multi-class labels
        else:
            truth = (y_true > 0.5).astype(int).flatten() # binary labels shaped (n,1)
        return np.mean(preds == truth) # returns accuracy


if __name__ == "__main__":
    np.random.seed(0)
mf = pd.read_csv(r"C:\Users\yadav\Downloads\archive (7)\synthetic_ecommerce_order_risk_dataset.csv") # Replace with the path of dataset I provided accordingly.
print(mf.shape)# Shape of data
# Risk label mapping 
mf['risk_label'] = mf['risk_label'].map({
    'Normal': 0,
    'Fraud Risk': 1,
    'Return Risk': 2
})

print(mf.risk_label.value_counts(normalize=True)) # Printing the number of each risk mapping.

# Categorical variables
cat_var = ['order_id',        'order_date', 'country', 'device_type',        'traffic_source', 'payment_method',        'product_category']

# Numerical variables
num_var = ['customer_age_days',        'previous_orders', 'avg_order_value_eur', 'order_value_eur',        'quantity',        'discount_rate', 'shipping_distance_km', 'delivery_days_estimated',        'late_delivery_risk', 'address_mismatch', 'high_risk_ip', 'customer_support_contacts', 'review_score',        'is_returned', 'is_fraud']

# One hoe encoding of categorical variables
mf = pd.get_dummies(mf, columns=cat_var)

# Test and train split
test_mf = mf.sample(frac=0.3, random_state=42)
train_mf = mf.drop(test_mf.index)

# Input variables
train_X = train_mf.drop('risk_label', axis=1)
test_X = test_mf.drop('risk_label', axis=1)

# True value of the data
train_Y = train_mf['risk_label']
test_Y = test_mf['risk_label']

# Normalizing data type
train_X = train_X.to_numpy().astype('float32')
test_X = test_X.to_numpy().astype('float32')

train_Y = train_Y.to_numpy().astype('int32')
test_Y = test_Y.to_numpy().astype('int32')

# user picks the architecture here:
model = Model(train_X, train_Y)
model.add(8, 'sigmoid')     # hidden layer, 8 neurons, ReLU
model.add(3, 'softmax')  # output layer, 3 classes, softmax
model.compile(loss='categorical_crossentropy')

model.train(train_X, train_Y, epochs=50, lr=0.001, print_every=1)  