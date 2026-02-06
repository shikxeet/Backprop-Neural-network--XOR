
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)



# Network structure: 2 inputs → 2 hidden neurons → 1 output

# Weights from input to hidden layer
# Shape: (2 hidden neurons, 2 inputs)
weights_input_hidden = np.random.randn(2, 2) * 0.5

# Weights from hidden to output layer
# Shape: (1 output, 2 hidden neurons)
weights_hidden_output = np.random.randn(1, 2) * 0.5

# Biases for hidden layer
bias_hidden = np.zeros((2, 1))

# Bias for output layer
bias_output = np.zeros((1, 1))


print(f"Total parameters: {weights_input_hidden.size + weights_hidden_output.size + 3}")
print()



# XOR inputs (each column is one example)
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

# XOR outputs
Y = np.array([[0, 1, 1, 0]])

print("TRAINING DATA:")
print("="*70)
for i in range(4):
    print(f"Input: [{X[0,i]}, {X[1,i]}] → Target: {Y[0,i]}")
print()



learning_rate = 0.5  # How big steps to take when learning
epochs = 10000       # How many times to go through all examples

print("TRAINING SETTINGS:")
print("="*70)
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")


for epoch in range(epochs):
    
    # Hidden layer calculation
    # Step 1: Multiply inputs by weights and add bias
    hidden_input = np.dot(weights_input_hidden, X) + bias_hidden
    # Step 2: Apply activation function
    hidden_output = sigmoid(hidden_input)
    
    # Output layer calculation
    # Step 1: Multiply hidden outputs by weights and add bias
    final_input = np.dot(weights_hidden_output, hidden_output) + bias_output
    # Step 2: Apply activation function
    final_output = sigmoid(final_input)
    
    #compute error
    error = Y - final_output
    

    # BACKWARD PASS: Compute how to adjust weights
    
    # Output layer gradients
    # "How much did the output layer contribute to the error?"
    output_delta = error * sigmoid_derivative(final_output)
    
    # Hidden layer gradients
    # "How much did the hidden layer contribute to the error?"
    # We propagate the error backward through the weights!
    hidden_error = np.dot(weights_hidden_output.T, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    
    # UPDATE WEIGHTS: Adjust weights to reduce error

    
    # Update weights from hidden to output
    weights_hidden_output += learning_rate * np.dot(output_delta, hidden_output.T)
    bias_output += learning_rate * np.sum(output_delta, axis=1, keepdims=True)
    
    # Update weights from input to hidden
    weights_input_hidden += learning_rate * np.dot(hidden_delta, X.T)
    bias_hidden += learning_rate * np.sum(hidden_delta, axis=1, keepdims=True)
    
    
    # PRINT PROGRESS
    
    
    if epoch % 1000 == 0:
        # Mean squared error
        mse = np.mean(error ** 2)
        print(f"Epoch {epoch:5d} | Error: {mse:.6f}")


# STEP 6: TEST THE NETWORK

print()
print("TESTING RESULTS:")
print("="*70)

# Make predictions
hidden_input = np.dot(weights_input_hidden, X) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(weights_hidden_output, hidden_output) + bias_output
predictions = sigmoid(final_input)

# Show results
for i in range(4):
    input_1 = X[0, i]
    input_2 = X[1, i]
    target = Y[0, i]
    predicted = predictions[0, i]
    predicted_class = 1 if predicted > 0.5 else 0
    
    
    print(f"{input_1} XOR {input_2} = {target} | Predicted: {predicted:.4f} → {predicted_class}")

print()
print("="*70)
print("TRAINING COMPLETE!")
print("="*70)


# ============================================================================
# BONUS: VISUALIZE WHAT HAPPENED
# ============================================================================

print()
print("WHAT DID THE NETWORK LEARN?")
print("="*70)
print()
print("Hidden Layer Weights (from inputs):")
print(weights_input_hidden)
print()
print("These weights determine what patterns each hidden neuron detects!")
print()
print("Output Layer Weights (from hidden neurons):")
print(weights_hidden_output)
print()
print("These combine the hidden neurons to make the final decision!")
print()

# Show hidden layer activations
print("Hidden Layer Activations for Each Input:")
print("-"*70)
hidden_input = np.dot(weights_input_hidden, X) + bias_hidden
hidden_output = sigmoid(hidden_input)

for i in range(4):
    print(f"Input [{X[0,i]}, {X[1,i]}]: Hidden neurons = [{hidden_output[0,i]:.3f}, {hidden_output[1,i]:.3f}]")

print()
print("The hidden neurons learned to transform the input so that")
print("XOR becomes solvable! This is the magic of neural networks.")