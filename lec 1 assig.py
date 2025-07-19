import random
def exp_approx(x, iterations=10):
    result = 1.0
    term = 1.0
    for n in range(1, iterations):
        term *= x / n
        result += term
    return result
def tanh(x):
    ex = exp_approx(x)
    e_minus_x = exp_approx(-x)
    return (ex - e_minus_x) / (ex + e_minus_x)
def tanh_derivative(x):
    tanh_x = tanh(x)
    return 1.0 - tanh_x * tanh_x
def initialize_weights():
    return {
        "w1": random.uniform(-0.5, 0.5), "w2": random.uniform(-0.5, 0.5),
        "w3": random.uniform(-0.5, 0.5), "w4": random.uniform(-0.5, 0.5),
        "w5": random.uniform(-0.5, 0.5), "w6": random.uniform(-0.5, 0.5),
        "w7": random.uniform(-0.5, 0.5), "w8": random.uniform(-0.5, 0.5)
    }
def forward_propagation(i1, i2, b1, b2, weights):
    # Hidden layer
    h1_input = i1 * weights["w1"] + i2 * weights["w3"] + b1
    h2_input = i1 * weights["w2"] + i2 * weights["w4"] + b1
    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)
    o1_input = h1_output * weights["w5"] + h2_output * weights["w7"] + b2
    o2_input = h1_output * weights["w6"] + h2_output * weights["w8"] + b2
    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)

    return h1_input, h1_output, h2_input, h2_output, o1_input, o1_output, o2_input, o2_output
def backward_propagation(i1, i2, h1_input, h1_output, h2_input, h2_output, o1_input, o1_output, o2_input, o2_output, t1,
                         t2, weights, learning_rate):
    o1_error = t1 - o1_output
    o2_error = t2 - o2_output
    o1_delta = o1_error * tanh_derivative(o1_input)
    o2_delta = o2_error * tanh_derivative(o2_input)
    h1_error = o1_delta * weights["w5"] + o2_delta * weights["w6"]
    h2_error = o1_delta * weights["w7"] + o2_delta * weights["w8"]
    h1_delta = h1_error * tanh_derivative(h1_input)
    h2_delta = h2_error * tanh_derivative(h2_input)
    weights["w5"] += learning_rate * o1_delta * h1_output
    weights["w6"] += learning_rate * o2_delta * h1_output
    weights["w7"] += learning_rate * o1_delta * h2_output
    weights["w8"] += learning_rate * o2_delta * h2_output
    weights["w1"] += learning_rate * h1_delta * i1
    weights["w2"] += learning_rate * h2_delta * i1
    weights["w3"] += learning_rate * h1_delta * i2
    weights["w4"] += learning_rate * h2_delta * i2

    return weights
def train_network(data, targets, b1, b2, weights, learning_rate, epochs):
    for epoch in range(epochs):
        total_error = 0
        for (i1, i2), (t1, t2) in zip(data, targets):
            h1_input, h1_output, h2_input, h2_output, o1_input, o1_output, o2_input, o2_output = forward_propagation(i1,
                                                                                                                     i2,
                                                                                                                     b1,
                                                                                                                     b2,
                                                                                                                     weights)
            error = 0.5 * ((t1 - o1_output) ** 2 + (t2 - o2_output) ** 2)
            total_error += error
            weights = backward_propagation(i1, i2, h1_input, h1_output, h2_input, h2_output, o1_input, o1_output,
                                           o2_input, o2_output, t1, t2, weights, learning_rate)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {total_error:.6f}")
    return weights
data = [(0.05, 0.10), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
targets = [(0.01, 0.99), (0.99, 0.01), (0.99, 0.01), (0.01, 0.99)]

b1, b2 = 0.35, 0.60
weights = initialize_weights()
learning_rate = 0.5
epochs = 10000

print("Training started...")
weights = train_network(data, targets, b1, b2, weights, learning_rate, epochs)
print("Training completed.")
print("\nTesting the network:")
for (i1, i2), (t1, t2) in zip(data, targets):
    h1_input, h1_output, h2_input, h2_output, o1_input, o1_output, o2_input, o2_output = forward_propagation(i1, i2, b1,
                                                                                                             b2,
                                                                                                             weights)
    print(f"Inputs: ({i1:.2f}, {i2:.2f}), Expected: ({t1:.2f}, {t2:.2f}), Got: ({o1_output:.4f}, {o2_output:.4f})")