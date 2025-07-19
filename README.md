*Neural Network Implementation*

A simple implementation of a feedforward neural network in Python with forward and backward propagation, using the tanh activation function and an approximation of the exponential function.

*Features*

Forward Propagation: Computes the output of a neural network with two input neurons, two hidden neurons, and two output neurons.
Backward Propagation: Updates weights using backpropagation to minimize the error between predicted and target outputs.
Tanh Activation: Uses the hyperbolic tangent (tanh) function, implemented with an exponential approximation.
Training: Trains the network on a small dataset with a configurable learning rate and number of epochs.

*Run the script to train the neural network on the provided dataset.*

The network is trained for 10,000 epochs with a learning rate of 0.5.
After training, the script tests the network on the input data and prints the predicted outputs compared to the expected targets.

*Example Output*

Training started...
Epoch 0, Error: 0.XXXXXX
Epoch 1000, Error: 0.XXXXXX
...
Training completed.
Testing the network:
Inputs: (0.05, 0.10), Expected: (0.01, 0.99), Got: (0.XXXX, 0.XXXX)
Inputs: (0.00, 1.00), Expected: (0.99, 0.01), Got: (0.XXXX, 0.XXXX)
...

*Dependencies*

*Python 3.11
*random (standard library)

*Notes*

The exponential function is approximated using a Taylor series for simplicity.
The network architecture is fixed: 2 input neurons, 2 hidden neurons, and 2 output neurons.
The dataset and targets are hardcoded for demonstration purposes.
<img width="1825" height="728" alt="Screenshot 2025-07-20 014014" src="https://github.com/user-attachments/assets/7e238ba7-58e8-4538-98a0-80c9b6217755" />
