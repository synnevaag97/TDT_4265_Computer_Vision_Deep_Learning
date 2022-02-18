import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    #batch_size, width = X.shape

    mean = np.mean(X.flatten())
    std = np.std(X.flatten())

    X = (X-mean)/std # Image normalization per batch. Need to find mean and std for entire set. 
    ones = np.ones((len(X), 1)) # Array for ones.
    X = np.concatenate((X,ones), axis=1) # Appends a one for each row for the bias trick

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)

    Cn = - np.sum(targets*np.log(outputs), axis=1)
    C = np.sum(Cn)/len(targets)

    return C


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        self.hidden_layer_output = [None for i in range(len(self.neurons_per_layer)-1)]

        # Initialize the weights
        self.ws = []
        prev = self.I
        input_neurons = [self.I]
        for i in range(len(self.neurons_per_layer)):
            input_neurons.append(self.neurons_per_layer[i])
        for size, size_input_neurons in zip(self.neurons_per_layer, input_neurons):
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(0,1/np.sqrt(size_input_neurons), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        #w1 = np.random.uniform(-1, 1, (785, 64))
        #w2 = np.random.uniform(-1, 1, (64, 10))
        self.grads = [None for i in range(len(self.ws))]
        self.z = [None for i in range(len(self.ws)-1)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        out = X
        for i in range(len(self.neurons_per_layer)-1):
            self.z[i] = out.dot(self.ws[i])
            if self.use_improved_sigmoid:
                self.hidden_layer_output[i] = 1.7159*np.tanh(2/3*self.z[i]) # Sigmoid as activation function. aka self.a_j
                out = self.hidden_layer_output[i]
            else:
                self.hidden_layer_output[i] = 1/(1+np.exp(-self.z[i])) # Sigmoid as activation function. aka self.a_j
                out = self.hidden_layer_output[i]

        zk = self.hidden_layer_output[-1].dot(self.ws[-1])
        zk_mark = np.sum(np.exp(zk),axis = 1) 
        pred_target = (np.exp(zk).T/(zk_mark)).T # Softmax as activation function. aka a_k

        return pred_target

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        
        delta_prev = -(targets - outputs)

        for i in range(len(self.hidden_layer_output), 0, -1):
            #print("In for loop")
            self.grads[i] = self.hidden_layer_output[i-1].T.dot(delta_prev)/self.hidden_layer_output[i-1].shape[0]
            #print(self.grads[i].shape)

        #zj = X.dot(self.ws[0])
            if self.use_improved_sigmoid:
                f_derivative = 1.7159*2/(3*np.cosh(2/3*self.z[i-1])**2)
            else:
                f_derivative = (1/(1+np.exp(-self.z[i-1])))*(1-(1/(1+np.exp(-self.z[i-1]))))

            delta_prev = f_derivative*(delta_prev@self.ws[i].T)
    
        self.grads[0] = X.T.dot(delta_prev)/X.shape[0]

        #print(self.grads[0].shape)
        #print(len(self.grads))
        #print(self.grads)

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    target_encode = np.zeros((len(Y),num_classes))
    for i in range(len(Y)):
        target_encode[i,Y[i]] = 1
    return target_encode


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
