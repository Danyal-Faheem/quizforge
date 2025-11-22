import { Question } from "./types";

export const QUIZ_DATA: Question[] = [
  {
    "question": "What is the 'Vanishing Gradient' problem primarily caused by?",
    "options": [
      "Using activation functions like Sigmoid or Tanh that saturate (derivative becomes $\\approx 0$) for large inputs.",
      "Using a learning rate that is too large.",
      "Using Rectified Linear Units (ReLU).",
      "The loss function being non-convex."
    ],
    "answer": ["Using activation functions like Sigmoid or Tanh that saturate (derivative becomes $\\approx 0$) for large inputs.", "Using Rectified Linear Units (ReLU)."],
    "explanation": "In deep networks, gradients are computed using the chain rule, which involves multiplying derivatives layer by layer. Sigmoid and Tanh derivatives are always $< 1$ (and close to 0 at tails). Multiplying many small numbers results in a gradient that vanishes to zero, stopping early layers from learning.",
    "hint": "Think about the slope of the S-curve at the far ends."
  },
  {
    "question": "In the context of optimization, what is a 'Saddle Point'?",
    "options": [
      "A point where the gradient is zero, but it is a minimum in some directions and a maximum in others.",
      "The global minimum of the loss surface.",
      "A point where the gradient is infinite.",
      "A local maximum where the Hessian is negative definite."
    ],
    "answer": "A point where the gradient is zero, but it is a minimum in some directions and a maximum in others.",
    "explanation": "Saddle points are stationary points (gradient is 0). However, the curvature (Hessian) implies it curves up like a U in one dimension but down like an inverted U in another. In high-dimensional neural nets, these are much more common than true local minima.",
    "hint": "Think of the shape of a horse saddle."
  },
  {
    "question": "Why is the Rprop optimizer generally unsuitable for Mini-batch training?",
    "options": [
      "It relies entirely on the sign of the gradient, which is noisy and fluctuates in mini-batches.",
      "It requires second-order derivative calculations (Hessian).",
      "It cannot handle positive weights.",
      "It is computationally too expensive for modern hardware."
    ],
    "answer": "It relies entirely on the sign of the gradient, which is noisy and fluctuates in mini-batches.",
    "explanation": "Rprop works by increasing step size if the gradient sign stays consistent and decreasing it if the sign flips. In mini-batch training, the gradient sign might flip just due to the randomness of the batch selection, not because the actual function shape changed. This confuses Rprop.",
    "hint": "Rprop assumes the gradient direction is 'true', but batches are 'estimates'."
  },
  {
    "question": "Given a scalar function $L$ and a weight matrix $W$, what is the shape of the gradient $\\nabla_W L$?",
    "options": [
      "The transpose of the shape of $W$.",
      "The same shape as $W$.",
      "A scalar value.",
      "A column vector regardless of $W$'s shape."
    ],
    "answer": "The same shape as $W$.",
    "explanation": "The gradient must represent the sensitivity of the Loss to *each* element in $W$. Therefore, for every $w_{ij}$ in $W$, there must be a corresponding $\\partial L / \\partial w_{ij}$ in the gradient matrix.",
    "hint": "To update W, we need to subtract something of the exact same dimensions."
  },
  {
    "question": "What is the primary mechanism of 'Momentum' in optimization?",
    "options": [
      "It accumulates a running average of past gradients to smooth out oscillations.",
      "It squares the gradients to normalize step sizes.",
      "It changes the sign of the update based on the previous iteration.",
      "It resets the weights to zero every few epochs."
    ],
    "answer": "It accumulates a running average of past gradients to smooth out oscillations.",
    "explanation": "Momentum introduces a 'velocity' term. $v_t = \\beta v_{t-1} + \\eta \\nabla L$. This allows the optimizer to build up speed in directions with consistent gradients and dampen zig-zagging in directions where gradients oscillate.",
    "hint": "Physics: A ball rolling down a hill."
  },
  {
    "question": "Which of the following statements about the Universal Approximation Theorem is true?",
    "options": [
      "It requires the network to be Deep (many layers).",
      "It states that a single hidden layer with sufficient width (neurons) can approximate any continuous function.",
      "It only applies to networks using Linear activation functions.",
      "It guarantees that we can always find the optimal weights using Gradient Descent."
    ],
    "answer": "It states that a single hidden layer with sufficient width (neurons) can approximate any continuous function.",
    "explanation": "The theorem guarantees *representational* power: a wide enough shallow network *can* represent the function. It does NOT guarantee that we can *train* it easily (optimization) or that it will generalize (overfitting).",
    "hint": "Width vs. Depth."
  },
  {
    "question": "In L2 Regularization (Weight Decay), what term is added to the Loss function?",
    "options": [
      "$\\lambda \\sum |w_i|$",
      "$\\lambda \\sum w_i^2$",
      "$\\lambda \\sum \\log(w_i)$",
      "$\\lambda \\sum e^{w_i}$"
    ],
    "answer": "$\\lambda \\sum w_i^2$",
    "explanation": "L2 regularization penalizes the squared magnitude of the weights. This forces the network to prefer smaller, more diffuse weights rather than relying heavily on a few large weights.",
    "hint": "Euclidean distance squared."
  },
  {
    "question": "How does 'Dropout' affect the network during the testing/inference phase?",
    "options": [
      "We continue to drop neurons randomly.",
      "We use the full network but scale the weights (multiply by keep-prob $p$) to maintain expected magnitude.",
      "We only use the neurons that were never dropped during training.",
      "We set all weights to 1."
    ],
    "answer": "We use the full network but scale the weights (multiply by keep-prob $p$) to maintain expected magnitude.",
    "explanation": "During training, if $p=0.5$, the output is roughly half what it would be if all neurons fired. During testing, all neurons fire. To keep the mathematical expectation consistent, we must scale the weights (or activations) by $p$.",
    "hint": "The 'Bagging' effect needs to be averaged out."
  },
  {
    "question": "What is 'Internal Covariate Shift'?",
    "options": [
      "The change in the distribution of layer inputs (activations) as the parameters of the previous layers change during training.",
      "The shift in the test set distribution compared to the training set.",
      "The oscillation of the loss function during training.",
      "The vanishing of gradients in recurrent neural networks."
    ],
    "answer": "The change in the distribution of the input data (or activations) during training.",
    "explanation": "As layer $k$ updates its weights, the distribution of data flowing into layer $k+1$ changes. Layer $k+1$ constantly has to adapt to this 'moving target', slowing down training. Batch Norm fixes this.",
    "hint": "Input distribution moving around inside the network."
  },
  {
    "question": "What is the derivative of the ReLU function, $f(x) = \\max(0, x)$, for $x < 0$?",
    "options": [
      "1",
      "x",
      "0",
      "Undefined"
    ],
    "answer": "0",
    "explanation": "For negative inputs, ReLU outputs a constant 0. The derivative (slope) of a constant horizontal line is 0. This causes the 'Dead ReLU' problem.",
    "hint": "Flat line."
  },
  {
    "question": "Why is the Softmax function used in the output layer of a classifier?",
    "options": [
      "It converts raw scores (logits) into a probability distribution (summing to 1).",
      "It is computationally the fastest activation function.",
      "It prevents the gradients from exploding.",
      "It converts all negative inputs to zero."
    ],
    "answer": "It converts raw scores (logits) into a probability distribution (summing to 1).",
    "explanation": "Softmax exponentiates inputs (making them positive) and divides by the sum (normalizing them). This creates a valid probability distribution suitable for multi-class classification.",
    "hint": "Soft-maximize to probabilities."
  },
  {
    "question": "What is the specific improvement of Adam over RMSProp?",
    "options": [
      "It removes the need for a learning rate.",
      "It includes a momentum term (first moment estimate) in addition to the squared gradient (second moment).",
      "It uses the Hessian matrix for updates.",
      "It only updates weights when the loss increases."
    ],
    "answer": "It includes a momentum term (first moment estimate) in addition to the squared gradient (second moment).",
    "explanation": "RMSProp only tracks variance ($v_t$). Adam tracks both the average gradient ($m_t$, momentum) and the variance ($v_t$). It is essentially RMSProp + Momentum.",
    "hint": "Adaptive Moment Estimation."
  },
  {
    "question": "Which of the following is a property of the 'Cross-Entropy' loss function?",
    "options": [
      "It heavily penalizes confident but wrong predictions.",
      "It is only used for regression problems.",
      "It has a derivative of zero when the prediction is far from the target.",
      "It is identical to L2 loss."
    ],
    "answer": "It heavily penalizes confident but wrong predictions.",
    "explanation": "Loss = $-\\log(y)$. If the true class is 1, but the model predicts $y=0.0001$, the loss is $-\\log(0.0001)$ which is a huge number. This creates a strong gradient to correct the error.",
    "hint": "Log loss approaches infinity as error increases."
  },
  {
    "question": "What is the Jacobian Matrix $J_x(y)$ if $y$ is a vector of size $M$ and $x$ is a vector of size $N$?",
    "options": [
      "An $N \\times M$ matrix.",
      "An $M \\times N$ matrix containing all partial derivatives $\\partial y_i / \\partial x_j$.",
      "A vector of size $M+N$.",
      "A scalar value."
    ],
    "answer": "An $M \\times N$ matrix containing all partial derivatives $\\partial y_i / \\partial x_j$.",
    "explanation": "The Jacobian represents the derivative of a vector function. Row $i$ corresponds to output $y_i$, and column $j$ corresponds to input $x_j$.",
    "hint": "Rows = Outputs, Columns = Inputs."
  },
  {
    "question": "Why do we initialize weights randomly instead of setting them all to zero?",
    "options": [
      "To break symmetry.",
      "To ensure the loss starts at zero.",
      "To increase the learning rate.",
      "To make the model deterministic."
    ],
    "answer": "To break symmetry.",
    "explanation": "If all weights are 0, all neurons in a layer compute the same output and receive the same gradient. They will update identically and remain identical forever. Randomness ensures they learn different features.",
    "hint": "We want neurons to be different from each other."
  },
  {
    "question": "In the context of 'Influence Diagrams' for backpropagation, if a node $x$ influences a node $z$ through multiple paths, how is the total derivative calculated?",
    "options": [
      "By multiplying the derivatives along all paths.",
      "By summing the derivatives along all independent paths from $x$ to $z$.",
      "By taking the maximum derivative among the paths.",
      "By taking the average of the derivatives."
    ],
    "answer": "By summing the derivatives along all independent paths from $x$ to $z$.",
    "explanation": "This is the multivariable chain rule. Total influence = Sum of influence through Path A + Influence through Path B + ...",
    "hint": "Summation of influences."
  },
  {
    "question": "True or False: A 2-layer MLP (1 hidden layer) requires exponentially fewer neurons than a deep MLP to represent the parity (XOR) function of N inputs.",
    "options": [
      "True",
      "False"
    ],
    "answer": "False",
    "explanation": "It is the opposite. A **deep** network requires linearly scaling neurons ($O(N)$). A **shallow** (1 hidden layer) network requires exponentially many neurons ($2^{N-1}$) to represent N-bit parity/XOR.",
    "hint": "Depth is efficient; Shallowness is expensive for complex logic."
  },
  {
    "question": "What is 'Early Stopping'?",
    "options": [
      "A regularization technique where training is halted when performance on a validation set degrades.",
      "Stopping the gradient descent when the learning rate is too high.",
      "Removing the first few layers of the network.",
      "Initializing weights to zero."
    ],
    "answer": "A regularization technique where training is halted when performance on a validation set degrades.",
    "explanation": "Early stopping prevents the model from overfitting to the training data. We monitor a held-out validation set; when validation error goes up (even if training error goes down), we stop.",
    "hint": "Quit while you're ahead."
  },
  {
    "question": "How does Gradient Clipping help with training Recurrent Neural Networks (RNNs) or very deep networks?",
    "options": [
      "It increases the learning rate dynamically.",
      "It prevents the Exploding Gradient problem by capping the norm of the gradient.",
      "It adds noise to the gradients.",
      "It enforces sparsity in weights."
    ],
    "answer": "It prevents the Exploding Gradient problem by capping the norm of the gradient.",
    "explanation": "In deep/recurrent nets, gradients can accumulate to massive numbers, causing weights to jump to NaN. Clipping rescales the gradient vector if it exceeds a certain threshold.",
    "hint": "Keeping the update size safe."
  },
  {
    "question": "Which of the following implies 'Overfitting'?",
    "options": [
      "High Training Error, High Test Error.",
      "Low Training Error, High Test Error.",
      "Low Training Error, Low Test Error.",
      "High Training Error, Low Test Error."
    ],
    "answer": "Low Training Error, High Test Error.",
    "explanation": "Overfitting means the model has memorized the training data (Low error) but fails to generalize to new, unseen data (High test error).",
    "hint": "Good at the practice test, bad at the real exam."
  },
  {
    "question": "In Batch Normalization, what statistics are used to normalize the data during the **training** phase?",
    "options": [
      "The global running mean and variance.",
      "The mean and variance of the current mini-batch.",
      "Fixed constants 0 and 1.",
      "The statistics of the test set."
    ],
    "answer": "The mean and variance of the current mini-batch.",
    "explanation": "During training, BN calculates the mean and variance specifically for the current batch of data passed through. (During testing, it uses the running averages).",
    "hint": "Based on the current batch."
  },
  {
    "question": "What is the derivative of the function $z = Wx$ with respect to the vector $x$? (Where $W$ is a matrix).",
    "options": [
      "$x^T$",
      "$W$",
      "$W^T$",
      "$x$"
    ],
    "answer": "$W$",
    "explanation": "If $z$ is a vector function of vector $x$, the Jacobian is the matrix $W$. $\\nabla_x (Wx) = W$. (Note: depending on layout convention, it might be $W^T$, but conceptually it is the weight matrix itself).",
    "hint": "Linear algebra analog to d(ax)/dx = a."
  },
  {
    "question": "Why is the 'Sigmoid' function typically avoided in the hidden layers of deep networks?",
    "options": [
      "It is not differentiable.",
      "It causes the Vanishing Gradient problem because its maximum derivative is only 0.25.",
      "It produces outputs greater than 1.",
      "It is linear."
    ],
    "answer": "It causes the Vanishing Gradient problem because its maximum derivative is only 0.25.",
    "explanation": "Since the derivative is at most 0.25, multiplying it across many layers ($0.25 \times 0.25 \times \dots$) results in gradients effectively becoming zero.",
    "hint": "It squashes gradients too much."
  },
  {
    "question": "What does the 'Bias-Variance Tradeoff' describe?",
    "options": [
      "The trade-off between training speed and memory usage.",
      "The tension between a model being too simple (high bias, underfitting) and too complex (high variance, overfitting).",
      "The trade-off between the number of neurons and layers.",
      "The balance between positive and negative weights."
    ],
    "answer": "The tension between a model being too simple (high bias, underfitting) and too complex (high variance, overfitting).",
    "explanation": "We want a model that captures the true trend (low bias) but isn't sensitive to noise in the specific training set (low variance).",
    "hint": "Simplicity vs. Complexity."
  },
  {
    "question": "What is 'Data Augmentation'?",
    "options": [
      "Adding more layers to the network.",
      "Collecting more real-world data.",
      "Artificially increasing the training set by applying transformations (e.g., rotation, noise) to existing data.",
      "Increasing the learning rate."
    ],
    "answer": "Artificially increasing the training set by applying transformations (e.g., rotation, noise) to existing data.",
    "explanation": "It's a cheap way to get 'more' data and force the model to learn invariants (e.g., a rotated cat is still a cat).",
    "hint": "Making new data from old data."
  },
  {
    "question": "True or False: The derivative of the Loss w.r.t a weight $w_{ij}$ in a standard MLP depends on the activation value of the input neuron $x_i$.",
    "options": [
      "False",
      "True"
    ],
    "answer": "True",
    "explanation": "In the chain rule for a weight $w$ connected to input $x$, the derivative term is often $\\frac{\\partial L}{\\partial z} \\cdot x$. If the input $x$ is 0, the gradient for that weight is 0, and it won't learn.",
    "hint": "Weights on inactive inputs don't get updated."
  },
  {
    "question": "What is the 'Condition Number' of the Hessian matrix relevant for?",
    "options": [
      "Determining the number of layers.",
      "Predicting the convergence speed of Gradient Descent; a high condition number means a 'ravine' shape and slow convergence.",
      "Calculating the classification accuracy.",
      "Deciding when to stop training."
    ],
    "answer": "Predicting the convergence speed of Gradient Descent; a high condition number means a 'ravine' shape and slow convergence.",
    "explanation": "A high condition number means the curve is very steep in one direction and very flat in another. SGD bounces back and forth on the steep sides and moves slowly along the flat bottom.",
    "hint": "Shape of the loss valley."
  },
  {
    "question": "Why do we use 'Mini-batches' instead of full 'Batch' training?",
    "options": [
      "It guarantees finding the global minimum.",
      "It is computationally more efficient (fits in memory) and the noise helps escape saddle points.",
      "It removes the need for backpropagation.",
      "It makes the math simpler."
    ],
    "answer": "It is computationally more efficient (fits in memory) and the noise helps escape saddle points.",
    "explanation": "Full batch is too big for RAM. Stochastic (1 sample) is too noisy and slow (cannot vectorize). Mini-batch is the sweet spot.",
    "hint": "Efficiency and helpful noise."
  },
  {
    "question": "Which of the following is a valid method to prevent 'Exploding Gradients'?",
    "options": [
      "Using ReLU activations.",
      "Gradient Clipping.",
      "Increasing the learning rate.",
      "Removing Batch Normalization."
    ],
    "answer": "Gradient Clipping.",
    "explanation": "If the gradient norm exceeds a threshold (e.g., 5.0), we scale it down. This effectively 'clips' the cliff so we don't jump to infinity.",
    "hint": "Cutting the top off."
  },
  {
    "question": "In the definition of Empirical Risk $Loss(W) = \\frac{1}{N} \\sum div(f(X_i), d_i)$, what does $d_i$ represent?",
    "options": [
      "The predicted output.",
      "The derivative.",
      "The desired (target) output for input $i$.",
      "The input vector."
    ],
    "answer": "The desired (target) output for input $i$.",
    "explanation": "$d$ stands for desired (or label/ground truth). We want the prediction $f(X)$ to match $d$.",
    "hint": "The Ground Truth."
  },
  {
    "question": "True or False: If a function is convex, Gradient Descent with an appropriately small learning rate is guaranteed to converge to the global minimum.",
    "options": [
      "True",
      "False"
    ],
    "answer": "True",
    "explanation": "This is a fundamental property of convex functions. There are no local minima to get stuck in, only one global minimum.",
    "hint": "Convex = Bowl shape."
  },
  {
    "question": "What is the Jacobian of the function $f(z) = \\tanh(z)$ (where $z$ is a vector) look like?",
    "options": [
      "A full matrix of ones.",
      "A diagonal matrix, where $J_{ii} = 1 - \\tanh^2(z_i)$.",
      "A matrix of zeros.",
      "A lower triangular matrix."
    ],
    "answer": "A diagonal matrix, where $J_{ii} = 1 - \\tanh^2(z_i)$.",
    "explanation": "Since activation functions operate element-wise ($z_1$ affects $y_1$, $z_2$ affects $y_2$), the cross-terms $\\partial y_i / \\partial z_j$ are 0. Only the diagonal exists.",
    "hint": "Element-wise operations have diagonal Jacobians."
  },
  {
    "question": "Why is 'Polyack Averaging' used?",
    "options": [
      "To initialize weights.",
      "To smooth the trajectory of the optimization and find a better final solution.",
      "To calculate the average gradient in a batch.",
      "To normalize inputs."
    ],
    "answer": "To smooth the trajectory of the optimization and find a better final solution.",
    "explanation": "In the final stages of training, parameters oscillate around the minimum. Averaging the parameters over the last few iterations helps center on the true minimum.",
    "hint": "Averaging the path."
  },
  {
    "question": "What is 'Online Learning'?",
    "options": [
      "Training with a batch size of 1, updating immediately as data arrives.",
      "Downloading weights from a server.",
      "Training on the cloud.",
      "Using a pre-trained model."
    ],
    "answer": "Training with a batch size of 1, updating immediately as data arrives.",
    "explanation": "Online learning implies the model learns 'on the fly' from a stream of data, without storing a large dataset.",
    "hint": "Immediate updates."
  },
  {
    "question": "Which component of the Adam optimizer provides the 'adaptive learning rate' capability?",
    "options": [
      "The first moment estimate ($m_t$).",
      "The second moment estimate ($v_t$, uncentered variance).",
      "The bias correction terms.",
      "The initial learning rate."
    ],
    "answer": "The second moment estimate ($v_t$, uncentered variance).",
    "explanation": "Adam divides the update by $\\sqrt{v_t}$. If gradients are large/variable ($v_t$ is large), the step size is reduced. This is the adaptive scaling part (inherited from RMSProp).",
    "hint": "The denominator."
  },
  {
    "question": "True or False: We can compute the gradient of a neural network w.r.t. the Input vector $X$.",
    "options": [
      "False",
      "True"
    ],
    "answer": "True",
    "explanation": "Yes! While we usually compute gradients w.r.t Weights for training, we can backpropagate all the way to the input. This is used for generating Adversarial Examples or DeepDream visualizations.",
    "hint": "Useful for adversarial attacks."
  },
  {
    "question": "What is the primary disadvantage of Full Batch Gradient Descent?",
    "options": [
      "It is unstable.",
      "It is computationally expensive and memory-intensive for large datasets.",
      "It cannot find the global minimum.",
      "It requires complex implementation."
    ],
    "answer": "It is computationally expensive and memory-intensive for large datasets.",
    "explanation": "Calculating the loss on 1 million images before taking a *single* step is incredibly slow and requires fitting all data into memory.",
    "hint": "Too much data at once."
  },
  {
    "question": "L1 Regularization tends to produce:",
    "options": [
      "Sparse weight vectors (many zeros).",
      "Small but non-zero weights.",
      "Large weights.",
      "Unstable training."
    ],
    "answer": "Sparse weight vectors (many zeros).",
    "explanation": "The gradient of L1 is a constant ($\pm \lambda$). This constant subtraction forces weights to cross zero and stay there, acting as feature selection.",
    "hint": "L1 leads to sparsity."
  },
  {
    "question": "What is the purpose of the 'Forget Gate' in heuristics like Dropout?",
    "options": [
      "To erase memory in RNNs.",
      "To randomly 'forget' or mask neurons to prevent co-adaptation.",
      "To reset the learning rate.",
      "To clear the GPU cache."
    ],
    "answer": "To randomly 'forget' or mask neurons to prevent co-adaptation.",
    "explanation": "By temporarily deleting neurons, we force the remaining neurons to take over the workload, making the network more robust.",
    "hint": "Preventing reliance on specific neurons."
  },
  {
    "question": "Which of these is NOT a benefit of using a 'Deep' network over a 'Wide' shallow network?",
    "options": [
      "Parameter efficiency (fewer weights for the same function).",
      "Hierarchical feature learning.",
      "Easier optimization (convex loss surface).",
      "Ability to model complex compositionality."
    ],
    "answer": "Easier optimization (convex loss surface).",
    "explanation": "Deep networks actually have highly NON-convex loss surfaces with many saddle points, making them *harder* to optimize than shallow (or linear) ones. Their benefit is efficiency and expressivity, not ease of training.",
    "hint": "Deep learning training is hard."
  },
  {
    "question": "If we multiply the inputs to a linear network by a constant $\\alpha$, the gradients w.r.t. the first layer weights will:",
    "options": [
      "Scale by $\\alpha$.",
      "Scale by $1/\\alpha$.",
      "Remain unchanged.",
      "Become zero."
    ],
    "answer": "Scale by $\\alpha$.",
    "explanation": "Since $y = w \cdot x$, if $x$ becomes $\\alpha x$, the derivative $\\partial y / \\partial w = \\alpha x$. The gradient scales linearly with the input magnitude. (This is why we normalize inputs!).",
    "hint": "Linear relationship."
  },
  {
    "question": "In a classification task with 10 classes, what is the dimension of the target vector $d$ (one-hot encoded)?",
    "options": [
      "1",
      "10",
      "100",
      "Log(10)"
    ],
    "answer": "10",
    "explanation": "One-hot encoding represents the target class as a vector of length $K$ (number of classes), with a 1 at the correct index and 0s elsewhere.",
    "hint": "Vector size = Number of classes."
  },
  {
    "question": "What is the key difference between 'Epoch' and 'Iteration'?",
    "options": [
      "They are the same.",
      "Epoch is one pass over the full dataset; Iteration is one update step (one mini-batch).",
      "Iteration is one pass over the full dataset; Epoch is one update step.",
      "Epoch applies to Testing; Iteration applies to Training."
    ],
    "answer": "Epoch is one pass over the full dataset; Iteration is one update step (one mini-batch).",
    "explanation": "If you have 1000 items and batch size 100, you need 10 Iterations to complete 1 Epoch.",
    "hint": "Iteration is a single step."
  },
  {
    "question": "Which problem was the 'Perceptron' unable to solve, leading to the 'AI Winter'?",
    "options": [
      "AND function",
      "OR function",
      "XOR function (Linear Separability)",
      "NOT function"
    ],
    "answer": "XOR function (Linear Separability)",
    "explanation": "Minsky and Papert proved that a single layer perceptron cannot solve non-linearly separable problems like XOR.",
    "hint": "Exclusive OR."
  },
  {
    "question": "How does 'Weight Decay' relate to the loss function?",
    "options": [
      "It subtracts a value from the loss.",
      "It adds a term proportional to the magnitude of weights to the loss.",
      "It multiplies the loss by a decay factor.",
      "It ignores the loss."
    ],
    "answer": "It adds a term proportional to the magnitude of weights to the loss.",
    "explanation": "We minimize $Loss_{total} = Loss_{data} + \lambda \|W\|^2$. We add the penalty so the optimizer tries to reduce both the error and the size of the weights.",
    "hint": "Adding a penalty."
  },
  {
    "question": "True or False: The gradient of the loss always points directly to the global minimum.",
    "options": [
      "True",
      "False"
    ],
    "answer": "False",
    "explanation": "The gradient points in the direction of steepest ascent (locally). In a complex, curved landscape (like a ravine), this might not point directly toward the minimum, but rather toward the wall of the valley.",
    "hint": "It points uphill, locally."
  },
  {
    "question": "What is the Jacobian $J_x(x)$ (derivative of a vector w.r.t. itself)?",
    "options": [
      "The Zero Matrix.",
      "The Identity Matrix $I$.",
      "A vector of ones.",
      "Undefined."
    ],
    "answer": "The Identity Matrix $I$.",
    "explanation": "$\partial x_i / \partial x_j$ is 1 if $i=j$ and 0 otherwise. This forms the Identity matrix.",
    "hint": "Change in x w.r.t itself is 1."
  },
  {
    "question": "In Adam, what are the recommended default values for $\\beta_1$ (momentum) and $\\beta_2$ (RMSProp)?",
    "options": [
      "0.9 and 0.999",
      "0.5 and 0.5",
      "0.1 and 0.001",
      "0.0 and 1.0"
    ],
    "answer": "0.9 and 0.999",
    "explanation": "These are the standard values proposed in the Adam paper and used in almost all libraries (PyTorch/TensorFlow). $\\beta_1=0.9$ (standard momentum), $\\beta_2=0.999$ (long-term variance).",
    "hint": "High momentum, very high variance tracking."
  },
  {
    "question": "Why do we need 'Bias Correction' in Adam?",
    "options": [
      "To prevent division by zero.",
      "To correct the fact that the running averages start at 0, making estimates biased towards 0 in early steps.",
      "To add a bias neuron to the network.",
      "To adjust the class imbalance."
    ],
    "answer": "To correct the fact that the running averages start at 0, making estimates biased towards 0 in early steps.",
    "explanation": "Since $m_0$ and $v_0$ are initialized to 0, the moving averages are skewed toward 0 at the start. Bias correction scales them up (e.g., $m_t / (1-\beta^t)$) to get a true estimate.",
    "hint": "Correcting the zero initialization."
  },
  {
    "question": "True or False: A deeper network is always better than a shallower one.",
    "options": [
      "True",
      "False"
    ],
    "answer": "False",
    "explanation": "While deeper networks are more expressive, they are harder to train (vanishing gradients) and prone to overfitting if data is scarce. Sometimes a simpler model is better.",
    "hint": "Complexity isn't always free."
  }
];