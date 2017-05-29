function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Implement fast forward propagation
a1 = [ones(1,m); X']; % 401 X 5000
a2 = sigmoid(Theta1*a1); % 25 X 5000
a3 = sigmoid(Theta2*[ones(1,m); a2]); % 10 * 5000
h = a3;

% recode y into Y matrix
Y = zeros(num_labels, m);
Y(sub2ind(size(Y), y', 1:m)) = 1;

% Calculate cost
J = (1 / m) * sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h)));

% Add regularization theta1 to cost
regTheta1 = Theta1(:, 2:end);
J = J + (lambda / (2*m)) .* sum(sum(regTheta1 .^ 2));
% Add regularization theta2 to cost
regTheta2 = Theta2(:, 2:end);
J = J + (lambda / (2*m)) .* sum(sum(regTheta2 .^ 2));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


for t = 1:m,

  % Set the input layer’s values (a(1)) to the t-th training example x(t)
  a1 = [ones(1,1); X(t, :)'];
  
  % Perform a feedforward pass, computing the activations (z(2), a(2), z(3), a(3)) for layers 2 and 3
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  % For each output unit k in layer 3 (the output layer), set δ(3) = (a(3) − yk)
  d3 = a3 - Y(:,t);
  
  % For the hidden layer l = 2, set δ(2) =  Θ(2) T δ(3). ∗ g′(z(2))
  d2 = (Theta2' * d3) .* sigmoidGradient([1 ; z2]);
  
  % remove δ(2)
  d2 = d2(2:end);
  
  % Accumulate the gradient
  Theta1_grad = Theta1_grad + d2 * a1';
  Theta2_grad = Theta2_grad + d3 * a2';
  
end

% Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by 1 / m
Theta1_grad = (1 / m) .* Theta1_grad;
Theta2_grad = (1 / m) .* Theta2_grad;

% Replace the thetaOne by 0 to avoid to regularize it
regTheta1 = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
regTheta2 = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

% Regularized Neural Network
Theta1_grad = Theta1_grad + (lambda / m) * regTheta1;
Theta2_grad = Theta2_grad + (lambda / m) * regTheta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
