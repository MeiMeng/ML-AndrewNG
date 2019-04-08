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
                 hidden_layer_size, (input_layer_size + 1));   % 25*401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));         % 10*26

% Setup some useful variables
m = size(X, 1);  % 5000
         
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------------------
% Part 1: Feedforward

%输入层
a1 = X;                % 5000*400
a01 = [ones(m,1) a1];  % 5000*401

%隐藏层
%25*5000 = 25*401 * 401*5000
z2 = Theta1 * a01';    % 25*5000
a2 = sigmoid(z2);      % 25*5000
a2 = a2';              % 5000*25
a02 = [ones(m,1) a2];  % 5000*26

%输出层
% 10*5000 = 10*26 * 26*5000
z3 = Theta2 * a02';    % 10*5000
a3 = sigmoid(z3);      % 10*5000
a3 = a3';              % 5000*10
H = a3;                % 5000*10

%将每一个y，转换成一个1*10的向量，以y值作为索引的那个元素是1，其他元素都为0
yTag = zeros(m, num_labels);  % 5000*10
for i = 1:m
    yTag(i, y(i)) = 1;
end

% Theta1 是 25*401
% Theta2 是 10*26
Theta1_reg = [zeros(hidden_layer_size, 1)  Theta1(:, 2:end)];
Theta2_reg = [zeros(num_labels, 1)  Theta2(:, 2:end)];

% 其中 yTag和H 同维。
J = (-1/m)*( sum(sum(yTag.*log(H))) + sum(sum((1-yTag).*log(1-H))) )  +  (lambda/(2*m))*(sum(sum((Theta1_reg).^2))+sum(sum((Theta2_reg).^2)));

% -------------------------------------------------------------------------
% Part 2: backpropagation
% d = δ
d3 = a3-yTag;   % 5000*10

Theta2 = Theta2(:, 2:end);   % 10*25

% (5000*10 * 10*25)  .* 5000*25
d2 = d3*Theta2 .* (sigmoidGradient(z2))';   % 5000*25


% -------------------------------------------------------------------------
% Part 3: Implement regularization with the gradients.

% Theta2_grad=Theta2=10*26    
% d3'=10*5000  a02=5000*26
Theta2_grad = (1/m)*(d3'*a02) + (lambda/m)*Theta2_reg;

% Theta1_grad=Theta1=25*401   
% d2'=25*5000  a01=5000*401
Theta1_grad = (1/m)*(d2'*a01) + (lambda/m)*Theta1_reg;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
