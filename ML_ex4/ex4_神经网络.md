#### 1.随机初始化参数：[randInitializeWeights.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/randInitializeWeights.m)
```matlab
function W = randInitializeWeights(L_in, L_out)

  % You need to return the following variables correctly 
  W = zeros(L_out, 1 + L_in);

  INIT_EPSILON = 0.1;
  % [-ε,ε]之间
  % rand(i,j) 是i行j列的随机矩阵，元素都在[0,1]之间
  W = rand(size(W))*2*INIT_EPSILON - INIT_EPSILON;

end
```

&nbsp;
&nbsp;


#### 2.写出代价函数 和 梯度向量：[nnCostFunction.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/nnCostFunction.m)
- Part 1: 前向传播写出输出层 ***假设函数***，再根据假设函数写出 ***代价函数***。
- Part 2: 后向传播写出每一层的 ***误差值***。 
- Part 3: 根据误差值写出对于每层的各个θ的 ***偏导数***。
```matlab
function g = sigmoidGradient(z)

  g = zeros(size(z));

  g = sigmoid(z).*(1-sigmoid(z));

end
```

```matlab
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
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

% Theta2_grad = Theta2 = 10*26    
% d3' = 10*5000     a02 = 5000*26
Theta2_grad = (1/m)*(d3'*a02) + (lambda/m)*Theta2_reg;

% Theta1_grad = Theta1 = 25*401   
% d2' = 25*5000     a01 = 5000*401
Theta1_grad = (1/m)*(d2'*a01) + (lambda/m)*Theta1_reg;

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
```
- ![](https://upload-images.jianshu.io/upload_images/6065021-bdac48453eff4860.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 3.利用随机确定的参数值，用数值法计算代价函数对于各个θ的偏导数，利用计算相对误差的形式来检验后向传播求得的梯度的正确性：
随机确定参数值：[debugInitializeWeights.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/debugInitializeWeights.m)
数值法计算代价函数对于各个θ的偏导数：[computeNumericalGradient.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/computeNumericalGradient.m)
计算相对误差来 ***检验后向传播求得的梯度向量的正确性***：[checkNNGradients.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/checkNNGradients.m)

&nbsp;

#### 4.利用梯度下降法求代价函数最优点的参数θ：[ex4.m](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex4/ex4.m)
```matlab
%% ================ Part 6: Initializing Pameters ================

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================

%  After you have completed the assignment, change the MaxIter to a larger
options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(initial_params) nnCostFunction(initial_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

```

