在这个练习中，您将实现正规化的线性回归，和用它来研究模型的 不同的偏差-方差 特性。
- 预测 水库的水位变化时，水流出水坝的量。

&nbsp;

#### 1.数据可视化
![](https://upload-images.jianshu.io/upload_images/6065021-1c479b250a9a3426.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 2.写出带正则化项的代价函数和梯度
```matlab
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  % n*1

% ====================== YOUR CODE HERE ======================

% X=m*n  theta=n*1
H = X*theta;  % m*1
E = H-y;      % m*1

theta_reg = [0; theta(2:end)];  
J = (1/(2*m)) * (E'*E + lambda*(theta_reg'*theta_reg));

grad = (1/m) * (X'*E + lambda*theta_reg);  % n*1

% =========================================================================
grad = grad(:);
end
```

&nbsp;
&nbsp;

#### 3.选择多项式次数
```matlab
function [degree_vec, error_train, error_val] = ...
    validationCurve4degree(X, y, Xval, yval)


m = size(X, 1);

% Selected values of dimension (you should not change this)
degree_vec = zeros(10,1);
for i = 1:10
    degree_vec(i) = i;
end

% You need to return these variables correctly.
error_train = zeros(length(degree_vec), 1);
error_val = zeros(length(degree_vec), 1);


% ====================== YOUR CODE HERE ======================

for p = 1:length(degree_vec)
    [X_poly] = polyFeatures(X, p);
    [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
    X_poly = [ones(m, 1), X_poly];                   % Add Ones
    
    X_poly_val = polyFeatures(Xval, p);
    X_poly_val = bsxfun(@minus, X_poly_val, mu);
    X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
    X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];   % Add Ones

    [theta] = trainLinearReg(X_poly, y, 0);
    [error_train(p), grad_train] = linearRegCostFunction(X_poly, y, theta, 0);
    [error_val(p), grad_val] = linearRegCostFunction(X_poly_val, yval, theta, 0);
end 

% =========================================================================
end
```
![p=3 比较合适](https://upload-images.jianshu.io/upload_images/6065021-92795ffe0033fc9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 4.选择正则化参数λ
```matlab
function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

% Selected values of lambda (you should not change this)
% lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
lambda_vec = zeros(11,1);
lambda_vec(1) = 0.01;
for i = 2:11
    lambda_vec(i) = lambda_vec(i-1) * 2;
end

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    [theta] = trainLinearReg(X, y, lambda);
    [error_train(i), grad_train] = linearRegCostFunction(X, y, theta, 0);
    [error_val(i), grad_val] = linearRegCostFunction(Xval, yval, theta, 0);
end

% =========================================================================
end
```
![λ=1.7 比较适合](https://upload-images.jianshu.io/upload_images/6065021-65c1277bcf5b0c93.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 5.绘制学习曲线
```matlab
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================

% 我们要用分别用i条数据训练出的参数，计算训练集代价和验证代价。每次验证代价都要用全部验证集计算。
for i = 1:m
    [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
    % 由于参数是已经训练好的，我们只是用参数来计算代价值，所以我们把lambda设置为0。
    [error_train(i), grad_train] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
    [error_val(i), grad_val] = linearRegCostFunction(Xval, yval, theta, 0);
end 

% =========================================================================
end
```
- ![过拟合](https://upload-images.jianshu.io/upload_images/6065021-ec9ed91e9efa59c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- ![just right](https://upload-images.jianshu.io/upload_images/6065021-88b0b981bab8fc89.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- ![欠拟合](https://upload-images.jianshu.io/upload_images/6065021-38ff17f301a5852e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




