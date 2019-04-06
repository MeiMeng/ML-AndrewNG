#一、[线性回归](https://github.com/MeiMeng/ML-AndrewNG/tree/master/ML_ex1)
### ex1 - [一元线性回归](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/ex1.m)
#### 1.[训练数据集：x,y](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/ex1data1.txt)
```matlab
6.1101,17.592
5.5277,9.1302
8.5186,13.662
7.0032,11.854
5.8598,6.8233
8.3829,11.886
7.4764,4.3483
8.5781,12
6.4862,6.5987
5.0546,3.8166
...
```

&nbsp;
&nbsp;

#### 2.[训练数据可视化：](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/plotData.m)
- ![](https://upload-images.jianshu.io/upload_images/6065021-2a76bdf11c2da04a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 3.假设函数
```matlab
H(theta) = theta0*x0 + theta1*x1   %x0=1
```

&nbsp;
&nbsp;

#### 4.[代价函数](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/computeCost.m)
```matlab
predictionVector = X*theta;
E = predictionVector-y;

%每个误差算平方，然后再加起来
errorQuadraticSum = E' * E;

%代价要在误差的平方和上除以2m
J = errorQuadraticSum/(2*m);
```

&nbsp;
&nbsp;

#### 5.[假设模型中，参数theta是未知的，利用梯度下降-最小化代价函数，来求得最优参数theta：](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/gradientDescent.m)
```matlab
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

  m = length(y);     %训练数据的条数
  J_history = zeros(num_iters, 1);   %记下每次迭代后，的代价值

  for iter = 1:num_iters
      Predictions = X*theta; 
      %批量梯度下降，总共有(n+1)个theta，J关于各个theta的偏导数组成一个(n+1)维向量：(X'*(Predictions-y)) 
      theta = theta - alpha * (X'*(Predictions-y))/m;  %alpha是学习率

      J_history(iter) = computeCost(X, y, theta);  %记录每次迭代后的代价值
  end

end
```

&nbsp;
&nbsp;

#### 6.最优参数theta的假设模型 曲线：
- ![](https://upload-images.jianshu.io/upload_images/6065021-245b390b918ab1b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 7.画出： 代价函数的曲面，代价函数的等高线
```matlab
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
[X1,X2] = meshgrid(theta0_vals, theta1_vals);
surf(X1,X2,J_vals);
xlabel('\theta_0'); ylabel('\theta_1');
```
- ![](https://upload-images.jianshu.io/upload_images/6065021-23474a5784ab91e5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```matlab
% Contour plot
figure;

% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
% 10^(-2) ~ 10^3  共20个点
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))  
xlabel('\theta_0'); ylabel('\theta_1');

hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
```
- ![](https://upload-images.jianshu.io/upload_images/6065021-3d07b667a3181da3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

---

&nbsp;
&nbsp;

### ex1_multi - [多元线性回归](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/ex1_multi.m)
#### 1.[训练数据集：x1,x2, y](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/ex1data2.txt)
```matlab
2104,3,399900
1600,3,329900
2400,3,369000
1416,2,232000
3000,4,539900
1985,4,299900
1534,3,314900
1427,3,198999
...
```

&nbsp;
&nbsp;

#### 2.[特征之间 数值大小 差别较大，需要标准化特征值：](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/featureNormalize.m)
```matlab
function [X_norm, mu, sigma] = featureNormalize(X)

  X_norm = X;   %标准化后的X
  mu = zeros(1, size(X, 2));     %均值，1行n列
  sigma = zeros(1, size(X, 2));  %标准差，1行n列

  for i = 1 : size(X, 2)         %遍历X的列
      mu(1,i) = mean(X(:,i));     %求出每一列的均值
      sigma(1,i) = std(X(:,i));   %求出每一列的标准差
  end

  for j = 1 : size(X, 2)
      X_norm(:,j) = (X(:,j)-mu(1,j)) / sigma(1,j);   % x_norm = (x-mu)/sigma
  end

end
```

&nbsp;
&nbsp;

3.[代价函数：](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/computeCostMulti.m)
```matlab
function J = computeCostMulti(X, y, theta)

  m = length(y);    %训练数据条数
  J = 0;

  predictionVector = X*theta;
  E = predictionVector-y;

  %每个误差算平方，然后再加起来
  errorQuadraticSum = E' * E;

  %代价要在误差的平方和上除以2m
  J = errorQuadraticSum/(2*m);

end
```

&nbsp;
&nbsp;

#### 4.[梯度下降](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/gradientDescentMulti.m)
```matlab
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters 
    
    Predictions = X*theta; 
    E = Predictions - y;
    %批量梯度下降，总共有(n+1)个theta，J关于各个theta的偏导数组成一个(n+1)维向量：(X'*E) 
    theta = theta - alpha * (X'*E)/m;  %alpha是学习率 

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
```

&nbsp;
&nbsp;

#### 5.画出 代价值 随梯度下降迭代次数 的变化曲线：
```matlab
alpha = 0.1;      %学习率
num_iters = 100;  %总迭代次数

theta = zeros(3, 1);  %初识参数值
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);  %梯度下降

%画出 代价值曲线
figure;
%numel(J_history) 是 J_history的元素个数
plot(1:numel(J_history),  J_history,  '-b', 'LineWidth', 2);  
xlabel('Number of iterations');
ylabel('Cost J');
```
- ![](https://upload-images.jianshu.io/upload_images/6065021-a236b9f0289d6a35.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

---

&nbsp;
&nbsp;

### [正规方程法求参数theta](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex1/normalEqn.m)
`X * theta = y` 这个方程不一定有解。
只有当y在X的列空间上才有解。
但是y不一定在X的列空间，需要将y投影到X的列空间上。
求出`X * theta = y_tou`的解，也就是`X * theta = y`的最优解。
```matlab
function [theta] = normalEqn(X, y)

  theta = zeros(size(X, 2), 1);

  %正规方程法无需标准化特征值。
  theta = pinv(X'*X)*X'*y;

end
```


