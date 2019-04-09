# 二、[逻辑回归(二元分类)](https://github.com/MeiMeng/ML-AndrewNG/tree/master/ML_ex2)
### ex2-[线性决策边界-无正则化](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/ex2.m)
#### 1.[训练数据集](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/ex2data1.txt)
```matlab
34.62365962451697,78.0246928153624,0
30.28671076822607,43.89499752400101,0
35.84740876993872,72.90219802708364,0
60.18259938620976,86.30855209546826,1
79.0327360507101,75.3443764369103,1
45.08327747668339,56.3163717815305,0
61.10666453684766,96.51142588489624,1
75.02474556738889,46.55401354116538,1
...
```

&nbsp;
&nbsp;

#### 2.[可视化数据集](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/plotData.m)
- ![](https://upload-images.jianshu.io/upload_images/6065021-3aa97fa938a1466e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 3.[代价函数](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/costFunction.m)
```matlab
function [J, grad] = costFunction(theta, X, y)

  m = length(y); 
  J = 0;
  grad = zeros(size(theta));

  % J对每一个theta求偏导，偏导有theta个。
  H = sigmoid(X*theta);  %假设模型函数

  % J = -(1/m) * sum(y.*log(H) + (1-y).*log(1-H));  %代价
  J = (- y'*log(H) - (1-y)'*log(1-H) )/m;

  E = H-y;  %误差
  for i = 1 : size(X, 2)  %遍历X的列,也可遍历theta的行
      % grad(i, 1) = (1/m) * sum(E .* X(:, i));  %J对theta求偏导
      grad(i, 1) = (X(:,i)'*E)/m;
  end

end
```

#### 4.利用最小化代价函数
```matlab
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

%  Set options for fminunc: 高级梯度下降, 开, 最大迭代次数, 400
options = optimset('GradObj', 'on', 'MaxIter', 400)

%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(theta)(costFunction(theta, X, y)), initial_theta, options);
```

&nbsp;
&nbsp;

#### 5.[画出决策界线](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/plotDecisionBoundary.m)
`z = X*theta`
- 决策边界是`z=0`。
- 因为`z>=0`时，预测`y=1`； `z<0`时，预测`y=0`。
```matlab
function plotDecisionBoundary(theta, X, y)

% Plot 数据
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min( X(:,2) )-2,  max( X(:,2) )+2];
    % Calculate the decision boundary line
    plot_y = (-1./theta(3)) .* (theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
```
- ![](https://upload-images.jianshu.io/upload_images/6065021-2196c74eb6ff1cb7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

---

&nbsp;
&nbsp;

### ex2_reg- [非线性决策边界-正则化](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/ex2_reg.m)
#### 1.[训练数据集](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/ex2data2.txt)
```matlab
-0.43836,0.21711,1
-0.21947,-0.016813,1
-0.13882,-0.27266,1
0.18376,0.93348,0
0.22408,0.77997,0
0.29896,0.61915,0
...
```

&nbsp;
&nbsp;

#### 2.[数据集可视化](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/plotData.m)
- ![](https://upload-images.jianshu.io/upload_images/6065021-72b093dbddb435ef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp;
&nbsp;

#### 3.[显然决策边界不是线性的了，我们需要将特征变量增加到6次](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/mapFeature.m)
```matlab
function out = mapFeature(X1, X2)
  degree = 6;
  out = ones(size(X1(:,1)));  %第一列已经是1了，且只有一列。
  for i = 1:degree   % i = 1 ~ 6
      for j = 0:i    % j = 0:1 ~ 0:6
          % out扩大一列，X1和X2的指数加起来等于i，
          % 总共(1+7)*7/2=28列
          % {i=1(2列)、i=2(3列)、i=3(4列)、i=4(5列)、i=5(6列)、i=6(7列)}
          out(:, end+1) = (X1.^(i-j)).*(X2.^j);  
      end
  end

end
```
`1`  |   `x1`  `x2`  |   `x1^2`  `x1*x2`  `x2^2`  |   `x1^3`   `(x1^2)*x2`   `x1*(x2^2)`   `x2^3`   |  
   `x1^4`  `(x1^3)*x2`   `(x1^2)*(x2^2)`   `x1*(x2^3)`    `x2^4`     |   
  `x1^5`   `(x1^4)*x2`   `(x1^3)*(x2^2)`   `(x1^2)*(x2^3)`   `x1*(x2^4)`   `x2^5`    |  
 `x1^6`   `(x1^5)*x2`   `(x1^4)*(x2^2)`   `(x1^3)*(x2^3)`   `(x1^2)*(x2^4)`   `x1*(x2^5)`    `x2^6`  

&nbsp;
&nbsp;

#### 4.[加了正则化项的 代价函数](https://github.com/MeiMeng/ML-AndrewNG/blob/master/ML_ex2/costFunctionReg.m)
```matlab
function [J, grad] = costFunctionReg(theta, X, y, lambda)

  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));


  H = sigmoid(X*theta);  %假设模型函数
  reg_theta = [0; theta(2:end)];  %不正则化 theta0

  %加了正则项的代价
  J = (-1/m)*(y'*log(H)+(1-y)'*log(1-H)) + lambda/(2*m)*(reg_theta'*reg_theta);  

  E = H-y;  %误差
  grad = ( X'*E )./m + (lambda/m).*reg_theta;

end
```

&nbsp;
&nbsp;

#### 5.优化代价函数
分别使正则化系数 `lamda = 0、10、1`
```matlab
initial_theta = zeros(size(X, 2), 1);

lambda = 0;

options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
str = sprintf('%s%d%s', 'lambda=',lambda,' Decision boundary');
legend('y = 1', 'y = 0', str);
```
- ![过拟合](https://upload-images.jianshu.io/upload_images/6065021-4fc6fb5fb7681670.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- ![欠拟合](https://upload-images.jianshu.io/upload_images/6065021-bb9d443ca342009e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- ![正好](https://upload-images.jianshu.io/upload_images/6065021-56eb30cb03447da3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


