### Logistic多元回归
#### 1.加载数据
```matlab
>> load('ex3data1.mat');
>> whos
  Name      Size          Type 
  X         5000x400      double              
  y         5000x1        double     
```
X是5000条数据，y是5000个标签。
X的每一行都有400个元素。这400个元素代表20\*20的矩阵。
每个20\*20的矩阵，是一个灰度图片。

&nbsp;

#### 2.可视化数据
```matlab
close all;clc;clear;
load('ex3data1.mat');

%X的Size
[m,n] = size(X);

%把1到m这些数字随机打乱,得到的一个1*m数字序列。
rands = randperm(m);
X = X(rands(1:100), :);  %随机选了100行,是100个小图片。

for d = 1:100
    %这100个小图片可以10*10排列。
    subplot(10,10, d);

    %初始化img
    img = zeros(sqrt(n),n/sqrt(n));

    %一个img的行和列
    row = size(img,1);
    col = size(img,2);
    for i = 1:row
        for j = 1:col
            img(i,j) = X(d, (i-1)*col+j);
        end
    end

    imshow(img', [-1 1]), colormap gray;
    title(y(rands(d)));
end
```
![](https://upload-images.jianshu.io/upload_images/6065021-026bce04ae712b85.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


&nbsp;
&nbsp;

#### 3.二元分类的代价函数
```matlab
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));  % (n+1)*1

H = sigmoid(X * theta);  % 1*m
theta_reg = [0; theta(2:end, :)];  % (n+1)*1
J = (-1/m)*(y'*log(H)+(1-y)'*log(1-H)) + (lambda/(2*m))*(theta_reg'*theta_reg);

grad = (1/m)*X'*(H-y) + (lambda/m).*theta_reg;  % (n+1)*1

end
```

&nbsp;
&nbsp;

#### 4.一对多优化，得到10组模型参数theta
```matlab
function [all_theta] = oneVsAll(X, y, num_labels, lambda)

  m = size(X, 1);  % 5000
  n = size(X, 2);  % 400

  all_theta = zeros(num_labels, n + 1);   % 10*(n+1)
  X = [ones(m, 1) X];

  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);

  % Run fmincg to obtain the optimal theta
  % This function will return theta and the cost 
  for c = 1:10
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
            initial_theta, options);
    all_theta(c,:) = theta';
  end

end
```

&nbsp;
&nbsp;

5.一对多，预测，选择预测值最大的
```matlab
function p = predictOneVsAll(all_theta, X)

  m = size(X, 1);  
  num_labels = size(all_theta, 1);  % 10

  % You need to return the following variables correctly 
  p = zeros(size(X, 1), 1);  % m*1

  % Add ones to the X data matrix
  X = [ones(m, 1) X];


  A = sigmoid(X * all_theta');  % m*10，每一行数据，用10列参数theta，均有10个预测值。
  [M,p] = max(A, [], 2);      % 每一行求最大值
  % M，是每行最大值，维数m*1；
  % p，是每行最大值的列号，维数m*1。

end
```

最终准确率是 94.96%
&nbsp;
&nbsp;
这部分用Logistic回归实现了多元分类问题，但是在特征变量多的时候，手动构造非线性边界太复杂，所以引出神经网络。
&nbsp;
---
&nbsp;
&nbsp;

### 神经网络
这个章节，将使用已经训练好的参数，做前向传播，去预测。
下个章节将实现后向传播，去训练参数。
#### 1.加载参数
```matlab
>> load('ex3weights.mat');
>> whos
  Name         Size          
  Theta1      25x401                   
  Theta2      10x26           
```
此模型只有三层：
1个输入层、1个隐藏层、1个输出层。
输入层有400个神经元、隐藏层有25个神经元、输出层有10个神经元。

&nbsp;

#### 2.利用已经给出的模型参数，编写前向传播代码，进行预测：
```matlab
function [p] = predict(Theta1, Theta2, X)

  m = size(X, 1);  % 5000
  num_labels = size(Theta2, 1); % 10

  % You need to return the following variables correctly 
  p = zeros(size(X, 1), 1);  % 5000*1


  % 输入层
  X = [ones(m,1) X];

  % 隐藏层
  % 25*1 = 25*401 * 401*1
  A = sigmoid(Theta1 * X');  % 25*5000
  A = [ones(m,1) A'];  % 5000*26

  % 输出层
  % 10*1 = 10*26 * 26*1
  H = sigmoid(Theta2 * A');  % 10*5000
  H = H';  % 5000*10

  [M,p] = max(H, [], 2);  %选出每一行中最大的那个
  % M，维数m*1，是每行最大值；
  % p，维数m*1，是每行最大值的列号

end
```

