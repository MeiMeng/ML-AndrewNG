close all;clc;clear;
load('ex3data1.mat');

%X的Size
[m,n] = size(X);

%把1到m这些数字随机打乱,得到的一个1*m数字序列。
rands = randperm(m);
X = X(rands(1:100), :);  %随机选了100行,是100个小图片。


for d = 1:100
    %这些小图片可以10*10排列。
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

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);


