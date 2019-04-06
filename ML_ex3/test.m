close all;clc;clear;
load('ex3data1.mat');

%X��Size
[m,n] = size(X);

%��1��m��Щ�����������,�õ���һ��1*m�������С�
rands = randperm(m);
X = X(rands(1:100), :);  %���ѡ��100��,��100��СͼƬ��


for d = 1:100
    %��ЩСͼƬ����10*10���С�
    subplot(10,10, d);

    %��ʼ��img
    img = zeros(sqrt(n),n/sqrt(n));

    %һ��img���к���
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


