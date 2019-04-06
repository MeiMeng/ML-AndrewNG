function [] = show_img(X)
%SHOW_IMG 此处显示有关此函数的摘要
%   此处显示详细说明

    n = length(X);

    img = zeros(sqrt(n), n/sqrt(n));

    %一个img的行和列
    row = size(img,1);
    col = size(img,2);
    for i = 1:row
        for j = 1:col
            img(i,j) = X(1, (i-1)*col+j);
        end
    end

    imshow(img', [-1 1]), colormap gray;
    

end

