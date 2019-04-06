function [] = show_img(X)
%SHOW_IMG �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

    n = length(X);

    img = zeros(sqrt(n), n/sqrt(n));

    %һ��img���к���
    row = size(img,1);
    col = size(img,2);
    for i = 1:row
        for j = 1:col
            img(i,j) = X(1, (i-1)*col+j);
        end
    end

    imshow(img', [-1 1]), colormap gray;
    

end

