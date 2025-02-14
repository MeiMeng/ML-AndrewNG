function [h, display_array] = displayData(X, example_width)  


%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% X只传入100行, example_width没有传入
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));   % example_width = 根号400 = 20   
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);  % 100*400
example_height = (n / example_width);   % example_height = 400/20 = 20

% Compute number of items to display    % 这100行(个)数字，显示成10*10表格展示:
display_rows = floor(sqrt(m));          % display_rows = 根号100 = 10
display_cols = ceil(m / display_rows);  % display_cols = 100/10 = 10

% Between images padding
pad = 1;

% Setup blank display
% -1 * ones[11*21行，11*21列]
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
