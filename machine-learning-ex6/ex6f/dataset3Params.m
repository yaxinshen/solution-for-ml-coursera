function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

Cl = [0.01;0.03;0.1;0.3;1;3;10;30];
sigmal = [0.01;0.03;0.1;0.3;1;3;10;30];
model= svmTrain(X, y, 0.3, @(x1, x2) gaussianKernel(x1, x2, 3));
predictions = svmPredict(model, Xval);
min = mean(double(predictions ~= yval));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for n=1:8
	for t=1:8
		model= svmTrain(X, y, Cl(n), @(x1, x2) gaussianKernel(x1, x2, sigmal(t)));
		predictions = svmPredict(model, Xval);
		a = mean(double(predictions ~= yval));
		if a<min
			cmin = Cl(n);
			smin = sigmal(t);
			min = a;
		end
	end
end


C = cmin;
sigma = smin;


% =========================================================================

end
