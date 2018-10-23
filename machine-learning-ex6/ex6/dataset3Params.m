function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
    % range of C values to try
    rangeC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    
    % range of sigma values to try
    rangeSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    
    errors = zeros(length(rangeC), length(rangeSigma));
    
    for c = 1:length(rangeC)
        for s = 1:length(rangeSigma)
            
            % calculating model parameters based on syntax given in tutorials
            % mechanism of svmTrain and gaussianKernel is opaque to me!
            model = svmTrain(X, y, rangeC(c), @(x1, x2)gaussianKernel(x1, x2, rangeSigma(s)));
            
            % make predictions using svmPredict
            predictions = svmPredict(model, Xval);
            
            % calculate classification error and store in errors matrix
            % rows of errors correspond to values in rangeC
            % columns of errors correspond to values in rangeSigma
            errors(c,s) = mean(double(predictions ~= yval));
            
        end
    end

    % find the indices of the minimum of the errors matrix
    % these correspond to the indices of rangeC and rangeSigma 
    % that correspond to the optimal C and sigma values
    [Y,~] = min(errors);
    [~, s_ind] = min(Y);
    [~, c_ind] = min(errors(:,s_ind));
    
    C = rangeC(c_ind);
    sigma = rangeSigma(s_ind);

    
% =========================================================================

end
