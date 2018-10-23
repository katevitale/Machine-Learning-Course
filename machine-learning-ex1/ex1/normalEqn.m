function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1); % size is (number of features x 1). for this exercise is 2 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% X is size (47 x 2)
temp = X'*X; % should make (2 x 2) matrix
first_term = inv(temp); % should make (2 x 2) matrix
second_term = X'*y; % should make (2 x 1) matrix

theta = first_term*second_term; % should make 2 x 1 matrix
% -------------------------------------------------------------


% ============================================================

end
