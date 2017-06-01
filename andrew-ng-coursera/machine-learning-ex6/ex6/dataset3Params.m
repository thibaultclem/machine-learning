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

% Init the value test for C and sigma
valueTest = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Init the error at very high number to be sure we will get a better one from model
error = 100000;

% Loop on C values
for c=1:length(valueTest)

  % Loop on sigma value
  for s=1:length(valueTest)
    
    % Train Model
    CTest= valueTest(c);
    sigmaTest = valueTest(s);
    fprintf(['Testing with combination of C/sigma: C = %f and sigma = %f \n'], CTest, sigmaTest);
    model= svmTrain(X, y, CTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
    
    % Do prediction on model for Xval
    predictions = svmPredict(model, Xval);
    
    % Compute the prediction error
    errorTest = mean(double(predictions ~= yval));
    
    % If new error test better that the current one, save C and sigma
    if errorTest < error
      error = errorTest;
      C = CTest;
      sigma = sigmaTest;
      fprintf(['New best combination of C/sigma is C = %f and sigma = %f \n'], C, sigma);
    end
    
  end
  
end

fprintf(['FINAL BEST COMBINATION of C/sigma is C = %f and sigma = %f \n'], C, sigma);


% =========================================================================

end
