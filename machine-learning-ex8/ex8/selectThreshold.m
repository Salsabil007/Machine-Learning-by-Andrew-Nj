function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
mf1 = 0;
pred = zeros(size(yval,1),1);
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    for i=1:size(yval,1)
      pred(i,1) = (pval(i,1) < epsilon);
    end
    tp=0;
    fp=0;
    fn=0;
    for i=1:size(yval,1)
      if (yval(i,1)==1 && pred(i,1)==1)
        tp = tp + 1;
      endif
      if (yval(i,1)==0 && pred(i,1)==1)
        fp = fp + 1;
      endif
      if (yval(i,1)==1 && pred(i,1)==0)
        fn = fn + 1;
      endif
      
    end
    prec = 0;
    rec = 0;
    if (tp+fp > 0)
      
      prec = tp / (tp+fp);
    endif
    if (tp + fn > 0)
      rec = tp / (tp+fn);
    endif
    F1 = 0;
    if (prec + rec > 0)
      F1 = (2*prec*rec)/(prec+rec);
    endif
    
    
    











    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
