%---------------------------------------------------------------------------------------------------------------------
% VSLD-FMR
% Author: Muhammad Reza Pourshahabi
% Version Date: FEB 6, 2024
% Email: reza.pourshahabi@yahoo.com

% VSLD-FMR: Voting based on similarity of local displacement vectors feature matching refinement scheme 
% 
% MATLAB implementation of the VSLD-FMR scheme for removing mismatches in the putative set of matches.
% The code is implemented using MATLAB R2022a.
%
% If you use this code, please cite the corresponding paper:
% M. R. Pourshahabi, M. Omair Ahmad, and M. N. S. Swamy, "A Very Fast and Robust Method for Refinement
% of Putative Matches of Features in MIS Images for Robotic-Assisted Surgery," IEEE Transactions on Medical
% Robotics and Bionics, 2024.
%---------------------------------------------------------------------------------------------------------------------

%% Main function
% putativeSet: Set of matches in the putative set
% putativeSet.Count: Number of matches in the putative set
% putativeSet.Fixed: A 2-by-N array containing the xy coordinates of pairs in the fixed image
% putativeSet.Moving: A 2-by-N array containing the xy coordinates of corresponding pairs in the moving image
% r: Number of rows in the image
% c: Number of columns in the image
%
% refinedSet: Set of refined matched features with the same structure as putativeSet
% L: Resulting label of each match (1 -> True, -1 -> False)


function [refinedSet, L] = VSLDFMR(putativeSet, r, c)

%Assuming a reference resolution of 704 × 480, for images with c columns and r rows, the scale factor is defined as
scalex = c / 704;
scaley = r / 480;


% Decreasing the coefficient value of 0.55 results in an increase in the Precision metric.
% Increasing the coefficient value of 0.55 results in a decrease in the Recall metric.
scale = 0.55 * (scalex + scaley);

n_th = 2; % Threshold on the number of matches
ns_th = 2; % Threshold on the number of matches with similar displacement vectors
nt_th = 2; % Threshold on the number of matches determined to be true in stage 1
R1 = 70 * scale; % Radius of the circular neighborhood used in Stage 1
R2 = 130 * scale; % Radius of the circular neighborhood used in Stage 2
D_th = 13 * scale; % Similarity threshold used for identifying similar displacement vectors
sig2 = (14 * scale)^2; % Variance parameter used in Gaussian weighted average

count = putativeSet.Count;
V = zeros(1, count); % Vote vector
L = zeros(count, 1);  % 0 -> Not Decided, 1 -> True, -1 -> False, 2 -> Unknown

% Calculating the displacement vectors
disVecX = (putativeSet.Moving(1, :))' - (putativeSet.Fixed(1, :))';
disVecY = (putativeSet.Moving(2, :))' - (putativeSet.Fixed(2, :))';


if count <= 2
    refinedSet.num = 0;
    refinedSet.Fixed = [];
    refinedSet.Moving = [];
    return;
end


% Finding the neighboring features in a circular region with radius R2 using a k-d tree
p = [(putativeSet.Fixed(1, :))' (putativeSet.Fixed(2, :))'];
[indNears, D] = rangesearch(p, p, R2,'NSMethod', 'kdtree');

%% Stage 1
for i = 1 : count
    idxNears = indNears{i}(2:length(find(D{i}<=R1)));
    if size(idxNears, 2) < n_th
        continue;
    end
    simVec = sqrt((disVecX(idxNears) - disVecX(i)).^2 + (disVecY(idxNears) - disVecY(i)).^2);
    vinds = idxNears(simVec < D_th);
    if length(vinds) >= ns_th
        V(i) = V(i) + 2;        
        V(vinds) = V(vinds) + 1;  
    end
end

nv_th = min(6, mean(V(V>=3))); % Threshold on the number of votes: This parameter determines the minimum number of votes required for a decision in the algorithm.
idxV = V >= nv_th;
L(idxV) = 1;

%% Stage 2
K = L;
idxU = find(L == 0); %Indices of those matches still remain Unknown after Satge 1

for j = 1 : length(idxU)
    i = idxU(j);
    idxNears = indNears{i}(2:end);
    idxT = idxNears(L(idxNears)==1); % Indices of the true matches within the neighborhood defined by radius R2.
    if length(idxT) >= nt_th
        distNears = D{i}(2:end);
        distT = distNears(L(idxNears)==1);
        w = exp(-(distT.^2) / (2 * sig2));
        mX = sum((disVecX(idxT) .* ((w))'))/sum((w));
        mY = sum((disVecY(idxT) .* ((w))'))/sum((w));
        sim = sqrt((mX - disVecX(i)).^2 + (mY - disVecY(i)).^2);
        if sim <= D_th
            K(i) = 1;
        end
    end
end
L = K;
L(L==0) = -1;
idxV = find(L == 1);

%% Constructing the refined set of matched features
refinedSet.Count = length(idxV);
refinedSet.Fixed = putativeSet.Fixed(:, idxV);
refinedSet.Moving = putativeSet.Moving(:, idxV);
return;
