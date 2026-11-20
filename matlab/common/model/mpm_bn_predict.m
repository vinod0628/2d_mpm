function [p, sd, logit] = mpm_bn_predict(mdl, D)
%MPM_BN_PREDICT Posterior P(deposit | evidence) of a Bayesian network classifier (MPM_BN_TRAIN).
%
%   [p, sd, logit] = mpm_bn_predict(mdl, D)      D: N x n discrete states
%   logit = log P(C=1)/P(C=0) + sum_i log P(x_i | C=1, x_pi(i)) / P(x_i | C=0, x_pi(i))
%   p = sigmoid(logit); sd = standard deviation of p over the Dirichlet-posterior parameter draws
%   (0 if the model has no draws). With binary evidence and the naive structure this is the
%   weights-of-evidence posterior logit: logit = prior logit + sum of W+ / W- (Bonham-Carter 1994).
D = double(D); N = size(D, 1);
logit = mdl.logprior * ones(N, 1);
S = mdl.draws; LD = [];
if nargout > 1 && S > 0, LD = repmat(logit, 1, S); end
for i = find(mdl.use)
    pa = mdl.parent(i); xp = ones(N, 1);
    if pa > 0, xp = D(:, pa); end
    [ri, rp] = size(mdl.L{i});
    k = D(:, i) + (xp - 1) * ri;                                     % linear index into the r_i x r_p table
    logit = logit + mdl.L{i}(k);
    if ~isempty(LD)
        T = reshape(mdl.Ld{i}, ri * rp, S);
        LD = LD + T(k, :);
    end
end
p = 1 ./ (1 + exp(-logit));
sd = zeros(N, 1);
if ~isempty(LD), sd = std(1 ./ (1 + exp(-LD)), 0, 2); end
end
