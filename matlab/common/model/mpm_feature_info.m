function out = mpm_feature_info(info)
%MPM_FEATURE_INFO Feature metadata of the current run, for methods that need to know which feature
%   columns belong to which evidence layer (e.g. clf_bn: one Bayesian-network node per layer).
%
%   mpm_feature_info(info)       set by the pipeline after the scaler is fitted
%   info = mpm_feature_info()    read by a method ([] when the method is used outside the pipeline)
%   info fields: names (1 x P), source (1 x P layer index), layers (1 x L), onehot (1 x P logical:
%   column is a one-hot class indicator 'layer=class'), mu, sigma (1 x P z-score parameters, so a
%   method can recover the raw values x = z .* sigma + mu).
persistent current
if nargin > 0, current = info; end
out = current;
end
