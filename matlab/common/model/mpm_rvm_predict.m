function [m, v] = mpm_rvm_predict(mdl, X)
%MPM_RVM_PREDICT Posterior predictive mean and variance of an RVM (MPM_RVM_TRAIN).
%
%   [m, v] = mpm_rvm_predict(mdl, X)
%   m  N x 1 predictive mean of the latent function phi(x)' mu
%   v  N x 1 variance phi(x)' Sigma phi(x) of the latent function (weight uncertainty only;
%      for regression add 1/beta for the noise, for classification use it to moderate the
%      sigmoid - see reg_rvm / clf_rvm)
Phi = mpm_rvm_basis(X, mdl.centres, mdl.opts.kernel, mdl.opts.kernel_scale, mdl.bias);
Phi = Phi ./ mdl.scales';
m = Phi * mdl.mu;
if nargout > 1
    v = max(sum((Phi * mdl.Sigma) .* Phi, 2), 0);
end
end
