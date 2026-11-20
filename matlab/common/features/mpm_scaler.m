function sc = mpm_scaler(X)
%MPM_SCALER Z-score scaler fitted on training samples.
%   sc = mpm_scaler(Xtrain);  Z = sc.apply(X);
%   Constant columns get sigma = 1 so they map to 0 instead of NaN.
mu = mean(X, 1, 'omitnan');
sg = std(double(X), 0, 1, 'omitnan');
sg(~isfinite(sg) | sg < 1e-12) = 1;
sc.mu = single(mu);
sc.sigma = single(sg);
sc.apply = @(Z) (single(Z) - sc.mu) ./ sc.sigma;
end
