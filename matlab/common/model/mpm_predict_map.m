function [scoreMap, uncMap] = mpm_predict_map(M, mdl, F, scaler, chunk, log)
%MPM_PREDICT_MAP Apply a trained model to every valid cell, in chunks (bounded memory).
%
%   [scoreMap, uncMap] = mpm_predict_map(M, mdl, F, scaler, chunk, log)
%   scoreMap  H x W single, NaN outside the modelling mask
%   uncMap    H x W single uncertainty if M.has_uncertainty (predict returns [y, unc]), else []
if nargin < 5 || isempty(chunk), chunk = 200000; end
if nargin < 6, log = mpm_Logger(true); end
n = size(F.X, 1);
if isfield(M, 'predict_map') && ~isempty(M.predict_map)      % method scores the whole map itself
    [p, q] = M.predict_map(mdl, F);                           % (e.g. fully convolutional CNN)
    scoreMap = nan(size(F.valid), 'single'); scoreMap(F.idx) = p;
    uncMap = [];
    if ~isempty(q), uncMap = nan(size(F.valid), 'single'); uncMap(F.idx) = q; end
    return;
end
s = nan(n, 1, 'single'); u = [];
nb = ceil(n / chunk);
for b = 1:nb
    i = (b - 1) * chunk + 1 : min(b * chunk, n);
    Z = scaler.apply(F.X(i, :));
    if isfield(M, 'has_uncertainty') && M.has_uncertainty
        [p, q] = M.predict(mdl, Z);
    else
        p = M.predict(mdl, Z); q = [];
    end
    s(i) = p;
    if ~isempty(q)
        if isempty(u), u = nan(n, 1, 'single'); end
        u(i) = q;
    end
    if nb > 1, log.add('  map prediction %d/%d', b, nb); end
end
scoreMap = nan(size(F.valid), 'single'); scoreMap(F.idx) = s;
uncMap = [];
if ~isempty(u), uncMap = nan(size(F.valid), 'single'); uncMap(F.idx) = u; end
end
