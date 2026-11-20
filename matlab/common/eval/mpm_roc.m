function C = mpm_roc(y, s, maxPoints)
%MPM_ROC ROC curve and AUC (ties handled exactly, AUC = Mann-Whitney U / (P*N)).
%
%   C = mpm_roc(y, s)          y logical labels (true = deposit / positive), s scores (high = positive)
%   C = mpm_roc(y, s, maxPts)  thin the stored curve to about maxPts points (default 5000)
%
%   C fields: fpr, tpr, thr (column vectors, from (0,0) to (1,1)), auc, npos, nneg,
%             youden_thr (threshold maximising TPR - FPR)
if nargin < 3, maxPoints = 5000; end
y = logical(y(:)); s = double(s(:));
ok = isfinite(s); y = y(ok); s = s(ok);
P = sum(y); N = sum(~y);
C = struct('fpr', [0; 1], 'tpr', [0; 1], 'thr', [Inf; -Inf], 'auc', NaN, 'npos', P, 'nneg', N, 'youden_thr', NaN);
if P == 0 || N == 0, return; end
[ss, o] = sort(s, 'descend'); yy = y(o);
tp = cumsum(yy); fp = cumsum(~yy);
last = [diff(ss) ~= 0; true];
tpr = [0; tp(last) / P]; fpr = [0; fp(last) / N]; thr = [Inf; ss(last)];
C.auc = trapz(fpr, tpr);
[~, j] = max(tpr - fpr); C.youden_thr = thr(j);
if numel(fpr) > maxPoints
    keep = unique([1; round(linspace(1, numel(fpr), maxPoints))'; numel(fpr)]);
    fpr = fpr(keep); tpr = tpr(keep); thr = thr(keep);
end
C.fpr = fpr; C.tpr = tpr; C.thr = thr;
end
