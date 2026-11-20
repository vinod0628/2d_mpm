function varargout = mpm_gb_loss(what, loss, y, F, w)
%MPM_GB_LOSS Losses of the gradient-boosting engines (first and second derivatives).
%
%   [g, h] = mpm_gb_loss('grad', loss, y, F, w)   gradient / hessian of the loss at raw score F
%   F0     = mpm_gb_loss('base', loss, y, [], w)  initial constant score (boost from average)
%   p      = mpm_gb_loss('link', loss, [], F)     response: sigmoid(F) (logistic) or F (squared)
%
%   loss 'logistic': binary cross-entropy, y in {0,1}:  g = p - y,  h = p (1 - p)
%   loss 'squared' : 1/2 (y - F)^2:                    g = F - y,  h = 1
%   w are instance weights (class balancing); g and h are multiplied by w.
switch what
    case 'grad'
        switch loss
            case 'logistic'
                p = 1 ./ (1 + exp(-F));
                varargout = {(p - y) .* w, max(p .* (1 - p), 1e-16) .* w};
            case 'squared'
                varargout = {(F - y) .* w, w};
        end
    case 'base'
        switch loss
            case 'logistic'
                p = min(max(sum(w .* y) / sum(w), 1e-6), 1 - 1e-6);
                varargout = {log(p / (1 - p))};
            case 'squared'
                varargout = {sum(w .* y) / sum(w)};
        end
    case 'link'
        if strcmp(loss, 'logistic'), varargout = {1 ./ (1 + exp(-F))}; else, varargout = {F}; end
end
end
