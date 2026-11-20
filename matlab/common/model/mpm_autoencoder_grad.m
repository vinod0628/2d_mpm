function [J, gW, gb] = mpm_autoencoder_grad(W, b, act, X, l2)
%MPM_AUTOENCODER_GRAD Loss and back-propagated gradients of an autoencoder on a batch X.
%
%   J = 1/(B P) sum ||x_hat - x||^2 + l2/2 sum ||W||^2   (see MPM_AUTOENCODER_TRAIN)
nl = numel(W); [B, P] = size(X);
A = cell(1, nl + 1); A{1} = X;
for l = 1:nl
    A{l + 1} = f(A{l} * W{l} + b{l}, act{l});
end
E = A{nl + 1} - X;
J = sum(E(:) .^ 2) / (B * P) + l2 / 2 * sum(cellfun(@(w) sum(w(:) .^ 2), W));
if nargout < 2, return; end
gW = cell(1, nl); gb = cell(1, nl);
dA = 2 * E / (B * P);
for l = nl:-1:1
    dZ = dA .* df(A{l + 1}, act{l});
    gW{l} = A{l}' * dZ + l2 * W{l};
    gb{l} = sum(dZ, 1);
    if l > 1, dA = dZ * W{l}'; end
end
end

function a = f(z, name)
switch name
    case 'linear',  a = z;
    case 'tanh',    a = tanh(z);
    case 'relu',    a = max(z, 0);
    case 'sigmoid', a = 1 ./ (1 + exp(-z));
    otherwise, error('mpm:autoencoder', 'Activation must be tanh, relu or sigmoid.');
end
end

function d = df(a, name)          % derivative expressed through the activation output a
switch name
    case 'linear',  d = 1;
    case 'tanh',    d = 1 - a .^ 2;
    case 'relu',    d = double(a > 0);
    case 'sigmoid', d = a .* (1 - a);
end
end
