function H = mpm_elm_hidden(mdl, X)
%MPM_ELM_HIDDEN Hidden layer output matrix of an ELM,  H(j, i) = g(w_i . x_j + b_i).
%
%   H = mpm_elm_hidden(mdl, X)    mdl from MPM_ELM_TRAIN (fields W, b, activation)
%
%   g: 'sigmoid' 1/(1+exp(-z)) | 'tanh' | 'sine' sin(z) | 'hardlim' (z >= 0) |
%      'tribas' max(1-|z|, 0) | 'radbas' exp(-z^2)
Z = double(X) * mdl.W + mdl.b;
switch mdl.activation
    case {'sigmoid', 'sig'}, H = 1 ./ (1 + exp(-Z));
    case 'tanh',             H = tanh(Z);
    case {'sine', 'sin'},    H = sin(Z);
    case 'hardlim',          H = double(Z >= 0);
    case 'tribas',           H = max(1 - abs(Z), 0);
    case 'radbas',           H = exp(-Z .^ 2);
    otherwise
        error('mpm:elm', 'Unknown activation "%s" (sigmoid|tanh|sine|hardlim|tribas|radbas).', mdl.activation);
end
end
