function F = mpm_build_features(S, opts)
%MPM_BUILD_FEATURES Turn an evidence stack into a feature matrix for every valid cell.
%
%   F = mpm_build_features(S, opts)
%
%   opts.categorical  'onehot' (default) | 'numeric'   how categorical layers enter the model
%   opts.as_numeric   cellstr of categorical layers to use as numbers anyway (e.g. classified
%                     geophysics whose class values are physical values: aeromag -400/-200/0 nT)
%   opts.missing      'impute' (default: continuous -> median, one-hot -> all zeros) | 'exclude'
%                     (drop cells where any selected layer is nodata)
%   opts.max_classes  error if a one-hot layer has more classes than this (default 60)
%
%   F fields:
%     X        Nvalid x P single feature matrix (unscaled)
%     idx      Nvalid x 1 linear indices of the valid cells in the grid
%     pos      H x W int32 lookup: grid cell -> row of X (0 = not a modelling cell)
%     valid    H x W logical modelling mask
%     names    1 x P feature names ('geology=Q' for one-hot columns)
%     source   1 x P index of the layer each column comes from (permutation importance groups)
%     layers   1 x F layer names;  impute 1 x P values used for missing data

if nargin < 2, opts = struct(); end
opts = mpm_merge(struct('categorical', 'onehot', 'as_numeric', {{}}, 'missing', 'impute', ...
    'max_classes', 60), opts);

[H, W, L] = size(S.data);
anyData = any(~isnan(S.data), 3);
valid = S.mask & anyData;
if strcmp(opts.missing, 'exclude')
    valid = valid & all(~isnan(S.data), 3);
end
idx = find(valid);
n = numel(idx);

cols = {}; names = {}; source = []; impute = [];
for k = 1:L
    v = S.data(:, :, k); v = v(idx);
    isCat = strcmp(S.kinds{k}, 'categorical') && strcmp(opts.categorical, 'onehot') ...
        && ~ismember(S.names{k}, cellstr(opts.as_numeric));
    if isCat
        cls = unique(v(~isnan(v)));
        if numel(cls) > opts.max_classes
            error('mpm:features', '%s has %d classes (> max_classes=%d); use it as numeric or reclassify.', ...
                S.names{k}, numel(cls), opts.max_classes);
        end
        for c = cls'
            cols{end+1} = single(v == c); %#ok<AGROW>
            names{end+1} = sprintf('%s=%s', S.names{k}, class_label(S.classes{k}, c)); %#ok<AGROW>
            source(end+1) = k; impute(end+1) = 0; %#ok<AGROW>
        end
    else
        med = median(v(~isnan(v)));
        if isempty(med) || isnan(med), med = 0; end
        v(isnan(v)) = med;
        cols{end+1} = v; names{end+1} = S.names{k}; %#ok<AGROW>
        source(end+1) = k; impute(end+1) = med; %#ok<AGROW>
    end
end

F.X = zeros(n, numel(cols), 'single');
for j = 1:numel(cols), F.X(:, j) = cols{j}; end
F.idx = idx;
F.pos = zeros(H, W, 'int32');
F.pos(idx) = int32(1:n);
F.valid = valid;
F.names = names;
F.source = source;
F.layers = S.names;
F.impute = impute;
F.opts = opts;
end

function s = class_label(T, value)
s = num2str(value);
if isempty(T) || ~ismember('value', T.Properties.VariableNames), return; end
i = find(double(T.value) == double(value), 1);
if ~isempty(i) && ismember('label', T.Properties.VariableNames)
    s = char(string(T.label(i)));
end
end
