function isTest = mpm_split_points(P, split, rs)
%MPM_SPLIT_POINTS Choose test points (deposits or grade samples) for independent validation.
%
%   isTest = mpm_split_points(P, split, rs)
%
%   P      point table with x, y (rows = points that fall in valid cells)
%   split  struct:
%     .method         'random' | 'block' | 'rule' | 'none'
%     .test_fraction  fraction of points held out (random / block), e.g. 0.3
%     .block_size     block edge (map units, m) for 'block': whole blocks go to the test set,
%                     which limits spatial autocorrelation between train and test
%     .test_rule      function handle on the table for 'rule', e.g. @(T) T.RANDOM >= 700
%   rs     RandStream

n = height(P);
isTest = false(n, 1);
switch lower(split.method)
    case 'none'
    case 'random'
        k = round(split.test_fraction * n);
        if split.test_fraction > 0, k = max(1, min(n - 1, k)); end
        isTest(randperm(rs, n, k)) = true;
    case 'block'
        b = split.block_size;
        key = floor(P.x / b) * 1e6 + floor(P.y / b);
        [blocks, ~, g] = unique(key);
        order = randperm(rs, numel(blocks));
        target = split.test_fraction * n;
        for i = order
            if sum(isTest) >= target, break; end
            if sum(isTest | g == i) >= n, continue; end     % keep at least one training block
            isTest(g == i) = true;
        end
    case 'rule'
        isTest = logical(split.test_rule(P));
        isTest = isTest(:);
    otherwise
        error('mpm:split', 'Unknown split method "%s".', split.method);
end
if all(isTest), error('mpm:split', 'The split leaves no training points.'); end
end
