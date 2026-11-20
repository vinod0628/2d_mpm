function t = mpm_param_text(p, f)
%MPM_PARAM_TEXT One-line text of selected fields F of a parameter struct P.
parts = cell(1, numel(f));
for i = 1:numel(f)
    v = p.(f{i}); if isnumeric(v) || islogical(v), v = mat2str(v); end
    parts{i} = sprintf('%s=%s', f{i}, v);
end
t = strjoin(parts, ', ');
end
