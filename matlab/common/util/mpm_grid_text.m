function t = mpm_grid_text(g)
%MPM_GRID_TEXT One-line text of a tuning grid struct, e.g. 'C=[0.1 1 10], KernelScale=[0.5 1]'.
f = fieldnames(g); parts = cell(1, numel(f));
for i = 1:numel(f)
    v = g.(f{i});
    if iscell(v)
        v = ['{' strjoin(cellfun(@one, v, 'UniformOutput', false), ', ') '}'];
    elseif isstring(v)
        v = ['{' strjoin(cellstr(v), ', ') '}'];
    else
        v = mat2str(v);
    end
    parts{i} = sprintf('%s=%s', f{i}, v);
end
t = strjoin(parts, ', ');
end

function s = one(e)
if isnumeric(e) || islogical(e), s = mat2str(e); else, s = char(string(e)); end
end
