function s = mpm_merge(defaults, user)
%MPM_MERGE Recursively overlay the fields of USER on DEFAULTS (both structs).
%   Nested structs are merged field by field; any other value in USER replaces the default.
%   Fields that exist only in USER are added (so methods can take extra parameters).
s = defaults;
if isempty(user), return; end
f = fieldnames(user);
for i = 1:numel(f)
    k = f{i};
    if isfield(s, k) && isstruct(s.(k)) && isscalar(s.(k)) && isstruct(user.(k)) && isscalar(user.(k))
        s.(k) = mpm_merge(s.(k), user.(k));
    else
        s.(k) = user.(k);
    end
end
end
