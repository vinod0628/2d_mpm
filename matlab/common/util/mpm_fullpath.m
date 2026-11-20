function p = mpm_fullpath(p)
%MPM_FULLPATH Absolute, normalised path ('a\b\..\c' -> 'a\c'); empty stays empty.
if isempty(p), return; end
p = char(p);
try
    p = char(java.io.File(p).getCanonicalPath());
catch
    if ~(numel(p) > 1 && (p(2) == ':' || startsWith(p, filesep))), p = fullfile(pwd, p); end
end
end
