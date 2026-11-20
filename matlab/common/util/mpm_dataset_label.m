function s = mpm_dataset_label(d)
%MPM_DATASET_LABEL Dataset name from its folder: '.../nevada_carlin_au/processed' -> 'nevada_carlin_au'.
[p, n] = fileparts(char(d));
if strcmpi(n, 'processed'), [~, n] = fileparts(p); end
s = n;
end
