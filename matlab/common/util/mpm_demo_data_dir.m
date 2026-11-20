function d = mpm_demo_data_dir(root, dataset)
%MPM_DEMO_DATA_DIR Folder of a pre-processed demo dataset (layers.csv + rasters/).
%
%   d = mpm_demo_data_dir(root, 'nevada_carlin_au')     root = matlab/ folder
%
%   Looks in  <root>/../demo_data/<dataset>/processed  and the older layout
%   <root>/../demo_data/2d_mpm/<dataset>/processed; returns the first that has layers.csv
%   (or the first candidate, so the loader reports a clear "layers.csv not found").
cand = {fullfile(root, '..', 'demo_data', dataset, 'processed'), ...
        fullfile(root, '..', 'demo_data', '2d_mpm', dataset, 'processed')};
d = cand{1};
for i = 1:numel(cand)
    if isfile(fullfile(cand{i}, 'layers.csv')), d = cand{i}; return; end
end
end
