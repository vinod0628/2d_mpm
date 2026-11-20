function root = mpm_setup()
%MPM_SETUP Put the MATLAB MPM code on the path and return the matlab/ folder.
%
%   root = mpm_setup();
%
%   Every run_*.m script calls this first, so running a script from any current folder works.
%   Folder layout (see README.md):
%     common/          shared I/O, features, sampling, evaluation, GUI (used by every method)
%     classification/  prospectivity / target delineation  (deposit vs. background)
%     regression/      grade estimation from orebody samples + evidence layers
%     data/            point CSVs exported from the Python-processed GeoPackages
%     results/         outputs (result.mat, GeoTIFFs, CSV tables, PNG figures)

root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'common')));
addpath(fullfile(root, 'classification'), fullfile(root, 'classification', 'methods'));
addpath(fullfile(root, 'regression'), fullfile(root, 'regression', 'methods'));
end
