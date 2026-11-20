%RUN_COMPARE_CLASSIFICATION  Run several classifiers on the same data / split and compare them.
%
%   Every method gets the same cfg and seed, so deposits split, negatives and folds are identical;
%   only the model differs. Each run is saved as usual; a comparison window shows test ROC/AUC
%   (samples and map), success/prediction-rate curves and a metric table. Add new methods to the
%   list as they are implemented (e.g. @clf_gbm, @clf_mlp).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';                         % 'carlin' | 'curnamona'
methods = {@clf_svm, @clf_rf, @clf_rvm, @clf_mlp, @clf_xgboost, @clf_lightgbm, @clf_elm, @clf_pubag, @clf_cnn, @clf_bn};   % <- methods to compare
% methods = {@clf_iforest, @clf_ocsvm, @clf_autoencoder};    % unsupervised anomaly detectors (M7)

cfg = clf_demo_config(dataset, root);
cfg.seed = 42;
cfg.show_gui = false;                       % one comparison window instead of one per method

results = cell(1, numel(methods));
for i = 1:numel(methods)
    results{i} = mpm_classification_run(cfg, methods{i}());
end
if usejava('desktop')
    mpm_compare_viewer(results);
else
    mpm_compare_viewer(results, 'Visible', 'off', 'Close', true, ...
        'ExportDir', fullfile(fileparts(results{end}.output_dir), 'comparison'));
end
