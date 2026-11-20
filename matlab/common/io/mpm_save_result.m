function res = mpm_save_result(res, log)
%MPM_SAVE_RESULT Write a classification / regression result to its output folder.
%
%   Folder: cfg.output_dir, default  matlab/results/<dataset>/<task>_<method>_<yyyymmdd_HHMMSS>/
%     result.mat            the full result struct (reload with mpm_results_viewer('.../result.mat'))
%     <score>.tif           prospectivity (classification) or predicted grade (regression)
%     targets.tif           target classes (classification), uncertainty.tif (regression, if any)
%     metrics.csv, importance.csv, tuning.csv, targets.csv / samples.csv, log.txt
if nargin < 2, log = mpm_Logger(true); end
res.output_dir = '';
cfg = res.cfg;
if ~cfg.save && ~cfg.write_geotiff, return; end
out = cfg.output_dir;
if isempty(out)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));     % matlab/
    out = fullfile(root, 'results', res.dataset, sprintf('%s_%s_%s', res.task, res.method.key, ...
        char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
end
if ~isfolder(out), mkdir(out); end
res.output_dir = out;

if cfg.write_geotiff
    R = res.grid.R; gk = res.grid.geokeys;
    switch res.task
        case 'classification'
            mpm_write_geotiff(fullfile(out, 'prospectivity.tif'), res.map.score, R, gk);
            mpm_write_geotiff(fullfile(out, 'targets.tif'), res.map.targets, R, gk);
            if isfield(res.map, 'uncertainty') && ~isempty(res.map.uncertainty)
                mpm_write_geotiff(fullfile(out, 'uncertainty.tif'), res.map.uncertainty, R, gk);
            end
        case 'regression'
            mpm_write_geotiff(fullfile(out, 'predicted_grade.tif'), res.map.grade, R, gk);
            if ~isempty(res.map.uncertainty)
                mpm_write_geotiff(fullfile(out, 'uncertainty.tif'), res.map.uncertainty, R, gk);
            end
    end
    if isfield(res, 'binary') && ~isempty(res.binary)          % 1 target / 0 non-target / NaN outside
        f = fieldnames(res.binary.maps);
        for i = 1:numel(f)
            mpm_write_geotiff(fullfile(out, sprintf('binary_%s.tif', f{i})), res.binary.maps.(f{i}), R, gk);
        end
    end
end

if cfg.save
    writetable(mpm_metrics_table(res), fullfile(out, 'metrics.csv'));
    if ~isempty(res.importance), writetable(res.importance, fullfile(out, 'importance.csv')); end
    if ~isempty(res.native_importance), writetable(res.native_importance, fullfile(out, 'importance_native.csv')); end
    if ~isempty(res.tuning), writetable(res.tuning, fullfile(out, 'tuning.csv')); end
    if isfield(res, 'binary') && ~isempty(res.binary)
        if ~isempty(res.binary.table), writetable(res.binary.table, fullfile(out, 'binary_targets.csv')); end
        if ~isempty(res.binary.ca), writetable(res.binary.ca.segments, fullfile(out, 'ca_fractal_segments.csv')); end
    end
    switch res.task
        case 'classification'
            writetable(res.targets, fullfile(out, 'targets.csv'));
            D = res.deposits; D.score = double(res.map.score(D.cell));
            D.target_class = double(res.map.targets(D.cell));
            writetable(D, fullfile(out, 'deposits_scores.csv'));
        case 'regression'
            writetable(res.samples_table, fullfile(out, 'samples.csv'));
    end
    log.add('saved to %s', out);
    res.log = log.lines;
    fid = fopen(fullfile(out, 'log.txt'), 'w');
    fprintf(fid, '%s\n', res.log{:});
    fclose(fid);
    save(fullfile(out, 'result.mat'), 'res', '-v7.3');
end
end
