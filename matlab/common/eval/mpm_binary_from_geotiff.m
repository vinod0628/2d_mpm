function B = mpm_binary_from_geotiff(scoreTif, pointsCsv, opts)
%MPM_BINARY_FROM_GEOTIFF Binary target grid (1 = predicted deposit area, 0 = non-deposit) for ANY
%   prospectivity GeoTIFF - MATLAB results, Python results (e.g. a WofE posterior), or a
%   knowledge-driven map - by P-A intersection and/or C-A fractal.
%
%   B = mpm_binary_from_geotiff(scoreTif)                       C-A only (no deposits needed)
%   B = mpm_binary_from_geotiff(scoreTif, depositsCsv)          P-A and C-A
%   B = mpm_binary_from_geotiff(scoreTif, depositsCsv, opts)
%
%   depositsCsv  CSV with x, y in the raster CRS (e.g. matlab/data/<dataset>/deposits.csv)
%   opts         as MPM_BINARY_TARGETS, plus
%     .mask_tif     optional study-area mask (cells with 0 / nodata excluded)
%     .out_dir      where to write <name>_binary_pa.tif / _binary_ca.tif / _binary.csv
%                   (default: next to scoreTif; '' with .write = false writes nothing)
%     .write        true (default)
%     .show         true: open the binary-target viewer (default: when a desktop is available)
%     .test_rule    @(T) logical, deposits used as TEST (P-A uses the rest)
%
%   Output GeoTIFFs: single, 1 = target, 0 = non-target, NaN outside the study area; same grid
%   and CRS as scoreTif.
if nargin < 2, pointsCsv = ''; end
if nargin < 3, opts = struct(); end
opts = mpm_merge(struct('mask_tif', '', 'out_dir', [], 'write', true, 'show', usejava('desktop'), ...
    'test_rule', []), opts);
[A, R] = readgeoraster(scoreTif);
info = georasterinfo(scoreTif);
A = double(A);
if ~isempty(info.MissingDataIndicator), A(A == info.MissingDataIndicator) = NaN; end
valid = isfinite(A);
if ~isempty(opts.mask_tif)
    [Mk, Rm] = readgeoraster(opts.mask_tif);
    if ~isequal(Rm.RasterSize, R.RasterSize), error('mpm:binary', 'Mask is not on the score grid.'); end
    mi = georasterinfo(opts.mask_tif); Mk = double(Mk);
    if ~isempty(mi.MissingDataIndicator), Mk(Mk == mi.MissingDataIndicator) = NaN; end
    valid = valid & isfinite(Mk) & Mk ~= 0;
end
trLin = []; teLin = []; P = table();
if ~isempty(pointsCsv)
    P = mpm_read_points(pointsCsv);
    [lin, ~, ~, inside] = mpm_points_to_cells(R, P.x, P.y, valid);
    P = P(inside, :); lin = lin(inside);
    [lin, k] = unique(lin, 'stable'); P = P(k, :);
    P.test = false(height(P), 1);
    if ~isempty(opts.test_rule), P.test = logical(opts.test_rule(P)); end
    trLin = lin(~P.test); teLin = lin(P.test);
end
methods = {'ca'}; if ~isempty(trLin), methods = {'pa', 'ca'}; end
o = mpm_merge(struct('methods', {methods}, 'cell_area_km2', R.CellExtentInWorldX * R.CellExtentInWorldY / 1e6), ...
    rmfield(opts, {'mask_tif', 'out_dir', 'write', 'show', 'test_rule'}));
B = mpm_binary_targets(A, valid, trLin, teLin, o);
[x, y] = mpm_cell_centers(R);
B.grid = struct('R', R, 'x', x, 'y', y);
B.points = P; B.source = scoreTif;
for i = 1:numel(B.messages), fprintf('%s\n', B.messages{i}); end
if ~isempty(B.table), disp(B.table(:, {'method', 'threshold', 'area_pct', 'train_captured_pct', 'test_captured_pct', 'train_efficiency'})); end

if opts.write
    out = opts.out_dir; if isempty(out), out = fileparts(scoreTif); end
    if ~isfolder(out), mkdir(out); end
    [~, name] = fileparts(scoreTif);
    gi = geotiffinfo(scoreTif);
    f = fieldnames(B.maps);
    for i = 1:numel(f)
        file = fullfile(out, sprintf('%s_binary_%s.tif', name, f{i}));
        mpm_write_geotiff(file, B.maps.(f{i}), R, gi.GeoTIFFTags.GeoKeyDirectoryTag);
        fprintf('wrote %s\n', file);
    end
    if ~isempty(B.table), writetable(B.table, fullfile(out, sprintf('%s_binary.csv', name))); end
    if ~isempty(B.ca), writetable(B.ca.segments, fullfile(out, sprintf('%s_ca_segments.csv', name))); end
end
if opts.show, mpm_binary_viewer(B, scoreTif); end
end
