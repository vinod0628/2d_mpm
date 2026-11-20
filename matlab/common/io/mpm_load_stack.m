function S = mpm_load_stack(dataDir, include, exclude, maskLayer)
%MPM_LOAD_STACK Load co-registered evidence rasters produced by the Python pre-processing.
%
%   S = mpm_load_stack(dataDir, include, exclude, maskLayer)
%
%   dataDir    a "processed" folder containing layers.csv and rasters/*.tif (all on one grid)
%   include    cellstr of layer names or wildcards ('mag_*'); {} = every raster layer whose group
%              is not 'mask' or 'reference'
%   exclude    cellstr of names / wildcards removed after INCLUDE (e.g. {'geochem_*'})
%   maskLayer  name of the study-area mask layer ('' = whole grid). Cells with mask 0/nodata
%              are outside the study area.
%
%   S fields: names, kinds ('continuous'|'categorical'), groups, descriptions, files,
%             data (H x W x F single, NaN = nodata), mask (H x W logical), R (MapCellsReference),
%             geokeys (GeoKeyDirectoryTag, used to write GeoTIFFs in the same CRS),
%             classes (1 x F cell: class table or []), x, y (cell-centre coordinates), dataDir.

if nargin < 2, include = {}; end
if nargin < 3, exclude = {}; end
if nargin < 4, maskLayer = 'study_mask'; end
include = cellstr(include); exclude = cellstr(exclude);
if isequal(include, {''}), include = {}; end
if isequal(exclude, {''}), exclude = {}; end

csv = fullfile(dataDir, 'layers.csv');
if ~isfile(csv)
    error('mpm:io', 'layers.csv not found in %s (run the Python pre-processing first).', dataDir);
end
T = readtable(csv, 'TextType', 'string', 'Delimiter', ',');
T = T(endsWith(lower(T.file), '.tif'), :);

if isempty(include)
    sel = ~ismember(lower(T.group), ["mask", "reference"]);
else
    sel = match_any(T.name, include);
    missing = include(~cellfun(@(p) any(match_any(T.name, {p})), include));
    if ~isempty(missing)
        error('mpm:io', 'Layer(s) not found in %s: %s', csv, strjoin(missing, ', '));
    end
end
if ~isempty(exclude), sel = sel & ~match_any(T.name, exclude); end
T = T(sel, :);
if isempty(T), error('mpm:io', 'No evidence layers selected.'); end

S.dataDir = dataDir;
S.names = cellstr(T.name)';
S.kinds = cellstr(T.kind)';
S.groups = cellstr(T.group)';
S.descriptions = cellstr(T.description)';
S.files = cellfun(@(f) fullfile(dataDir, f), cellstr(T.file)', 'UniformOutput', false);
S.classes = cell(1, height(T));

[A, R] = read_one(S.files{1});               % Mapping Toolbox: grid reference + CRS
S.R = R;
gi = geotiffinfo(S.files{1});
S.geokeys = gi.GeoTIFFTags.GeoKeyDirectoryTag;
geo0 = grid_signature(imfinfo(S.files{1}));
S.data = zeros([R.RasterSize, height(T)], 'single');
S.data(:, :, 1) = A;
for k = 2:height(T)
    [A, geo] = read_fast(S.files{k});          % plain TIFF read (~50x faster than readgeoraster)
    if isempty(geo), [A, Rk] = read_one(S.files{k}); geo = grid_signature_R(Rk); geo0r = grid_signature_R(R);
    else, geo0r = geo0; end
    if numel(geo) ~= numel(geo0r) || max(abs(geo - geo0r) ./ max(1, abs(geo0r))) > 1e-9
        error('mpm:io', ['%s is not on the grid of %s. All evidence layers must share one grid ', ...
            '(re-run the Python alignment).'], S.names{k}, S.names{1});
    end
    S.data(:, :, k) = A;
end
for k = 1:height(T)
    ct = regexprep(S.files{k}, '\.tif$', '.classes.csv');
    if strcmp(S.kinds{k}, 'categorical') && isfile(ct)
        S.classes{k} = readtable(ct, 'TextType', 'string', 'Delimiter', ',');
    end
end

S.mask = true(R.RasterSize);
S.maskLayer = '';
if ~isempty(maskLayer)
    mfile = fullfile(dataDir, 'rasters', [maskLayer '.tif']);
    if ~isfile(mfile)
        error('mpm:io', 'Mask layer %s not found (%s).', maskLayer, mfile);
    end
    [M, geo] = read_fast(mfile);
    if isempty(geo), M = read_one(mfile); end
    if ~isequal(size(M), R.RasterSize), error('mpm:io', 'Mask %s is not on the evidence grid.', maskLayer); end
    S.mask = ~isnan(M) & M ~= 0;
    S.maskLayer = maskLayer;
end
[S.x, S.y] = mpm_cell_centers(R);
end

function [A, R] = read_one(file)
[A, R] = readgeoraster(file);
info = georasterinfo(file);
nd = info.MissingDataIndicator;
A = single(A);
if ~isempty(nd), A(A == single(nd)) = NaN; end
A(~isfinite(A)) = NaN;
end

function [A, geo] = read_fast(file)
% Single-band GeoTIFF via the Tiff class; nodata from the GDAL_NODATA tag.
% Returns geo = [] when the file is not a plain north-up single-band GeoTIFF (caller falls back).
geo = []; A = [];
try
    ii = imfinfo(file);
    if numel(ii) ~= 1 || ii.SamplesPerPixel ~= 1 || ~isfield(ii, 'ModelTiepointTag'), return; end
    ws = warning('off', 'all');
    c = onCleanup(@() warning(ws));
    t = Tiff(file, 'r'); A = single(read(t)); close(t);
    if isfield(ii, 'GDAL_NODATA') && ~isempty(ii.GDAL_NODATA)
        nd = str2double(strtrim(char(ii.GDAL_NODATA)));
        if ~isnan(nd), A(A == single(nd)) = NaN; end
    end
    A(~isfinite(A)) = NaN;
    geo = grid_signature(ii);
catch
    geo = []; A = [];
end
end

function g = grid_signature(ii)
% [ncols nrows cellX cellY left top] from the GeoTIFF tags
g = [double(ii.Width), double(ii.Height), ii.ModelPixelScaleTag(1), ii.ModelPixelScaleTag(2), ...
    ii.ModelTiepointTag(4) - ii.ModelTiepointTag(1) * ii.ModelPixelScaleTag(1), ...
    ii.ModelTiepointTag(5) + ii.ModelTiepointTag(2) * ii.ModelPixelScaleTag(2)];
end

function g = grid_signature_R(R)
g = [R.RasterSize(2), R.RasterSize(1), R.CellExtentInWorldX, R.CellExtentInWorldY, ...
    R.XWorldLimits(1), R.YWorldLimits(2)];
end

function tf = match_any(names, patterns)
tf = false(size(names));
for i = 1:numel(patterns)
    rx = ['^' regexptranslate('wildcard', patterns{i}) '$'];
    tf = tf | ~cellfun(@isempty, regexpi(cellstr(names), rx, 'once'));
end
end
