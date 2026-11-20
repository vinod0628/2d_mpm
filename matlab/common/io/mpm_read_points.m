function P = mpm_read_points(file, xField, yField)
%MPM_READ_POINTS Read a point table (deposits, orebody / drillhole samples) from CSV or shapefile.
%
%   P = mpm_read_points(file)            CSV: columns 'x' and 'y';  .shp: point geometry
%   P = mpm_read_points(file, xF, yF)    CSV with other coordinate columns (case-insensitive)
%
%   Coordinates must be in the CRS of the evidence rasters. Point files come from the data
%   pre-processing (Python, QGIS, ArcGIS, ...): a CSV with x / y columns (QGIS: "Export > Save
%   Features As > CSV, GEOMETRY = AS_XY") or an ESRI point shapefile. The MATLAB tool does not
%   need Python. Returns a table with all attribute columns plus numeric columns x, y.

if nargin < 2 || isempty(xField), xField = 'x'; end
if nargin < 3 || isempty(yField), yField = 'y'; end
if ~isfile(file)
    error('mpm:io', ['Point file not found: %s\nProvide deposits / samples as a CSV with x, y columns ', ...
        'or as a point shapefile (raster CRS).'], file);
end
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.shp')
    S = shaperead(file);
    if isempty(S), error('mpm:io', '%s contains no features.', file); end
    if ~all(strcmpi({S.Geometry}, 'Point')), error('mpm:io', '%s is not a point shapefile.', file); end
    x = [S.X]'; y = [S.Y]';
    S = rmfield(S, intersect(fieldnames(S), {'Geometry', 'X', 'Y', 'BoundingBox'}));
    if isempty(fieldnames(S)), A = table(); else, A = struct2table(S(:), 'AsArray', true); end
    for v = A.Properties.VariableNames
        if iscellstr(A.(v{1})) || ischar(A.(v{1})), A.(v{1}) = string(A.(v{1})); end
    end
    P = [table(x, y), A];
else
    opts = detectImportOptions(file, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    P = readtable(file, opts);
    vn = P.Properties.VariableNames;
    ix = find(strcmpi(vn, xField), 1); iy = find(strcmpi(vn, yField), 1);
    if isempty(ix) || isempty(iy)
        error('mpm:io', '%s needs coordinate columns "%s" and "%s".', file, xField, yField);
    end
    x = double(P.(vn{ix})); y = double(P.(vn{iy}));
    P.(vn{ix}) = []; vn = P.Properties.VariableNames; iy = find(strcmpi(vn, yField), 1);
    P.(vn{iy}) = [];
    P = addvars(P, x, y, 'Before', 1, 'NewVariableNames', {'x', 'y'});
end
P = P(isfinite(P.x) & isfinite(P.y), :);
end
