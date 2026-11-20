function [lin, rows, cols, inside] = mpm_points_to_cells(R, x, y, valid)
%MPM_POINTS_TO_CELLS Map world coordinates to raster cells.
%
%   [lin, rows, cols, inside] = mpm_points_to_cells(R, x, y, valid)
%
%   lin     linear cell index (NaN where the point is outside the grid or VALID is false)
%   inside  true where the point falls in a valid cell
%   valid   optional H x W logical (e.g. the modelling mask)
[rows, cols] = worldToDiscrete(R, x(:), y(:));
inside = ~isnan(rows) & ~isnan(cols);
lin = nan(numel(x), 1);
lin(inside) = sub2ind(R.RasterSize, rows(inside), cols(inside));
if nargin > 3 && ~isempty(valid)
    ok = inside;
    ok(inside) = valid(lin(inside));
    lin(~ok) = NaN;
    inside = ok;
end
end
