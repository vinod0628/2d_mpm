function [x, y] = mpm_cell_centers(R)
%MPM_CELL_CENTERS Cell-centre coordinate vectors of a north-up MapCellsReference.
%   x: 1 x W (west -> east), y: H x 1 (north -> south), i.e. y(1) is the first raster row.
dx = R.CellExtentInWorldX; dy = R.CellExtentInWorldY;
x = R.XWorldLimits(1) + dx * ((1:R.RasterSize(2)) - 0.5);
if strcmp(R.ColumnsStartFrom, 'north')
    y = R.YWorldLimits(2) - dy * ((1:R.RasterSize(1))' - 0.5);
else
    y = R.YWorldLimits(1) + dy * ((1:R.RasterSize(1))' - 0.5);
end
end
