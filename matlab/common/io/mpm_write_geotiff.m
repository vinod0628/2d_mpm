function mpm_write_geotiff(file, A, R, geokeys)
%MPM_WRITE_GEOTIFF Write a raster on the evidence grid, in the CRS of the source rasters.
%
%   mpm_write_geotiff(file, A, R, geokeys)
%
%   geokeys is the GeoKeyDirectoryTag of an input raster (S.geokeys from MPM_LOAD_STACK), so the
%   output carries the same CRS even when it has no EPSG code (e.g. the Carlin LCC grid).
%   Float rasters use NaN outside the study area; integer rasters use 0.
folder = fileparts(file);
if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
geotiffwrite(file, A, R, 'GeoKeyDirectoryTag', geokeys);
end
