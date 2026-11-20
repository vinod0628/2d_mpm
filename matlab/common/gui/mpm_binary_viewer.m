function fig = mpm_binary_viewer(B, ttl, varargin)
%MPM_BINARY_VIEWER Window with the C-A plot, binary target maps and summary of MPM_BINARY_FROM_GEOTIFF.
%   fig = mpm_binary_viewer(B, title, 'Visible', 'off', 'ExportFile', 'x.png')
ip = inputParser; ip.addParameter('Visible', 'on'); ip.addParameter('ExportFile', '');
ip.parse(varargin{:}); o = ip.Results;
if nargin < 2, ttl = 'Binary targets'; end
fig = uifigure('Name', ['Binary targets - ' char(ttl)], 'Position', [100 80 1250 760], 'Visible', o.Visible);
mpm_binary_panel(uigridlayout(fig, [1 1]), B, B.grid.x, B.grid.y, B.points, 'deposits', 'score');
if ~isempty(o.ExportFile), drawnow; exportapp(fig, o.ExportFile); end
end
