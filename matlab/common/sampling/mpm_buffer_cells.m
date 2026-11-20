function [lin, group] = mpm_buffer_cells(R, centerLin, radiusCells, valid)
%MPM_BUFFER_CELLS Cells within RADIUSCELLS of each centre cell (positive-sample augmentation).
%
%   [lin, group] = mpm_buffer_cells(R, centerLin, r, valid)
%   lin    unique valid cells (each assigned to the nearest centre)
%   group  index of the centre each cell belongs to (keeps CV folds grouped by deposit)
%   r = 0 returns the centres themselves.
centerLin = centerLin(:);
if radiusCells <= 0
    lin = centerLin; group = (1:numel(centerLin))'; return;
end
[dr, dc] = meshgrid(-radiusCells:radiusCells);
keep = dr.^2 + dc.^2 <= radiusCells^2 + 1e-9;
dr = dr(keep); dc = dc(keep); d2 = dr.^2 + dc.^2;
[r0, c0] = ind2sub(R.RasterSize, centerLin);
rr = r0' + dr; cc = c0' + dc; gg = repmat(1:numel(centerLin), numel(dr), 1); dd = repmat(d2, 1, numel(centerLin));
ok = rr >= 1 & cc >= 1 & rr <= R.RasterSize(1) & cc <= R.RasterSize(2);
l = sub2ind(R.RasterSize, rr(ok), cc(ok)); g = gg(ok); d = dd(ok);
ok = valid(l); l = l(ok); g = g(ok); d = d(ok);
[~, o] = sortrows([l d]);            % nearest centre first for shared cells
l = l(o); g = g(o);
[lin, first] = unique(l, 'stable');
group = g(first);
end
