function mpm_binary_panel(parent, B, x, y, P, pointLabel, scoreName)
%MPM_BINARY_PANEL GUI block for binary targets: C-A plot | binary map (P-A / C-A) | summary table.
%   Used by the results viewer (Binary targets tab) and by MPM_BINARY_VIEWER.
%   B from MPM_BINARY_TARGETS; x, y cell-centre vectors; P table with x, y, test (points shown on
%   the map, e.g. deposits or high-grade samples); pointLabel e.g. 'deposits'.
if nargin < 6, pointLabel = 'deposits'; end
if nargin < 7, scoreName = 'prospectivity'; end
g = uigridlayout(parent, [3 2], 'RowHeight', {30, '1x', 130}, 'ColumnWidth', {'1x', '1.2x'});
ctl = uigridlayout(g, [1 4], 'ColumnWidth', {60, 260, 170, '1x'}, 'Padding', [0 0 0 0]);
ctl.Layout.Column = [1 2];
uilabel(ctl, 'Text', 'Binary:');
f = fieldnames(B.maps);
names = cellfun(@(k) method_name(k), f, 'UniformOutput', false);
if isempty(f)
    uilabel(ctl, 'Text', strjoin(B.messages, ' '));
    return
end
dd = uidropdown(ctl, 'Items', names, 'ItemsData', f);
cb = uicheckbox(ctl, 'Text', ['Show ' pointLabel], 'Value', ~isempty(P) && height(P) > 0);
axC = uiaxes(g); axC.Layout.Row = 2; axC.Layout.Column = 1;
axM = uiaxes(g); axM.Layout.Row = 2; axM.Layout.Column = 2;
tb = uigridlayout(g, [1 1], 'Padding', [0 0 0 0]); tb.Layout.Row = 3; tb.Layout.Column = [1 2];
T = B.table;
for i = 1:width(T), if isfloat(T.(i)), T.(i) = round(T.(i), 4, 'significant'); end, end
uitable(tb, 'Data', T);
if ~isempty(B.ca)
    thr = []; if isfield(B.thresholds, 'ca'), thr = B.thresholds.ca; end
    mpm_plot_ca(axC, B.ca, thr, scoreName);
else
    title(axC, 'C-A fractal not computed');
end
redraw();
dd.ValueChangedFcn = @(~, ~) redraw(); cb.ValueChangedFcn = @(~, ~) redraw();
    function redraw()
        cla(axM, 'reset'); hold(axM, 'on');
        A = double(B.maps.(dd.Value));
        imagesc(axM, x, y, A, 'AlphaData', isfinite(A));
        colormap(axM, [0.88 0.88 0.88; 0.80 0.10 0.12]); clim(axM, [-0.5 1.5]);
        c = colorbar(axM); c.Ticks = [0 1]; c.TickLabels = {'0 non-deposit', '1 target'};
        if cb.Value && ~isempty(P) && height(P) > 0
            tr = ~P.test;
            plot(axM, P.x(tr), P.y(tr), 'p', 'MarkerSize', 10, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
                'DisplayName', ['Training ' pointLabel]);
            if any(P.test)
                plot(axM, P.x(P.test), P.y(P.test), 'p', 'MarkerSize', 10, 'MarkerFaceColor', [0.85 0.33 0.1], ...
                    'MarkerEdgeColor', 'k', 'DisplayName', ['Test ' pointLabel]);
            end
            legend(axM, 'Location', 'northeastoutside');
        end
        r = B.table(strcmpi(B.table.method, dd.Value), :);
        title(axM, sprintf('%s: t = %.3g, %.1f %% of area', method_name(dd.Value), r.threshold, r.area_pct));
        axis(axM, 'image'); axM.YDir = 'normal'; box(axM, 'on');
        axM.XAxis.Exponent = 0; axM.YAxis.Exponent = 0; xtickformat(axM, '%.0f'); ytickformat(axM, '%.0f');
    end
end

function s = method_name(k)
switch lower(k)
    case 'pa', s = 'P-A intersection';
    case 'ca', s = 'C-A fractal';
    otherwise, s = upper(k);
end
end
