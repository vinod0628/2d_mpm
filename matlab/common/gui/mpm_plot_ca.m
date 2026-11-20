function mpm_plot_ca(ax, CA, thr, scoreName)
%MPM_PLOT_CA Log-log concentration-area plot with the fitted fractal segments and breaks.
%   mpm_plot_ca(ax, CA, thr)   CA from MPM_CA_FRACTAL, thr = selected target threshold ([] = none)
if nargin < 3, thr = []; end
if nargin < 4, scoreName = 'score'; end
cla(ax); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, CA.logt, CA.logA, 'o', 'MarkerSize', 4, 'Color', [0.35 0.35 0.35], 'DisplayName', 'C-A data');
cols = lines(height(CA.segments));
for k = 1:height(CA.segments)
    i = CA.t >= CA.segments.t_from(k) & CA.t <= CA.segments.t_to(k);
    plot(ax, CA.logt(i), CA.fit_logA(i), '-', 'LineWidth', 2.2, 'Color', cols(k, :), ...
        'DisplayName', sprintf('Segment %d: D = %.2f (R^2 %.3f)', k, CA.segments.D(k), CA.segments.r2(k)));
end
for b = CA.breaks(:)'
    xline(ax, log10(b), ':', sprintf('%.3g', b), 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off', ...
        'LabelVerticalAlignment', 'bottom');
end
if ~isempty(thr) && isfinite(thr)
    xline(ax, log10(thr), '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.8, ...
        'DisplayName', sprintf('Target threshold %.3g', thr));
end
xlabel(ax, sprintf('log_{10} %s threshold', scoreName)); ylabel(ax, 'log_{10} area (km^2) with value >= threshold');
title(ax, 'Concentration-area (C-A) fractal plot'); legend(ax, 'Location', 'southwest');
end
