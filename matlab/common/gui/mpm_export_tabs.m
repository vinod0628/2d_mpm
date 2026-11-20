function files = mpm_export_tabs(fig, tg, folder, prefix)
%MPM_EXPORT_TABS Write one PNG per tab of a uitabgroup (used by the results / compare viewers).
%   Tabs with axes are exported with exportgraphics; tabs holding only tables or text (which
%   exportgraphics rejects) fall back to a screenshot of the window (exportapp).
if nargin < 4, prefix = ''; end
if ~isfolder(folder), mkdir(folder); end
prev = tg.SelectedTab;
files = {};
for i = 1:numel(tg.Children)
    tab = tg.Children(i);
    tg.SelectedTab = tab; drawnow;
    name = strip(regexprep(lower(tab.Title), '[^a-z0-9]+', '_'), '_');
    file = fullfile(folder, sprintf('%s%02d_%s.png', prefix, i, name));
    try
        exportgraphics(tab, file, 'Resolution', 150);
    catch
        try
            exportapp(fig, file);
        catch err
            warning('mpm:export', '%s: %s', tab.Title, err.message); continue;
        end
    end
    files{end+1} = file; %#ok<AGROW>
end
tg.SelectedTab = prev;
end
