function cfg = reg_demo_config(dataset, root)
%REG_DEMO_CONFIG Regression (grade estimation) settings for the demo datasets.
%
%   cfg = reg_demo_config('carlin' | 'curnamona', root)    root = matlab/ folder
%
%   The demo sets contain no orebody grades, so surface geochemistry stands in for grade.
%   Geochemical evidence layers interpolated from the SAME samples are excluded (leakage).
%
%   'carlin' (default): Th (ppm) in NURE stream sediments. Chosen for its geological link to the
%       evidence: airborne gamma-ray Th measures the same element (2.61 MeV 208Tl line, top ~30 cm),
%       Th is immobile, and stream sediments and radiometrics both integrate surface material over
%       an area. Spearman rho of log Th with the airborne thorium layer at the samples = 0.52
%       (K: 0.46, U: 0.33 - mobile; As / Sb pathfinders <= 0.31).
%   'curnamona': La (ppm) in 615 exploration rock samples - weak link to every layer
%       (rho <= 0.27 for all elements; mineralisation-biased point samples), kept for comparison.
%
%   Your own orebody data: a CSV with x, y (raster CRS) and a grade column, e.g.
%       cfg.points_csv = 'D:\data\composites.csv'; cfg.target_field = 'Au_gpt'; cfg.target_label = 'Au (g/t)';
cfg = struct();
switch lower(dataset)
    case 'curnamona'   % 615 surface rock samples, La (ppm) as REE grade proxy
        cfg.data_dir     = mpm_demo_data_dir(root, 'curnamona_ree');
        cfg.points_csv   = fullfile(root, 'data', 'curnamona_ree', 'geochem_samples.csv');
        cfg.target_field = 'La';
        cfg.target_label = 'La (ppm)';
        cfg.exclude_layers = {'geochem_*', 'dist_faults_class'};
    case 'carlin'      % 3 472 NURE stream sediments, Th (ppm); 3 283 above detection
        cfg.data_dir     = mpm_demo_data_dir(root, 'nevada_carlin_au');
        cfg.points_csv   = fullfile(root, 'data', 'nevada_carlin_au', 'nure_geochem.csv');
        cfg.target_field = 'NAA_TH';
        cfg.target_label = 'Th (ppm)';
        cfg.layers = {'geology', 'aeromag', 'bouguer', 'potassium', 'thorium', 'uranium', 'dist_faults'};
        cfg.features.as_numeric = {'aeromag', 'bouguer', 'potassium', 'thorium', 'uranium'};
        cfg.nonpositive = 'drop';              % NURE 0 = below detection / not analysed
    otherwise
        error('mpm:config', 'Unknown demo dataset "%s" (curnamona | carlin).', dataset);
end
end
