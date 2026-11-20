function cfg = clf_demo_config(dataset, root)
%CLF_DEMO_CONFIG Classification (target delineation) settings for the demo datasets.
%
%   cfg = clf_demo_config('carlin' | 'curnamona', root)    root = matlab/ folder (mpm_setup output)
%
%   Shared by every run_*_classification.m script so all methods use the same data, split and
%   sampling. For your own project write a similar function or set the cfg fields in the script.
cfg = struct();
switch lower(dataset)
    case 'carlin'      % 35 Carlin-type Au deposits, 250 m grid, Nevada (USGS OFR 01-221)
        cfg.data_dir   = mpm_demo_data_dir(root, 'nevada_carlin_au');
        cfg.points_csv = fullfile(root, 'data', 'nevada_carlin_au', 'deposits.csv');
        cfg.layers     = {'geology', 'aeromag', 'bouguer', 'potassium', 'thorium', 'uranium', ...
                          'dist_faults', 'as_log_idw', 'sb_log_idw'};
        % classified geophysics: class values are physical values (nT, mGal, ...) -> numeric
        cfg.features.as_numeric = {'aeromag', 'bouguer', 'potassium', 'thorium', 'uranium'};
        cfg.negatives.min_distance = 2000;     % m
        cfg.split.method = 'random';           % or 'block' (spatial) / 'rule' with
        cfg.split.test_fraction = 0.3;         %   cfg.split.test_rule = @(T) T.RANDOM >= 700;
    case 'curnamona'   % 7 REE occurrences, 200 m grid, South Australia (DEEP-SEAM)
        cfg.data_dir   = mpm_demo_data_dir(root, 'curnamona_ree');
        cfg.points_csv = fullfile(root, 'data', 'curnamona_ree', 'deposits.csv');
        cfg.layers     = {'mag_tmi_vrtp', 'mag_tmi_vrtp_1vd', 'mag_tmi_vrtp_as', 'grav', 'grav_1vd', ...
                          'rad_k', 'rad_th', 'rad_u', 'dist_faults', 'dist_granite', ...
                          'dist_mesoproterozoic', 'geochem_*_log_idw'};
        cfg.positives.buffer_cells = 1;        % only 7 deposits -> use the 3x3 neighbourhood
        cfg.negatives.ratio = 2;
        cfg.negatives.min_distance = 1000;
        cfg.split.test_fraction = 0.3;         % 2 test deposits
    otherwise
        error('mpm:config', 'Unknown demo dataset "%s" (carlin | curnamona).', dataset);
end
end
