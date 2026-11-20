classdef mpm_Logger < handle
    %MPM_LOGGER Tiny progress logger: prints to the command window and keeps the lines.
    %   log = mpm_Logger(); log.add('trained %d models', n); lines = log.lines;
    properties
        lines = {}
        t0
        quiet = false
    end
    methods
        function obj = mpm_Logger(quiet)
            if nargin > 0, obj.quiet = quiet; end
            obj.t0 = tic;
        end
        function add(obj, fmt, varargin)
            msg = sprintf(fmt, varargin{:});
            line = sprintf('[%7.1fs] %s', toc(obj.t0), msg);
            obj.lines{end+1, 1} = line;
            if ~obj.quiet, fprintf('%s\n', line); end
        end
    end
end
