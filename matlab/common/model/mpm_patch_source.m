classdef mpm_patch_source < handle
%MPM_PATCH_SOURCE Z-scored evidence cube from which patch-based methods (M.input = 'patch',
%   e.g. clf_cnn) cut w x w windows around grid cells.
%
%   src = mpm_patch_source(F, scaler)   cube(r, c, :) = scaler.apply(features of cell (r, c));
%                                        cells outside the study area / missing = 0 (= training mean)
%   P = src.extract(lin, w)             w x w x nFeat x N single patches centred on cells lin
%   P = src.extract(lin, w, dr, dc)     ... centred on lin shifted by (dr, dc) rows / columns
%   d = src.chess_distance(lin)         H x W chessboard distance (cells) to the cells lin
%   src.set_permutation(cols, lin, linFrom) / src.clear_permutation()
%                                        permutation importance: for samples lin, the channels
%                                        cols are taken from the windows of linFrom
%   mpm_patch_source.active(src) / src = mpm_patch_source.active()
%                                        the source of the current run (the pipeline sets it, the
%                                        method reads it; it is not stored in result.mat)
%
%   The pipeline passes patch methods the LINEAR CELL INDICES of the samples as "X" (N x 1).

    properties (Transient)
        cube single = single([])          % H x W x nFeat, zero outside the study area
    end
    properties
        H double = 0
        W double = 0
        nFeat double = 0
        valid logical = false(0)          % H x W study-area mask
        names cell = {}                   % feature names
        source double = []                % layer index of each feature (importance groups)
    end
    properties (Transient, Access = private)
        perm = []                         % active permutation for importance
    end

    methods
        function obj = mpm_patch_source(F, scaler, chunk)
            if nargin < 3 || isempty(chunk), chunk = 200000; end
            [obj.H, obj.W] = size(F.valid);
            obj.nFeat = size(F.X, 2);
            obj.valid = F.valid; obj.names = F.names; obj.source = F.source;
            C = zeros(obj.H * obj.W, obj.nFeat, 'single');
            for i = 1:chunk:numel(F.idx)
                j = i:min(i + chunk - 1, numel(F.idx));
                Z = scaler.apply(F.X(j, :));
                Z(~isfinite(Z)) = 0;
                C(F.idx(j), :) = Z;
            end
            obj.cube = reshape(C, obj.H, obj.W, obj.nFeat);
        end

        function P = extract(obj, lin, w, dr, dc)
            % w x w x nFeat x N patches centred on cells lin (+ optional shifts dr, dc)
            if isempty(obj.cube), error('mpm:patch', 'Patch source has no data (results reloaded from disk cannot be re-predicted).'); end
            lin = lin(:); n = numel(lin);
            [r, c] = ind2sub([obj.H obj.W], lin);
            if nargin > 3, r = r + dr(:); c = c + dc(:); end
            P = zeros(w, w, obj.nFeat, n, 'single');
            if n == 0, return; end
            h = (w - 1) / 2;
            C = reshape(obj.cube, [], obj.nFeat);
            for dx = -h:h
                for dy = -h:h
                    rr = r + dy; cc = c + dx;
                    ok = rr >= 1 & rr <= obj.H & cc >= 1 & cc <= obj.W;
                    v = zeros(n, obj.nFeat, 'single');
                    v(ok, :) = C(rr(ok) + (cc(ok) - 1) * obj.H, :);
                    P(dy + h + 1, dx + h + 1, :, :) = reshape(v', 1, 1, obj.nFeat, n);
                end
            end
            if ~isempty(obj.perm) && nargin <= 3     % permutation importance (unshifted samples only)
                [tf, k] = ismember(lin, obj.perm.lin);
                if any(tf)
                    Q = obj.extract_raw(obj.perm.from(k(tf)), w);
                    P(:, :, obj.perm.cols, tf) = Q(:, :, obj.perm.cols, :);
                end
            end
        end

        function d = chess_distance(obj, lin)
            m = false(obj.H, obj.W); m(lin) = true;
            d = bwdist(m, 'chessboard');
        end

        function set_permutation(obj, cols, lin, linFrom)
            obj.perm = struct('cols', cols(:)', 'lin', lin(:), 'from', linFrom(:));
        end

        function clear_permutation(obj)
            obj.perm = [];
        end
    end

    methods (Access = private)
        function P = extract_raw(obj, lin, w)
            p = obj.perm; obj.perm = [];
            P = obj.extract(lin, w);
            obj.perm = p;
        end
    end

    methods (Static)
        function out = active(src)
            persistent current
            if nargin > 0, current = src; end
            out = current;
        end
    end
end
