function c = mpm_iforest_c(n)
%MPM_IFOREST_C Average path length of an unsuccessful BST search (Liu et al. 2008, Eq. 1).
%
%   c(n) = 2 H(n-1) - 2 (n-1) / n,   H(i) = harmonic number = psi(i+1) - psi(1)  (digamma, exact)
%   c(1) = c(0) = 0  (an external node with one point needs no further isolation).
n = double(n);
c = zeros(size(n));
k = n > 1;
c(k) = 2 * (psi(n(k)) - psi(1)) - 2 * (n(k) - 1) ./ n(k);
end
