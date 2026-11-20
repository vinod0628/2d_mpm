"""MS-KPRB: mineral-system knowledge-prior residual boosting (reference prototype)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.tree import DecisionTreeRegressor

EPS = 1e-6

class KnowledgeModel:
    """Two-level soft fuzzy inference network with expert priors (MAP estimation)."""
    def __init__(self, process, direction, w0=None, v0=None, p=3.0,
                 s0=1.5, c0=0.5, kappa=10.0, tau=0.75):
        self.process = np.asarray(process)          # process index of each layer (0..K-1)
        self.direction = np.asarray(direction, float)  # +1: high is favourable, -1: low is favourable
        self.m = len(self.process); self.K = self.process.max() + 1
        self.p, self.s0, self.c0, self.kappa, self.tau = p, s0, c0, kappa, tau
        w0 = np.ones(self.m) if w0 is None else np.asarray(w0, float)
        self.w0 = np.array([w0[j] / w0[self.process == self.process[j]].sum() for j in range(self.m)])
        v0 = np.ones(self.K) if v0 is None else np.asarray(v0, float)
        self.v0 = v0 / v0.sum()
        self.theta = self._theta0()

    def fit_scaler(self, X):                        # robust scaling on ALL cells (no labels used)
        self.med = np.median(X, 0)
        q75, q25 = np.percentile(X, [75, 25], 0)
        self.iqr = np.where(q75 - q25 > 0, q75 - q25, 1.0)
        return self

    def _u(self, X):
        return self.direction * (X - self.med) / self.iqr

    def _theta0(self):                               # [a (m), c (m), omega (m), nu (K), alpha, b]
        return np.concatenate([np.full(self.m, np.log(self.s0)), np.full(self.m, self.c0),
                               np.log(self.w0), np.log(self.v0), [0.0, 0.0]])

    def _unpack(self, th):
        m, K = self.m, self.K
        a, c, om, nu = th[:m], th[m:2*m], th[2*m:3*m], th[3*m:3*m+K]
        w = np.empty(m)
        for k in range(K):
            idx = self.process == k
            w[idx] = softmax(om[idx])
        return np.exp(a), c, w, softmax(nu), th[-2], np.exp(th[-1])

    def parts(self, X, th=None):
        s, c, w, v, alpha, beta = self._unpack(self.theta if th is None else th)
        mu = expit(s * (self._u(X) - c))                                   # memberships
        F = np.column_stack([(mu[:, self.process == k] ** self.p @ w[self.process == k]) ** (1 / self.p)
                             for k in range(self.K)])                      # OR-like within a process
        Kf = np.exp(np.log(np.clip(F, EPS, 1)) @ v)                        # AND-like across processes
        eta = alpha + beta * logit(np.clip(Kf, EPS, 1 - EPS))              # log-odds of the knowledge prior
        return mu, F, Kf, eta

    def _neg_log_post(self, th, X, y, sw):
        eta = self.parts(X, th)[3]
        nll = np.sum(sw * (np.logaddexp(0, eta) - y * eta))
        s, c, w, v, _, _ = self._unpack(th)
        pen = (np.sum((np.log(s) - np.log(self.s0)) ** 2) + np.sum((c - self.c0) ** 2)) / (2 * self.tau ** 2)
        pen -= self.kappa * (np.sum(self.w0 * np.log(w + EPS)) + np.sum(self.v0 * np.log(v + EPS)))
        return nll + pen

    def fit(self, X, y, sample_weight=None):
        sw = np.ones(len(y)) if sample_weight is None else sample_weight
        th0 = self._theta0(); th0[-2] = logit(np.clip(np.average(y, weights=sw), EPS, 1 - EPS))
        res = minimize(self._neg_log_post, th0, args=(X, y, sw), method="L-BFGS-B",
                       options={"maxiter": 200})
        self.theta = res.x
        return self

def boost_residual(X, y, offset, n_trees=150, lr=0.05, depth=2, min_leaf=5, rng=None):
    """Gradient boosting (logistic loss) that starts from a fixed offset = knowledge log-odds."""
    F = offset.copy(); trees = []
    for _ in range(n_trees):
        p = expit(F); r = y - p
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=rng).fit(X, r)
        leaf = t.apply(X); gam = {}
        for l in np.unique(leaf):
            i = leaf == l
            gam[l] = np.clip(r[i].sum() / max((p[i] * (1 - p[i])).sum(), 1e-3), -4, 4)   # Newton step
        F = F + lr * np.vectorize(gam.get)(leaf)
        trees.append((t, gam))
    return trees

def boost_predict(trees, X, offset, lr=0.05, n_use=None):
    F = offset.copy()
    for t, gam in trees[:n_use]:
        F = F + lr * np.vectorize(gam.get)(t.apply(X))
    return F

def choose_n_trees(Xk, Xt, y, groups, km_factory, grid, **bk):
    """Pick the number of residual trees by grouped (spatial-block) cross-validated deviance. 0 is allowed."""
    dev = np.zeros(len(grid)); folds = np.unique(groups)
    for g in folds:
        tr, te = groups != g, groups == g
        if y[tr].sum() == 0 or y[te].sum() == 0: continue
        km = km_factory().fit(Xk[tr], y[tr])
        off_tr, off_te = km.parts(Xk[tr])[3], km.parts(Xk[te])[3]
        trees = boost_residual(Xt[tr], y[tr], off_tr, n_trees=max(grid), **bk)
        for i, T in enumerate(grid):
            eta = boost_predict(trees, Xt[te], off_te, lr=bk.get("lr", 0.05), n_use=T)
            dev[i] += np.sum(np.logaddexp(0, eta) - y[te] * eta)
    return grid[int(np.argmin(dev))], dev
