import numpy as np, sys, time
from scipy.ndimage import gaussian_filter
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from kprb import KnowledgeModel, boost_residual, boost_predict, choose_n_trees

G = 120
def field(rng, sig):
    f = gaussian_filter(rng.standard_normal((G, G)), sig, mode="wrap"); return ((f - f.mean()) / f.std()).ravel()

def make_world(seed):
    rng = np.random.default_rng(seed)
    base = [field(rng, s) for s in (10, 6, 8, 5, 7, 4, 6, 3, 6, 5)]
    X = np.column_stack(base)
    X[:, 1] = 0.6 * X[:, 0] + 0.8 * X[:, 1]          # correlated proxies inside process 0
    X[:, 7] = 0.5 * X[:, 6] + 0.87 * X[:, 7]
    X[:, 2] = -X[:, 2]                                # layer 2: LOW values favourable (e.g. distance to fault)
    process = np.array([0, 0, 1, 1, 2, 2, 3, 3]); direction = np.array([1, 1, -1, 1, 1, 1, 1, 1.])
    u = direction * X[:, :8]
    mu = expit(np.array([2.5, 2, 3, 2, 2.5, 2, 3, 2]) * (u - np.array([0.8, 0.5, 0.3, 0.6, 0.7, 0.4, 0.9, 0.5])))
    F = np.column_stack([np.max(mu[:, process == k], 1) for k in range(4)])
    Ktrue = np.exp(np.log(F + 1e-6) @ np.array([0.15, 0.35, 0.2, 0.3]))
    # control unknown to the expert: layer 8, plus an interaction of layers 3 and 5
    lam = Ktrue ** 2.0 * np.exp(1.0 * X[:, 8] + 0.8 * (X[:, 3] > 0.5) * (X[:, 5] > 0.3))
    return X, process, direction, lam

def blocks(n_side=4):
    r, c = np.divmod(np.arange(G * G), G); return (r // (G // n_side)) * n_side + (c // (G // n_side))

def run(seed, N, B=8, ratio=3, expert_noise=True):
    X, process, direction, lam = make_world(seed)
    rng = np.random.default_rng(1000 + seed)
    p = lam / lam.sum()
    pos = rng.choice(G * G, N, replace=False, p=p)
    test_pos = rng.choice(np.setdiff1d(np.arange(G * G), pos), 400, replace=False,
                          p=np.delete(p, pos) / np.delete(p, pos).sum())
    bg = rng.choice(np.setdiff1d(np.arange(G * G), np.r_[pos, test_pos]), 4000, replace=False)
    yt = np.r_[np.ones(400), np.zeros(4000)]; it = np.r_[test_pos, bg]
    w0 = np.array([1, 1, 2, 1, 1, 1, 1, 2.]) if expert_noise else None   # imperfect expert weights
    v0 = np.array([1, 1, 1, 1.])
    mk = lambda: KnowledgeModel(process, direction, w0=w0, v0=v0).fit_scaler(X[:, :8])
    km0 = mk(); expert = km0.parts(X[:, :8])[2]                       # pure expert model, no deposits used
    Xs = (X - np.median(X, 0)) / (np.percentile(X, 75, 0) - np.percentile(X, 25, 0))
    blk = blocks(); out = {k: np.zeros(G * G) for k in ("calib", "rf", "gb", "hyb")}; Ts = []
    unl = np.setdiff1d(np.arange(G * G), pos)
    for b in range(B):
        neg = rng.choice(unl, ratio * N, replace=False)
        idx = np.r_[pos, neg]; y = np.r_[np.ones(N), np.zeros(len(neg))]
        km = mk().fit(X[idx, :8], y); eta_all = km.parts(X[:, :8])[3]
        out["calib"] += expit(eta_all)
        rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=b).fit(Xs[idx], y)
        out["rf"] += rf.predict_proba(Xs)[:, 1]
        prior = np.full(len(y), logit(y.mean()))
        tg = boost_residual(Xs[idx], y, prior, n_trees=150)
        out["gb"] += expit(boost_predict(tg, Xs, np.full(G * G, logit(y.mean()))))
        grid = [0, 10, 25, 50, 100, 150]
        T, _ = choose_n_trees(X[idx, :8], Xs[idx], y, blk[idx], mk, grid)
        Ts.append(T)
        th = boost_residual(Xs[idx], y, km.parts(X[idx, :8])[3], n_trees=max(T, 1))
        out["hyb"] += expit(boost_predict(th, Xs, eta_all, n_use=T))
    res = {"expert": expert, **{k: v / B for k, v in out.items()}}
    return {k: (roc_auc_score(yt, v[it]), spearmanr(v, lam).correlation) for k, v in res.items()}, np.mean(Ts)

if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1].split(",")]; R = int(sys.argv[2])
    for N in Ns:
        t = time.time(); acc = {}; Tm = []
        for s in range(R):
            r, T = run(s, N); Tm.append(T)
            for k, v in r.items(): acc.setdefault(k, []).append(v)
        print(f"N={N} (mean residual trees {np.mean(Tm):.0f}, {time.time()-t:.0f}s)")
        for k, v in acc.items():
            a = np.array(v); print(f"  {k:7s} AUC {a[:,0].mean():.3f} +/- {a[:,0].std():.3f}   Spearman {a[:,1].mean():.3f}")
