"""논문 그림 생성 스크립트 (ch05_mc.tex의 그림 두 개를 만든다).

fig_mc_error.png
    왼쪽 : 에피소드 수에 따른 MC 정책 평가의 오차(모든 방문/첫 방문)를 로그-로그로 그린다.
           점선이 1/sqrt(n) 기준선이고, 가로 점선은 4장 DP가 문턱 1e-3에서 남긴 오차다.
    오른쪽: 상태별 오차를 '방문 횟수'와 '방문한 에피소드 수' 두 기준으로 예측한 값과 견준다.

fig_mc_alpha_rho.png
    왼쪽 : 1/n 평균과 고정값 alpha가 한 상태의 추정치를 어떻게 다르게 움직이는지.
    오른쪽: 오프-정책 예제에서 에피소드 길이와 '중요도 가중치가 살아 있는 꼬리 길이'의 분포.

실행:  python ch05/paper/fig_mc.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np

from common.gridworld import GridWorld

GAMMA = 0.9
DP_ERROR_1EM3 = 4.877e-3   # 4장 표: 문턱 1e-3, 23스윕에서 남은 최대 오차

# ---------------------------------------------------------------- 환경 테이블
_env = GridWorld()
STATES = [(h, w) for h in range(3) for w in range(4)]
SIDX = {s: i for i, s in enumerate(STATES)}
NS, GOAL = 12, SIDX[(0, 3)]
START = SIDX[(2, 0)]
VALID = [SIDX[s] for s in STATES if s not in ((0, 3), (1, 1))]
NEXT = [[SIDX[_env.next_state(s, a)] for a in range(4)] for s in STATES]
REW = [[float(_env.reward(s, a, _env.next_state(s, a))) for a in range(4)] for s in STATES]
UNIFORM_CUM = [[.25, .5, .75, 1.0]] * NS


class Sampler:
    """np.random.Generator.choice(p=...)는 호출당 수십 마이크로초라 역누적분포로 뽑는다."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.buf = self.rng.random(65536)
        self.i = 0

    def u(self):
        if self.i >= 65536:
            self.buf = self.rng.random(65536)
            self.i = 0
        v = self.buf[self.i]
        self.i += 1
        return v


def run_episode(sp, cum):
    s, S, A, R = START, [], [], []
    while True:
        v, c = sp.u(), cum[s]
        a = 0 if v < c[0] else (1 if v < c[1] else (2 if v < c[2] else 3))
        s2 = NEXT[s][a]
        S.append(s); A.append(a); R.append(REW[s][a])
        if s2 == GOAL:
            return S, A, R
        s = s2


def dp_v_pi(thr=1e-13):
    V = [0.0] * NS
    while True:
        d = 0.0
        for s in range(NS):
            if s == GOAL:
                continue
            new = 0.25 * sum(REW[s][a] + GAMMA * V[NEXT[s][a]] for a in range(4))
            d = max(d, abs(new - V[s])); V[s] = new
        if d < thr:
            return np.array(V)


V_PI = dp_v_pi()


def mc_eval(seed, episodes, first_visit=False, alpha=None, trace_state=None):
    sp = Sampler(seed)
    V = np.zeros(NS); cnt = np.zeros(NS); epc = np.zeros(NS); tr = []
    for _ in range(episodes):
        S, A, R = run_episode(sp, UNIFORM_CUM)
        G = 0.0
        first = {}
        for i, s in enumerate(S):
            first.setdefault(s, i)
        for s in first:
            epc[s] += 1
        for i in range(len(S) - 1, -1, -1):
            s = S[i]; G = GAMMA * G + R[i]
            if first_visit and first[s] != i:
                continue
            cnt[s] += 1
            V[s] += (G - V[s]) / cnt[s] if alpha is None else (G - V[s]) * alpha
        if trace_state is not None:
            tr.append(V[trace_state])
    return V, cnt, epc, np.array(tr)


def ret_var(seed, episodes=60000):
    sp = Sampler(seed)
    acc = [[] for _ in range(NS)]
    for _ in range(episodes):
        S, A, R = run_episode(sp, UNIFORM_CUM)
        G = 0.0
        for i in range(len(S) - 1, -1, -1):
            G = GAMMA * G + R[i]
            acc[S[i]].append(G)
    return np.array([np.var(a) if a else 0.0 for a in acc])


def off_policy_tails(seed, episodes=10000, eps=0.1, alpha=0.2):
    """오프-정책 예제를 돌리며 에피소드 길이와 rho가 살아 있는 꼬리 길이를 모은다."""
    sp = Sampler(seed)
    Q = np.zeros((NS, 4))
    pi = np.full((NS, 4), 0.25); b = np.full((NS, 4), 0.25)
    cum = [[.25, .5, .75, 1.0] for _ in range(NS)]
    lens, tails = [], []

    def greedy(q):
        m = q.max(); idx = [i for i in range(4) if q[i] == m]
        return idx[0] if len(idx) == 1 else idx[int(sp.u() * len(idx))]

    for _ in range(episodes):
        S, A, R = run_episode(sp, cum)
        G, rho, tail, dead = 0.0, 1.0, 0, False
        for i in range(len(S) - 1, -1, -1):
            s, a = S[i], A[i]
            G = GAMMA * rho * G + R[i]
            if rho == 0.0:
                dead = True
            elif not dead:
                tail += 1
            Q[s, a] += (G - Q[s, a]) * alpha
            rho *= pi[s, a] / b[s, a]
            gp, gb = greedy(Q[s]), greedy(Q[s])
            pi[s] = 0.0; pi[s, gp] = 1.0
            b[s] = eps / 4; b[s, gb] += 1 - eps
            p = b[s]
            cum[s] = [p[0], p[0] + p[1], p[0] + p[1] + p[2], 1.0]
        lens.append(len(S)); tails.append(tail)
    return np.array(lens), np.array(tails)


def fig_error(outdir):
    ns = [1000, 4000, 16000, 64000, 256000]
    seeds = [120, 120, 80, 40, 20]
    ev, fv = [], []
    for n, k in zip(ns, seeds):
        e = [np.sqrt(np.mean((mc_eval(20000 + s, n)[0][VALID] - V_PI[VALID]) ** 2))
             for s in range(k)]
        f = [np.sqrt(np.mean((mc_eval(20000 + s, n, first_visit=True)[0][VALID]
                              - V_PI[VALID]) ** 2)) for s in range(k)]
        ev.append(np.mean(e)); fv.append(np.mean(f))
        print(f'  n={n}: every-visit {ev[-1]:.5f}  first-visit {fv[-1]:.5f}')

    VG = ret_var(999)
    Vs = np.array([mc_eval(40000 + s, 1000) for s in range(300)], dtype=object)
    per_v = np.array([v[0] for v in Vs]); per_c = np.array([v[1] for v in Vs])
    per_e = np.array([v[2] for v in Vs])
    rmse = np.sqrt(((per_v[:, VALID] - V_PI[VALID]) ** 2).mean(axis=0))
    cnt = per_c[:, VALID].mean(axis=0); epc = per_e[:, VALID].mean(axis=0)
    sig = np.sqrt(VG[VALID])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))

    ax1.loglog(ns, ev, 'o-', label='every-visit MC')
    ax1.loglog(ns, fv, 's-', label='first-visit MC')
    ref = ev[0] * (np.array(ns, float) / ns[0]) ** -0.5
    ax1.loglog(ns, ref, 'k:', lw=0.9, label=r'$\propto 1/\sqrt{n}$')
    ax1.axhline(DP_ERROR_1EM3, color='crimson', ls='--', lw=0.9,
                label=r'DP, $\varepsilon=10^{-3}$ (23 sweeps)')
    ax1.set_xlabel('episodes')
    ax1.set_ylabel(r'RMSE over 10 states')
    ax1.set_title('MC policy evaluation error')
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)

    ax2.loglog(cnt, rmse, 'o', label='measured RMSE (1000 episodes)')
    o = np.argsort(cnt)
    ax2.loglog(cnt[o], (sig / np.sqrt(cnt))[o], 'v--', ms=4,
               label=r'$\sigma_G/\sqrt{\mathrm{visits}}$')
    ax2.loglog(cnt[o], (sig / np.sqrt(epc))[o], '^--', ms=4,
               label=r'$\sigma_G/\sqrt{\mathrm{episodes\ visited}}$')
    ax2.set_xlabel('visits to the state (1000 episodes)')
    ax2.set_ylabel('RMSE of V(s)')
    ax2.set_title('per-state error and its two predictions')
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, 'fig_mc_error.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('그래프 저장:', out)


def fig_alpha_rho(outdir):
    s = SIDX[(2, 2)]
    EP = 30000
    tr_n = [mc_eval(50000 + k, EP, trace_state=s)[3] for k in range(6)]
    tr_a = [mc_eval(50000 + k, EP, alpha=0.1, trace_state=s)[3] for k in range(6)]
    VG = ret_var(999)
    band = np.sqrt(0.1 / (2 - 0.1) * VG[s])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    for t in tr_a:
        ax1.plot(t, color='tab:orange', lw=0.5, alpha=0.6)
    for t in tr_n:
        ax1.plot(t, color='tab:blue', lw=0.8)
    ax1.axhline(V_PI[s], color='k', lw=1.0)
    ax1.axhline(V_PI[s] + band, color='k', ls=':', lw=0.8)
    ax1.axhline(V_PI[s] - band, color='k', ls=':', lw=0.8)
    ax1.plot([], [], color='tab:blue', label=r'$1/n$ average')
    ax1.plot([], [], color='tab:orange', label=r'fixed $\alpha=0.1$')
    ax1.plot([], [], 'k:', label=r'$v_\pi(s)\pm\sqrt{\alpha/(2-\alpha)}\,\sigma_G$')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('V(s) for s = (2,2)')
    ax1.set_title(r'$1/n$ converges, fixed $\alpha$ keeps wandering')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    lens, tails = off_policy_tails(5000)
    bins = np.arange(0, 31) - 0.5
    ax2.hist(np.clip(lens, 0, 30), bins=bins, alpha=0.55, label='episode length')
    ax2.hist(np.clip(tails, 0, 30), bins=bins, alpha=0.55,
             label=r'steps with $\rho \neq 0$')
    ax2.set_xlabel('steps (clipped at 30)')
    ax2.set_ylabel('episodes')
    ax2.set_title('off-policy: how far back the weight survives')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, 'fig_mc_alpha_rho.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('그래프 저장:', out)


if __name__ == '__main__':
    d = os.path.dirname(os.path.abspath(__file__))
    fig_error(d)
    fig_alpha_rho(d)
