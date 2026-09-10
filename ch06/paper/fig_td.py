"""논문 그림 생성 스크립트 (ch06_td.tex의 그림 두 개를 만든다).

fig_td_bias_var.png
    왼쪽 : 무작위 정책 평가에서 TD법과 MC법의 오차를 에피소드 수에 따라 로그-로그로 그린다.
    오른쪽: 같은 실험의 오차를 편향과 표준편차로 갈라 그린다.

fig_td_alpha.png
    왼쪽 : 상태 (2,2)에서 UP의 Q 값이 학습 중에 어떻게 움직이는지.
           SARSA(alpha=0.8), SARSA(alpha=0.1), Q 러닝(alpha=0.8) 세 가지.
    오른쪽: 10,000 에피소드 이후 그 값의 분포. Q 러닝은 한 점에 못 박히고 SARSA는 퍼진다.

실행:  python ch06/paper/fig_td.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from common.gridworld import GridWorld

GAMMA = 0.9
EPS = 0.1

# ---------------------------------------------------------------- 환경 테이블
_env = GridWorld()
STATES = [(h, w) for h in range(3) for w in range(4)]
SIDX = {s: i for i, s in enumerate(STATES)}
NS, GOAL, START = 12, SIDX[(0, 3)], SIDX[(2, 0)]
VALID = [SIDX[s] for s in STATES if s not in ((0, 3), (1, 1))]
S22, A_UP = SIDX[(2, 2)], 0
NEXT = [[SIDX[_env.next_state(s, a)] for a in range(4)] for s in STATES]
REW = [[float(_env.reward(s, a, _env.next_state(s, a))) for a in range(4)] for s in STATES]


class Sampler:
    """호출당 비용을 줄이려고 균등 난수를 미리 뽑아 두고 쓴다."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.buf = self.rng.random(1 << 16)
        self.i = 0

    def u(self):
        if self.i >= len(self.buf):
            self.buf = self.rng.random(1 << 16)
            self.i = 0
        v = self.buf[self.i]
        self.i += 1
        return v


def v_random(thr=1e-14):
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


V_PI = v_random()


def v_star(thr=1e-14):
    V = [0.0] * NS
    while True:
        d = 0.0
        for s in range(NS):
            if s == GOAL:
                continue
            new = max(REW[s][a] + GAMMA * V[NEXT[s][a]] for a in range(4))
            d = max(d, abs(new - V[s])); V[s] = new
        if d < thr:
            return np.array(V)


V_STAR = v_star()


def eps_soft_optimal():
    """eps-소프트 정책 가운데 가치가 가장 높은 것. SARSA의 수렴 대상."""
    det = [0] * NS
    for _ in range(200):
        V = [0.0] * NS
        while True:
            d = 0.0
            for s in range(NS):
                if s == GOAL or s == SIDX[(1, 1)]:
                    continue
                p = [EPS / 4] * 4
                p[det[s]] += 1 - EPS
                new = sum(p[a] * (REW[s][a] + GAMMA * V[NEXT[s][a]]) for a in range(4))
                d = max(d, abs(new - V[s])); V[s] = new
            if d < 1e-14:
                break
        Q = [[REW[s][a] + GAMMA * V[NEXT[s][a]] for a in range(4)] for s in range(NS)]
        new_det = [int(np.argmax(Q[s])) for s in range(NS)]
        if new_det == det:
            return np.array(Q)
        det = new_det
    return np.array(Q)


Q_EPS_SOFT = eps_soft_optimal()
Q_STAR = np.array([[REW[s][a] + GAMMA * V_STAR[NEXT[s][a]] for a in range(4)]
                   for s in range(NS)])


# ------------------------------------------------------------------ 정책 평가
def eval_run(seed, snaps, alpha=0.01):
    """한 시드의 궤적을 만들어 TD / MC(alpha) / MC(1/n) 셋에 똑같이 먹인다.

    같은 데이터를 세 방법에 주므로 비교가 짝지어지고, 궤적 생성 비용도 한 번만 든다.
    """
    sp = Sampler(seed)
    Vtd = np.zeros(NS); Vma = np.zeros(NS); Vmn = np.zeros(NS); cnt = np.zeros(NS)
    out = {}
    snapset = set(snaps)
    for ep in range(1, max(snaps) + 1):
        s = START; traj = []
        while True:
            a = int(sp.u() * 4)
            s2 = NEXT[s][a]; r = REW[s][a]
            Vtd[s] += (r + GAMMA * (0.0 if s2 == GOAL else Vtd[s2]) - Vtd[s]) * alpha
            traj.append((s, r))
            if s2 == GOAL:
                break
            s = s2
        g = 0.0
        for s0, r0 in reversed(traj):
            g = GAMMA * g + r0
            Vma[s0] += (g - Vma[s0]) * alpha
            cnt[s0] += 1
            Vmn[s0] += (g - Vmn[s0]) / cnt[s0]
        if ep in snapset:
            out[ep] = (Vtd.copy(), Vma.copy(), Vmn.copy())
    return out


def _log_xticks(ax, ticks):
    """로그 x축의 자동 눈금은 라벨이 겹치므로 스냅숏 지점에만 숫자를 찍는다."""
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', labelsize=8)


LABELS = ['TD  ' + r'$\alpha=0.01$', 'MC  ' + r'$\alpha=0.01$', 'MC  ' + r'$1/n$']


def fig_bias_var(outdir, seeds=60):
    snaps = [250, 500, 1000, 2000, 4000, 8000]
    acc = [{n: [] for n in snaps} for _ in range(3)]
    for sd in range(seeds):
        out = eval_run(20000 + sd, snaps)
        for n in snaps:
            for k in range(3):
                acc[k][n].append(out[n][k][VALID])
    ref = V_PI[VALID]

    print(f'  {"ep":>6} | ' + ' | '.join(f'{n:^22}' for n in ('TD a=0.01', 'MC a=0.01', 'MC 1/n')))
    print(f'  {"":>6} | ' + ' | '.join(['  RMSE   |bias|    std'] * 3))
    for n in snaps:
        row = f'  {n:>6} | '
        cells = []
        for k in range(3):
            A = np.array(acc[k][n])
            cells.append(f'{np.sqrt(((A - ref) ** 2).mean()):>6.4f}'
                         f'{np.abs(A.mean(axis=0) - ref).mean():>9.4f}'
                         f'{A.std(axis=0).mean():>7.4f}')
        print(row + ' | '.join(cells))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    for k, mk in enumerate(['o-', 's-', '^-']):
        rmse = [np.sqrt(((np.array(acc[k][n]) - ref) ** 2).mean()) for n in snaps]
        ax1.loglog(snaps, rmse, mk, ms=4, label=LABELS[k])
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('RMSE over 10 states')
    ax1.set_title('policy evaluation error (same trajectories)')
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)
    _log_xticks(ax1, snaps)

    for k, c in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
        bias = [np.abs(np.array(acc[k][n]).mean(axis=0) - ref).mean() for n in snaps]
        sd_ = [np.array(acc[k][n]).std(axis=0).mean() for n in snaps]
        ax2.loglog(snaps, bias, 'o-', ms=4, color=c, label=LABELS[k] + '  |bias|')
        ax2.loglog(snaps, sd_, 'o--', ms=4, mfc='none', color=c, label=LABELS[k] + '  std')
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('averaged over 10 states')
    ax2.set_title('bias (solid) and standard deviation (dashed)')
    ax2.legend(fontsize=7)
    ax2.grid(True, which='both', alpha=0.3)
    _log_xticks(ax2, snaps)

    fig.tight_layout()
    out = os.path.join(outdir, 'fig_td_bias_var.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('그래프 저장:', out)


# ---------------------------------------------------------------------- 제어
def control_trace(seed, episodes, kind, alpha):
    """kind: 'sarsa' | 'qlearn'. 에피소드마다 Q[(2,2)][UP]을 기록한다."""
    sp = Sampler(seed)
    Q = np.zeros((NS, 4))
    greedy = np.zeros(NS, dtype=int)
    tr = np.empty(episodes)

    def act(s):
        if sp.u() < EPS:
            return int(sp.u() * 4)
        return greedy[s]

    def bump(s):
        greedy[s] = int(np.argmax(Q[s]))

    for ep in range(episodes):
        s = START
        prev = None
        while True:
            a = act(s)
            s2 = NEXT[s][a]; r = REW[s][a]; done = (s2 == GOAL)
            if kind == 'sarsa':
                if prev is not None:
                    ps, pa, pr, pd = prev
                    nq = 0.0 if pd else Q[s, a]
                    Q[ps, pa] += (pr + GAMMA * nq - Q[ps, pa]) * alpha
                    bump(ps)
                prev = (s, a, r, done)
                if done:
                    Q[s, a] += (r - Q[s, a]) * alpha
                    bump(s)
                    break
            else:
                nqm = 0.0 if done else Q[s2].max()
                Q[s, a] += (r + GAMMA * nqm - Q[s, a]) * alpha
                bump(s)
                if done:
                    break
            s = s2
        tr[ep] = Q[S22, A_UP]
    return tr


def fig_alpha(outdir, episodes=20000, seeds=5, warm=10000):
    cfgs = [('sarsa', 0.8, 'tab:orange', r'SARSA  $\alpha=0.8$'),
            ('sarsa', 0.1, 'tab:green', r'SARSA  $\alpha=0.1$'),
            ('qlearn', 0.8, 'tab:blue', r'Q-learning  $\alpha=0.8$')]
    trs = {}
    for kind, al, _, _ in cfgs:
        trs[(kind, al)] = [control_trace(7000 + k, episodes, kind, al) for k in range(seeds)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    for kind, al, c, lab in cfgs:
        for t in trs[(kind, al)]:
            ax1.plot(t, color=c, lw=0.5, alpha=0.75)
        ax1.plot([], [], color=c, label=lab)
    ax1.axhline(Q_STAR[S22, A_UP], color='k', lw=1.0)
    ax1.axhline(Q_EPS_SOFT[S22, A_UP], color='k', ls=':', lw=1.0)
    ax1.set_ylim(-0.55, 1.15)
    ax1.text(episodes * 0.02, Q_STAR[S22, A_UP] + 0.06,
             r'$q_* = %.4f$  (Q-learning의 수렴 대상)'.replace('의 수렴 대상', ' target')
             .replace('Q-learning', 'Q-learning') % Q_STAR[S22, A_UP], fontsize=8)
    ax1.text(episodes * 0.02, Q_EPS_SOFT[S22, A_UP] - 0.16,
             r'$\varepsilon$-soft optimum $= %.4f$  (SARSA target)' % Q_EPS_SOFT[S22, A_UP],
             fontsize=8)
    ax1.set_xlabel('episode'); ax1.set_ylabel(r'$Q((2,2),\mathrm{UP})$')
    ax1.set_title('Q-learning locks on, SARSA keeps wandering')
    ax1.legend(fontsize=8, loc='lower right'); ax1.grid(True, alpha=0.3)

    for kind, al, c, lab in cfgs:
        vals = np.concatenate([t[warm:] for t in trs[(kind, al)]])
        print(f'  {kind:<7} alpha={al}: {warm} 에피소드 이후 Q((2,2),UP) '
              f'평균 {vals.mean():.4f}  표준편차 {vals.std():.4f}')
        ax2.hist(vals, bins=60, color=c, alpha=0.55, label=f'{lab}  (std {vals.std():.3f})')
    ax2.axvline(Q_STAR[S22, A_UP], color='k', lw=1.0)
    ax2.axvline(Q_EPS_SOFT[S22, A_UP], color='k', ls=':', lw=1.0)
    ax2.set_xlabel(r'$Q((2,2),\mathrm{UP})$ after %d episodes' % warm)
    ax2.set_ylabel('episodes')
    ax2.set_title('where the estimate actually sits')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, 'fig_td_alpha.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('그래프 저장:', out)


if __name__ == '__main__':
    d = os.path.dirname(os.path.abspath(__file__))
    fig_bias_var(d)
    fig_alpha(d)
