
#!/usr/bin/env python3
# Fixed-Budget, Fixed-Memory Root-Bandit MCTS (refined)
# Environments: Nim(15), Othello 4x4 (with pass), Connect-4 5x6
# Baselines: Random, Greedy
# Harness: matchups, logging, Wilson CI from p-hat (draw=0.5)、per-move latency p50/p95、
#          node-usage grouped by matchup + (1+K) theoretical cap and observed cap

import os, sys, math, time, json, argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Utility ----------------

def wilson_from_phat(p: float, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval given p-hat in [0,1] and trials n."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (p, center - half, center + half)

def prf_seed(*args)->int:
    """Deterministic 64-bit PRF seed from a tuple of hashables."""
    h = 0x9e3779b97f4a7c15
    for x in args:
        h ^= (hash(x) + 0x9e3779b97f4a7c15 + ((h<<6)&0xffffffffffffffff) + (h>>2)) & 0xffffffffffffffff
    return h & 0xffffffffffffffff

# ---------------- Root-Bandit MCTS ----------------

class RootBanditMCTS:
    """
    One-step MCTS as a root-child bandit with truncated rollouts.
    O(1) per move if B,K,L are constants. Memory O(K).
    """
    def __init__(self, legal_fn, step_fn, terminal_fn, reward_fn, state_key_fn,
                 B:int=32, K:int=8, L:int=16, C:float=0.5, stratified:int=1, name:Optional[str]=None):
        self.legal_fn = legal_fn
        self.step_fn = step_fn
        self.terminal_fn = terminal_fn
        self.reward_fn = reward_fn
        self.state_key = state_key_fn
        self.B = int(B); self.K = int(K); self.L = int(L); self.C = float(C)
        self.m = int(stratified)  # floor samples/action
        self.name = name or f"MCTS(B={B},K={K},L={L},C={C},m={self.m})"
        self.telemetry_last = {"used_B":0, "nodes_used":0, "expanded":0}

    def decide(self, state)->Any:
        root_player = state.player
        acts = self.legal_fn(state)
        if len(acts) == 0:
            acts = [None]  # pass

        # progressive widening cap
        if len(acts) > self.K:
            r = random.Random(prf_seed(self.state_key(state), "root"))
            r.shuffle(acts)
            acts = acts[:self.K]

        # precompute root children
        children = {a: self.step_fn(state, a) for a in acts}
        W = {a:0.0 for a in acts}; N = {a:0 for a in acts}
        used = 0

        # stratified floor
        for _ in range(self.m):
            for a, s1 in children.items():
                if used >= self.B: break
                R = self._rollout(s1, root_player, used)
                W[a] += R; N[a] += 1; used += 1

        # UCB on remaining budget
        while used < self.B:
            totalN = max(1, sum(N.values()))
            best_a, best_s, best_score = None, None, -1.0
            for a, s1 in children.items():
                if N[a] == 0:
                    score = float("inf")
                else:
                    score = (W[a]/N[a]) + self.C * math.sqrt(max(1e-12, math.log(totalN+1))/(N[a]))
                if score > best_score:
                    best_score, best_a, best_s = score, a, s1
            R = self._rollout(best_s, root_player, used)
            W[best_a] += R; N[best_a] += 1; used += 1

        best_a = max(acts, key=lambda a: (-1e9 if N[a]==0 else W[a]/N[a], -hash(a)))
        self.telemetry_last = {"used_B": used, "nodes_used": 1 + len(acts), "expanded": len(acts)}
        return best_a

    def _rollout(self, state, root_player:int, used:int)->float:
        s = state
        key0 = self.state_key(s)
        rng = random.Random(prf_seed(key0, used, self.B))
        steps = 0
        while (not self.terminal_fn(s)) and steps < self.L:
            acts = self.legal_fn(s)
            if len(acts) == 0:
                s = self.step_fn(s, None)
            else:
                a = rng.choice(acts)
                s = self.step_fn(s, a)
            steps += 1
        return self.reward_fn(root_player, s)

# ---------------- Environments ----------------

# Nim(15)
@dataclass(frozen=True)
class NimState:
    stones: int
    player: int  # +1 or -1

def nim_initial(stones:int=15)->NimState:
    return NimState(stones, +1)

def nim_legal(s:NimState)->List[int]:
    return list(range(1, min(3, s.stones) + 1))

def nim_step(s:NimState, a:Optional[int])->NimState:
    if a is None:
        return NimState(s.stones, -s.player)
    return NimState(s.stones - a, -s.player)

def nim_terminal(s:NimState)->bool:
    return s.stones == 0

def nim_reward(root_player:int, s:NimState)->float:
    if not nim_terminal(s): return 0.0
    winner = -s.player
    return 1.0 if winner == root_player else 0.0

def nim_state_key(s:NimState)->int:
    return (s.stones<<1) | (1 if s.player==+1 else 0)

class RandomNim:
    def decide(self, s:NimState)->int:
        return random.choice(nim_legal(s))

class GreedyNim:
    def decide(self, s:NimState)->int:
        acts = nim_legal(s)
        for a in acts:
            if (s.stones - a) % 4 == 0:
                return a
        return acts[-1]

# Othello 4x4
@dataclass(frozen=True)
class OthState:
    board: Tuple[Tuple[int,...], ...]  # 4x4, 0 empty, +1 black, -1 white
    player: int
    passed: bool

def oth_initial()->OthState:
    B = [[0]*4 for _ in range(4)]
    B[1][1] = -1; B[2][2] = -1; B[1][2] = +1; B[2][1] = +1
    return OthState(tuple(tuple(row) for row in B), +1, False)

DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def _inside(r,c): return 0<=r<4 and 0<=c<4

def oth_legal(s:OthState)->List[Tuple[int,int]]:
    B = s.board; p=s.player; opp=-p
    moves=[]
    for r in range(4):
        for c in range(4):
            if B[r][c]!=0: continue
            ok=False
            for dr,dc in DIRS:
                rr=r+dr; cc=c+dc; seen=False
                while _inside(rr,cc) and B[rr][cc]==opp:
                    seen=True; rr+=dr; cc+=dc
                if seen and _inside(rr,cc) and B[rr][cc]==p:
                    ok=True; break
            if ok: moves.append((r,c))
    return moves

def oth_step(s:OthState, a:Optional[Tuple[int,int]])->OthState:
    B=[list(row) for row in s.board]; p=s.player; opp=-p
    if a is None:
        return OthState(tuple(tuple(row) for row in B), -p, True)
    r,c=a; B[r][c]=p
    for dr,dc in DIRS:
        rr=r+dr; cc=c+dc; toflip=[]
        while _inside(rr,cc) and B[rr][cc]==opp:
            toflip.append((rr,cc)); rr+=dr; cc+=dc
        if toflip and _inside(rr,cc) and B[rr][cc]==p:
            for fr,fc in toflip: B[fr][fc]=p
    return OthState(tuple(tuple(row) for row in B), -p, False)

def oth_terminal(s:OthState)->bool:
    if any(0 in row for row in s.board):
        if len(oth_legal(s))>0: return False
        return s.passed
    return True

def oth_reward(root_player:int, s:OthState)->float:
    black = sum(1 for r in range(4) for c in range(4) if s.board[r][c]==+1)
    white = sum(1 for r in range(4) for c in range(4) if s.board[r][c]==-1)
    if root_player==+1:
        if black>white: return 1.0
        if black<white: return 0.0
    else:
        if white>black: return 1.0
        if white<black: return 0.0
    return 0.5

def oth_state_key(s:OthState)->int:
    v=0
    for r in range(4):
        for c in range(4):
            cell = s.board[r][c]
            code = 0 if cell==0 else (1 if cell==+1 else 2)
            v = (v<<2)|code
    v = (v<<1)|(1 if s.player==+1 else 0)
    v = (v<<1)|(1 if s.passed else 0)
    return v

class RandomOth:
    def decide(self, s:OthState):
        acts = oth_legal(s)
        return None if not acts else random.choice(acts)

class GreedyOth:
    CORNERS={(0,0),(0,3),(3,0),(3,3)}
    EDGES={(0,1),(0,2),(1,0),(2,0),(1,3),(2,3),(3,1),(3,2)}
    def decide(self, s:OthState):
        acts = oth_legal(s)
        if not acts: return None
        def score(a):
            if a in GreedyOth.CORNERS: base=100
            elif a in GreedyOth.EDGES: base=10
            else: base=1
            cnt=0; B=[list(row) for row in s.board]; p=s.player; opp=-p; r,c=a; B[r][c]=p
            for dr,dc in DIRS:
                rr=r+dr; cc=c+dc; t=0
                while _inside(rr,cc) and B[rr][cc]==opp:
                    t+=1; rr+=dr; cc+=dc
                if t and _inside(rr,cc) and B[rr][cc]==p: cnt+=t
            return base*100 + cnt
        return max(acts, key=score)

# Connect-4 (5x6)
@dataclass(frozen=True)
class C4State:
    board: Tuple[Tuple[int,...], ...]  # 6 rows x 5 cols; row 0 bottom
    player: int

def c4_initial()->C4State:
    return C4State(tuple(tuple(0 for _ in range(5)) for _ in range(6)), +1)

def c4_legal(s:C4State)->List[int]:
    return [c for c in range(5) if s.board[5][c]==0]

def c4_step(s:C4State, a:Optional[int])->C4State:
    if a is None:
        return s
    B=[list(row) for row in s.board]; p=s.player
    for r in range(6):
        if B[r][a]==0:
            B[r][a]=p; break
    return C4State(tuple(tuple(row) for row in B), -p)

def _c4_line(B, r, c, dr, dc):
    p=B[r][c]
    if p==0: return 0
    for k in range(1,4):
        rr=r+dr*k; cc=c+dc*k
        if not (0<=rr<6 and 0<=cc<5): return 0
        if B[rr][cc]!=p: return 0
    return p

def c4_winner(B)->int:
    for r in range(6):
        for c in range(5):
            for dr,dc in [(1,0),(0,1),(1,1),(1,-1)]:
                p=_c4_line(B,r,c,dr,dc)
                if p!=0: return p
    return 0

def c4_terminal(s:C4State)->bool:
    if c4_winner(s.board)!=0: return True
    return all(s.board[5][c]!=0 for c in range(5))

def c4_reward(root_player:int, s:C4State)->float:
    w=c4_winner(s.board)
    if w==0: return 0.5
    return 1.0 if w==root_player else 0.0

def c4_state_key(s:C4State)->int:
    v=0
    for r in reversed(range(6)):
        for c in range(5):
            cell=s.board[r][c]
            code=0 if cell==0 else (1 if cell==+1 else 2)
            v=(v<<2)|code
    v=(v<<1)|(1 if s.player==+1 else 0)
    return v

class RandomC4:
    def decide(self, s:C4State):
        return random.choice(c4_legal(s))

class GreedyC4:
    def decide(self, s:C4State):
        acts=c4_legal(s)
        p=s.player; opp=-p
        def sim_drop(B,c,pl):
            BB=[list(row) for row in B]
            for r in range(6):
                if BB[r][c]==0: BB[r][c]=pl; break
            return tuple(tuple(row) for row in BB)
        for c in acts:
            if c4_winner(sim_drop(s.board,c,p))==p: return c
        for c in acts:
            if c4_winner(sim_drop(s.board,c,opp))==opp: return c
        return min(acts, key=lambda x: abs(x-2))

# ---------------- Harness ----------------

def play_game(env:str, agent_first, agent_second, seed:int=0):
    if env=="nim":
        state=nim_initial(15); legal=nim_legal; step=nim_step; term=nim_terminal; rew=nim_reward
    elif env=="othello4x4":
        state=oth_initial(); legal=oth_legal; step=oth_step; term=oth_terminal; rew=oth_reward
    elif env=="connect4_5x6":
        state=c4_initial(); legal=c4_legal; step=c4_step; term=c4_terminal; rew=c4_reward
    else:
        raise ValueError("unknown env")

    moves=[]; t_move_ns=[]; nodes_used_max=0; usedB_total=0; ply=0
    while True:
        pl = state.player
        agent = agent_first if pl==+1 else agent_second
        t0=time.perf_counter_ns()
        a = agent.decide(state)
        t1=time.perf_counter_ns()
        lat_ns = t1-t0
        t_move_ns.append(lat_ns if hasattr(agent, "telemetry_last") else 0)
        used = getattr(agent, "telemetry_last", {}).get("used_B", 0)
        nodes = getattr(agent, "telemetry_last", {}).get("nodes_used", 0)
        nodes_used_max = max(nodes_used_max, nodes)
        usedB_total += used
        moves.append((ply, pl, a, lat_ns, used, nodes))
        state = step(state, a)
        ply += 1
        if term(state): break
    r = rew(+1, state)
    res_first = 1.0 if r>0.5 else (0.0 if r<0.5 else 0.5)
    return res_first, moves, {
        "plies": ply,
        "latency_ns_p50": float(np.percentile(t_move_ns,50)),
        "latency_ns_p95": float(np.percentile(t_move_ns,95)),
        "nodes_used_max": nodes_used_max,
        "usedB_total": usedB_total
    }

def make_agents(env:str, B:int, K:int, L:int, C:float, m:int):
    if env=="nim":
        mcts = RootBanditMCTS(nim_legal, nim_step, nim_terminal, nim_reward, nim_state_key,
                              B=B,K=K,L=L,C=C,stratified=m, name=f"MCTS(B={B})")
        rnd = RandomNim(); gry = GreedyNim()
    elif env=="othello4x4":
        mcts = RootBanditMCTS(oth_legal, oth_step, oth_terminal, oth_reward, oth_state_key,
                              B=B,K=K,L=L,C=C,stratified=m, name=f"MCTS(B={B})")
        rnd = RandomOth(); gry = GreedyOth()
    elif env=="connect4_5x6":
        mcts = RootBanditMCTS(c4_legal, c4_step, c4_terminal, c4_reward, c4_state_key,
                              B=B,K=K,L=L,C=C,stratified=m, name=f"MCTS(B={B})")
        rnd = RandomC4(); gry = GreedyC4()
    else:
        raise ValueError("unknown env")
    return mcts, rnd, gry

def run_card(env:str, matchup:str, B:int, games:int, K:int, L:int, C:float, m:int, out_dir:str):
    mcts, rnd, gry = make_agents(env, B, K, L, C, m)
    wins=0.0; n=0
    per_game=[]; per_move=[]
    for g in range(games):
        if g%2==0:
            first = mcts
            second = rnd if matchup=="MCTS_vs_Random" else gry
            res_first, moves, gsum = play_game(env, first, second, seed=1000+B+g)
            mcts_won = res_first
        else:
            first = rnd if matchup=="MCTS_vs_Random" else gry
            second = mcts
            res_first, moves, gsum = play_game(env, first, second, seed=1000+B+g)
            mcts_won = 1.0 - res_first if res_first in (0.0,1.0) else 0.5
        wins += mcts_won; n += 1
        per_game.append({
            "game_id": g, "plies": gsum["plies"],
            "p50_ms": gsum["latency_ns_p50"]/1e6,
            "p95_ms": gsum["latency_ns_p95"]/1e6,
            "nodes_used_max": gsum["nodes_used_max"],
            "B": B, "matchup": matchup
        })
        for (ply, pl, a, lat_ns, used, nodes) in moves:
            is_mcts_turn = (pl==+1 and g%2==0) or (pl==-1 and g%2==1)
            per_move.append({
                "game_id": g, "ply": ply, "matchup": matchup,
                "latency_ms": lat_ns/1e6 if is_mcts_turn else np.nan,
                "used_B": used if is_mcts_turn else np.nan,
                "nodes_used": nodes if is_mcts_turn else np.nan,
                "B": B
            })

    p_hat = wins / n
    _, lo, hi = wilson_from_phat(p_hat, n)
    summary = {
        "env": env, "matchup": matchup, "B": B,
        "games": n, "wins": wins, "winrate": p_hat,
        "wilson_lo": lo, "wilson_hi": hi,
        "p50_ms": float(np.median([r["p50_ms"] for r in per_game])),
        "p95_ms": float(np.percentile([r["p95_ms"] for r in per_game],95)),
        "nodes_used_max": int(max([r["nodes_used_max"] for r in per_game]) if per_game else 0)
    }
    subdir = os.path.join(out_dir, f"{env}_B{B}_{matchup}")
    os.makedirs(subdir, exist_ok=True)
    pd.DataFrame([summary]).to_json(os.path.join(subdir,"summary.json"), orient="records", indent=2)
    pd.DataFrame(per_game).to_csv(os.path.join(subdir,"games.csv"), index=False)
    pd.DataFrame(per_move).to_csv(os.path.join(subdir,"moves.csv"), index=False)
    return summary, pd.DataFrame(per_game), pd.DataFrame(per_move)

def plot_suite(env:str, K_cap:int, summaries:pd.DataFrame, moves_df:pd.DataFrame, out_dir:str):
    # 1) Win rate vs B
    fig1 = plt.figure(figsize=(6,4))
    for matchup in summaries["matchup"].unique():
        sub = summaries[summaries["matchup"]==matchup].sort_values("B")
        plt.plot(sub["B"], sub["winrate"], marker="o", label=matchup)
        yerr_lower = sub["winrate"] - sub["wilson_lo"]
        yerr_upper = sub["wilson_hi"] - sub["winrate"]
        yerr = np.vstack([yerr_lower.values, yerr_upper.values])
        plt.errorbar(sub["B"], sub["winrate"], yerr=yerr, fmt="none")
    plt.xlabel("B (rollouts per move)")
    plt.ylabel("Win rate (MCTS)")
    plt.title(f"Win rate vs B ({env})")
    plt.legend()
    plt.tight_layout()
    p1=os.path.join(out_dir, f"{env}_winrate_vs_B.png")
    plt.savefig(p1, dpi=160)
    plt.show()

    # 2) Latency vs B using per-move distribution
    lat_rows=[]
    for B in sorted(summaries["B"].unique()):
        sub = moves_df[(moves_df["B"]==B) & (~moves_df["latency_ms"].isna())]
        lat_rows.append({
            "B": B,
            "p50_ms": float(np.percentile(sub["latency_ms"],50)) if len(sub) else 0.0,
            "p95_ms": float(np.percentile(sub["latency_ms"],95)) if len(sub) else 0.0
        })
    df_lat = pd.DataFrame(lat_rows).sort_values("B")
    fig2 = plt.figure(figsize=(6,4))
    plt.plot(df_lat["B"], df_lat["p50_ms"], marker="o", label="p50 (per-move)")
    plt.plot(df_lat["B"], df_lat["p95_ms"], marker="s", label="p95 (per-move)")
    plt.xlabel("B (rollouts per move)")
    plt.ylabel("Latency (ms) per move")
    plt.title(f"Latency vs B ({env})")
    plt.legend()
    plt.tight_layout()
    p2=os.path.join(out_dir, f"{env}_latency_vs_B.png")
    plt.savefig(p2, dpi=160)
    plt.show()

    # 3) Node usage vs caps (grouped by matchup)
    N_theory = 1 + K_cap
    fig3 = plt.figure(figsize=(6,4))
    for matchup in summaries["matchup"].unique():
        sub = summaries[summaries["matchup"]==matchup].sort_values("B")
        plt.plot(sub["B"], sub["nodes_used_max"], marker="o", label=f"{matchup} nodes")
    xs = sorted(summaries["B"].unique())
    plt.plot(xs, [N_theory]*len(xs), linestyle="-", label=f"Theoretical cap 1+K={N_theory}")
    observed_cap = int(summaries["nodes_used_max"].max())
    plt.plot(xs, [observed_cap]*len(xs), linestyle="--", label=f"Observed cap N={observed_cap}")
    plt.xlabel("B (rollouts per move)")
    plt.ylabel("Nodes (max used per move)")
    plt.title(f"Node usage vs caps ({env})")
    plt.legend()
    plt.tight_layout()
    p3=os.path.join(out_dir, f"{env}_nodes_used_vs_N.png")
    plt.savefig(p3, dpi=160)
    plt.show()
    return (p1,p2,p3)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["nim","othello4x4","connect4_5x6"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--B", type=int, nargs="+", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--stratified", type=int, default=1)
    parser.add_argument("--K", type=int, default=None, help="override K (root children cap)")
    parser.add_argument("--L", type=int, default=None, help="override rollout horizon")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    # defaults per env
    if args.env=="nim":         K_def,L_def = 3,15
    elif args.env=="othello4x4":K_def,L_def = 8,16
    else:                       K_def,L_def = 5,30
    K = args.K if args.K is not None else K_def
    L = args.L if args.L is not None else L_def

    os.makedirs(args.out, exist_ok=True)

    summaries=[]; all_games=[]; all_moves=[]
    for B in args.B:
        for matchup in ["MCTS_vs_Random","MCTS_vs_Greedy"]:
            s, df_g, df_m = run_card(args.env, matchup, B, args.games, K, L, args.C, args.stratified, args.out)
            summaries.append(s); all_games.append(df_g); all_moves.append(df_m)

    df_summary = pd.DataFrame(summaries).sort_values(["matchup","B"])
    games_all = pd.concat(all_games, ignore_index=True)
    moves_all = pd.concat(all_moves, ignore_index=True)

    df_summary.to_csv(os.path.join(args.out, f"{args.env}_summary_all.csv"), index=False)
    moves_all.to_csv(os.path.join(args.out, f"{args.env}_moves_all.csv"), index=False)

    if args.plot:
        plot_suite(args.env, K, df_summary, moves_all, args.out)

if __name__ == "__main__":
    main()
