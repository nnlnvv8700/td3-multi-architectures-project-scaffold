# utils/trajectory_planner.py
import numpy as np

def _poly5_coeff(p0, v0, a0, pf, vf, af, T):
    # Quintic polynomial boundary conditions
    T2, T3, T4, T5 = T*T, T**3, T**4, T**5
    A = np.array([
        [1,    T,    T2,     T3,      T4,       T5],
        [0,    1,   2*T,   3*T2,    4*T3,     5*T4],
        [0,    0,     2,     6*T,   12*T2,    20*T3],
        [1,    0,     0,       0,      0,        0],
        [0,    1,     0,       0,      0,        0],
        [0,    0,     2,       0,      0,        0],
    ], dtype=float)
    b = np.array([pf, vf, af, p0, v0, a0], dtype=float)
    # Reorder rows to solve for a0..a5
    # We solve with constraints at t=0, t=T; rewrite to standard form:
    # Here we directly construct the solution for a0..a5:
    # a0 = p0, a1 = v0, a2 = a0/2, (use linear system for a3..a5)
    a0c = p0
    a1c = v0
    a2c = a0/2.0
    M = np.array([
        [  T**3,    T**4,     T**5],
        [3*T**2,  4*T**3,   5*T**4],
        [  6*T,   12*T**2,  20*T**3],
    ], dtype=float)
    y = np.array([
        pf - (a0c + a1c*T + a2c*T**2),
        vf - (a1c + 2*a2c*T),
        af - (2*a2c),
    ], dtype=float)
    a3c, a4c, a5c = np.linalg.solve(M, y)
    return np.array([a0c, a1c, a2c, a3c, a4c, a5c], dtype=float)

def quintic_trajectory(p_start, p_goal, T=3.0, dt=0.04):
    """五次多项式平滑轨迹（默认零始末速度/加速度）"""
    p_start = np.asarray(p_start, dtype=float).reshape(3,)
    p_goal  = np.asarray(p_goal,  dtype=float).reshape(3,)
    N = max(2, int(np.round(T/dt)))
    t = np.linspace(0.0, T, N)
    traj = np.zeros((N, 3), dtype=float)
    for i in range(3):
        a = _poly5_coeff(p_start[i], 0.0, 0.0, p_goal[i], 0.0, 0.0, T)
        # p(t) = a0 + a1 t + a2 t^2 + ... + a5 t^5
        traj[:, i] = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    return traj

def minimum_jerk_trajectory(p_start, p_goal, T=3.0, dt=0.04):
    """最小加加速度（minimum-jerk）轨迹：p(t)=p0+(pf-p0)*(10 s^3 -15 s^4 +6 s^5)"""
    p_start = np.asarray(p_start, dtype=float).reshape(3,)
    p_goal  = np.asarray(p_goal,  dtype=float).reshape(3,)
    N = max(2, int(np.round(T/dt)))
    t = np.linspace(0.0, T, N)
    s = t / T
    s_poly = 10*s**3 - 15*s**4 + 6*s**5
    traj = p_start + (p_goal - p_start) * s_poly[:, None]
    return traj

def path_length(points):
    """路径长度（L1..LN 的折线长度）"""
    points = np.asarray(points, dtype=float)
    if len(points) < 2: return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
