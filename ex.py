import numpy as np
import matplotlib.pyplot as plt

M = 1.0            
a = 0.5            
N_PHOTONS = 150
LAMBDA_MAX = 150.0
H_INIT = 0.01        
RTOL = 1e-6          
ATOL = 1e-9        
H_MIN = 1e-6         
H_MAX = 0.1

R_OBS = 50.0  
THETA_OBS = np.pi/2 
IMG_EXTENT = 5.0      

PLOT_RADIUS = 20.0



def kerr_metric(r, theta):
    sigma = r*r + (a * np.cos(theta))**2
    delta = r*r - 2.0*M*r + a*a
    if abs(delta) < 1e-12:
        delta = np.sign(delta) * 1e-12
    g = np.zeros((4,4), dtype=float)
    g[0,0] = -(1.0 - (2.0*M*r) / sigma)
    g[0,3] = g[3,0] = - (2.0*M*r*a*np.sin(theta)**2) / sigma
    g[1,1] = sigma / delta
    g[2,2] = sigma
    g[3,3] = ( (r*r + a*a) + (2.0*M*r*a*a*np.sin(theta)**2) / sigma ) * np.sin(theta)**2
    return g


def christoffel_symbols(r, theta, h=1e-6):
    g0 = kerr_metric(r, theta)
    g_inv = np.linalg.inv(g0)

    partial = np.zeros((4,4,4), dtype=float)
    g_pr = kerr_metric(r + h, theta)
    g_mr = kerr_metric(r - h, theta)
    partial[1,:,:] = (g_pr - g_mr) / (2.0*h)
    g_pt = kerr_metric(r, theta + h)
    g_mt = kerr_metric(r, theta - h)
    partial[2,:,:] = (g_pt - g_mt) / (2.0*h)

    Gamma = np.zeros((4,4,4), dtype=float)
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                s = 0.0
                for nu in range(4):
                    pa = partial[alpha, nu, beta] if alpha in (1,2) else 0.0
                    pb = partial[beta, nu, alpha] if beta in (1,2) else 0.0
                    pn = partial[nu, alpha, beta] if nu in (1,2) else 0.0
                    s += g_inv[mu, nu] * (pa + pb - pn)
                Gamma[mu, alpha, beta] = 0.5 * s
    return Gamma


def geodesic_equations(lmbda, y, M_local=M, a_local=a):
    t, r, th, ph, kt, kr, kth, kph = y
    r_h = M_local + np.sqrt(max(0.0, M_local*M_local - a_local*a_local))
    if r <= r_h + 1e-8 or r <= 0.0:
        return np.zeros_like(y)
    Gamma = christoffel_symbols(r, th)
    dx = np.array([kt, kr, kth, kph], dtype=float)
    k = dx.copy()
    dk = np.zeros(4, dtype=float)
    for mu in range(4):
        s = 0.0
        for alpha in range(4):
            for beta in range(4):
                s += Gamma[mu, alpha, beta] * k[alpha] * k[beta]
        dk[mu] = -s
    return np.concatenate((dx, dk))


def rk45_step(f, lmbda, y, h):
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    B_high = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
    B_low  = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    k1 = f(lmbda, y)
    k2 = f(lmbda + c2*h, y + h * a21 * k1)
    k3 = f(lmbda + c3*h, y + h * (a31*k1 + a32*k2))
    k4 = f(lmbda + c4*h, y + h * (a41*k1 + a42*k2 + a43*k3))
    k5 = f(lmbda + c5*h, y + h * (a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(lmbda + c6*h, y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

    y5 = y + h * (B_high[0]*k1 + B_high[2]*k3 + B_high[3]*k4 + B_high[4]*k5 + B_high[5]*k6)
    k7 = f(lmbda + h, y5)

    y_high = y + h * (B_high[0]*k1 + B_high[2]*k3 + B_high[3]*k4 + B_high[4]*k5 + B_high[5]*k6 + B_high[6]*k7)
    y_low  = y + h * (B_low[0]*k1  + B_low[2]*k3  + B_low[3]*k4  + B_low[4]*k5  + B_low[5]*k6  + B_low[6]*k7)

    err_vec = y_high - y_low
    err = np.linalg.norm(err_vec)
    return y_high, err


def integrate_rk45(f, y0, l0, l_max):
    y = y0.copy()
    l = l0
    h = H_INIT
    traj = [y.copy()]
    steps = 0
    while l < l_max and steps < 500000:
        y_new, err = rk45_step(f, l, y, h)
        tol = ATOL + RTOL * np.linalg.norm(y)
        accept = (err <= tol) or (h <= H_MIN)
        if accept:
            y = y_new
            l += h
            traj.append(y.copy())
            steps += 1
        if err == 0.0:
            fac = 5.0
        else:
            fac = 0.9 * (tol / err)**0.25
            fac = max(0.2, min(5.0, fac))
        h = np.clip(h * fac, H_MIN, H_MAX)
        if l + h > l_max:
            h = l_max - l
            if h <= 0:
                break
    return np.array(traj)


def image_plane_to_initial_state(alpha, beta, r_obs=R_OBS, theta_obs=THETA_OBS, E=1.0):
    t0 = 0.0
    r0 = r_obs
    th0 = theta_obs
    ph0 = 0.0

    n_r = -1.0
    n_th = beta / r0
    n_ph = alpha / (r0 * np.sin(th0) + 1e-16)

    kt0 = E
    kr0 = n_r
    kth0 = n_th
    kph0 = n_ph

    g = kerr_metric(r0, th0)

    g_tt = g[0,0]
    g_ti = np.array([g[0,1], g[0,2], g[0,3]])
    g_ij = np.array([[g[1,1], g[1,2], g[1,3]],
                     [g[2,1], g[2,2], g[2,3]],
                     [g[3,1], g[3,2], g[3,3]]])
    svec = np.array([kr0, kth0, kph0])
    A = float(svec @ (g_ij @ svec))
    B = 2.0 * float(kt0 * (g_ti @ svec))
    C = float(g_tt * kt0 * kt0)
    s_val = 0.0
    if abs(A) < 1e-16:
        if abs(B) > 1e-16:
            s_val = -C / B
        else:
            s_val = 1.0
    else:
        disc = B*B - 4*A*C
        if disc < 0:
            s_val = 1.0
        else:
            r1 = (-B + np.sqrt(disc)) / (2.0*A)
            r2 = (-B - np.sqrt(disc)) / (2.0*A)
            cand = [r1, r2]
            chosen = None
            for c in cand:
                if (c * kr0) < 0:
                    chosen = c
                    break
            if chosen is None:
                chosen = cand[0]
            s_val = chosen

    kr = s_val * kr0
    kth = s_val * kth0
    kph = s_val * kph0
    y0 = np.array([t0, r0, th0, ph0, kt0, kr, kth, kph], dtype=float)
    return y0


def simulate_photons(N_photons=N_PHOTONS, img_extent=IMG_EXTENT):
    n = int(np.ceil(np.sqrt(N_photons)))
    alphas = np.linspace(-img_extent, img_extent, n)
    betas  = np.linspace(-img_extent, img_extent, n)
    
    coords = []
    for b in betas:
        for a in alphas:
            if len(coords) < N_photons:
                coords.append((a, b))

    results = []
    for (alpha, beta) in coords:
        y0 = image_plane_to_initial_state(alpha, beta)
        traj = integrate_rk45(lambda l, yy: geodesic_equations(l, yy, M, a), 
                             y0, 0.0, LAMBDA_MAX)
        results.append({'alpha':alpha, 'beta':beta, 'traj':traj})
    
    return results


def plot_geodesics(results, plot_radius=PLOT_RADIUS):
    plt.figure(figsize=(6,6))
    r_h = M + np.sqrt(max(0.0, M*M - a*a))
    ax = plt.gca()
    ax.add_artist(plt.Circle((0,0), r_h, color='k'))

    for res in results:
        traj = res['traj']
        r = traj[:,1]
        phi = traj[:,3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        mask = (r < plot_radius)
        plt.plot(x[mask], y[mask], linewidth=0.8)

    plt.xlabel("x (M)")
    plt.ylabel("y (M)")
    plt.title(f"Photon geodesics (N={len(results)}, a={a})")
    plt.axis("equal")
    plt.xlim(-plot_radius, plot_radius)
    plt.ylim(-plot_radius, plot_radius)
    plt.show()


if __name__ == "__main__":
    print("Simulating", N_PHOTONS, "photons with IMG_EXTENT=", IMG_EXTENT)
    results = simulate_photons(N_PHOTONS)
    
    # DEBUGGING: Check what happened to each photon
    print("\nPhoton trajectory summary:")
    for i, res in enumerate(results):
        traj = res['traj']
        r_start = traj[0, 1]
        r_end = traj[-1, 1]
        n_points = len(traj)
        r_min = np.min(traj[:, 1])
        mask = (traj[:, 1] < PLOT_RADIUS)
        n_visible = np.sum(mask)
        print(f"Photon {i}: α={res['alpha']:6.2f}, β={res['beta']:6.2f} | "
              f"r: {r_start:.1f}→{r_end:.1f} | min_r={r_min:.2f} | "
              f"pts={n_points} | visible_pts={n_visible}")
    
    plot_geodesics(results)
