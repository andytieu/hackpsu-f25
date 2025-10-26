import numpy as np
import matplotlib.pyplot as plt
import math

M = 1.0            
a = 0.5            
N_PHOTONS = 50
LAMBDA_MAX = 200.0
H_INIT = .1      
RTOL = 1e-10     
ATOL = 1e-13        
H_MIN = 1e-8 
H_MAX = .1

R_OBS = 50.0  
THETA_OBS = np.pi/2 
IMG_EXTENT = 5.0      

PLOT_RADIUS = 20.0


def kerr_metric(r, theta):
    sigma = r*r + (a * np.cos(theta))**2
    delta = r*r - 2.0*M*r + a*a
    if np.isrealobj(r) and abs(delta) < 1e-12:
        delta = np.sign(delta) * 1e-12
    g = np.zeros((4,4), dtype=float)
    g[0,0] = -(1.0 - (2.0*M*r) / sigma)
    g[0,3] = g[3,0] = - (2.0*M*r*a*np.sin(theta)**2) / sigma
    g[1,1] = sigma / delta
    g[2,2] = sigma
    g[3,3] = ( (r*r + a*a) + (2.0*M*r*a*a*np.sin(theta)**2) / sigma ) * np.sin(theta)**2
    return g

def kerr_metric_complex(r, theta):
    sigma = r*r + (a * np.cos(theta))**2
    delta = r*r - 2.0*M*r + a*a
    g = np.zeros((4,4), dtype=complex)
    g[0,0] = -(1.0 - (2.0*M*r) / sigma)
    g[0,3] = g[3,0] = - (2.0*M*r*a*np.sin(theta)**2) / sigma
    g[1,1] = sigma / delta
    g[2,2] = sigma
    g[3,3] = ( (r*r + a*a) + (2.0*M*r*a*a*np.sin(theta)**2) / sigma ) * np.sin(theta)**2
    return g

def christoffel_symbols(r, theta, h=1e-8):
    g0 = kerr_metric(r, theta)            
    g_inv = np.linalg.inv(g0)
    partial = np.zeros((4,4,4), dtype=float)
    g_r_complex = kerr_metric_complex(r + 1j*h, theta) 
    partial[1,:,:] = np.imag(g_r_complex) / h
    g_t_complex = kerr_metric_complex(r, theta + 1j*h)
    partial[2,:,:] = np.imag(g_t_complex) / h
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
            r_current = y[1]
            r_h = M + np.sqrt(max(0.0, M*M - a*a))
            if r_current <= r_h * 1.01: 
                break
            if r_current > R_OBS * 1.5: 
                break
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


def build_local_tetrad(r_obs, th_obs, a_local=a, M_local=M):
    g = kerr_metric(r_obs, th_obs)
    
    g_tphi = g[0,3]
    g_phiphi = g[3,3]
    omega = -g_tphi / g_phiphi 
    
    u_t = 1.0
    u_phi = omega
    
    norm_sq = g[0,0] * u_t**2 + 2*g[0,3] * u_t * u_phi + g[3,3] * u_phi**2
    if norm_sq >= 0:
        u_t = 1.0 / np.sqrt(max(1e-12, -g[0,0]))
        u = np.array([u_t, 0.0, 0.0, 0.0])
    else:
        A = 1.0 / np.sqrt(-norm_sq)
        u = A * np.array([u_t, 0.0, 0.0, u_phi])

    e_r_coord = np.array([0.0, 1.0, 0.0, 0.0])
    norm_r = np.sqrt(abs(e_r_coord @ g @ e_r_coord))
    e_r = e_r_coord / norm_r
    
    e_th_coord = np.array([0.0, 0.0, 1.0, 0.0])
    norm_th = np.sqrt(abs(e_th_coord @ g @ e_th_coord))
    e_th = e_th_coord / norm_th
    
    e_phi_coord = np.array([0.0, 0.0, 0.0, 1.0])
    proj_u = (e_phi_coord @ g @ u) * u
    e_phi_orth = e_phi_coord - proj_u
    norm_phi = np.sqrt(abs(e_phi_orth @ g @ e_phi_orth))
    e_phi = e_phi_orth / norm_phi
    
    return np.vstack([u, e_r, e_th, e_phi])


def image_plane_to_initial_state(alpha, beta, r_obs=R_OBS, theta_obs=THETA_OBS):
    t0, r0, th0, ph0 = 0.0, r_obs, theta_obs, 0.0
    
    tetrad = build_local_tetrad(r0, th0)
    u = tetrad[0]    
    e_r = tetrad[1]     
    e_th = tetrad[2]    
    e_phi = tetrad[3]  
    
    n_local = -e_r + (alpha / r0) * e_phi + (beta / r0) * e_th
    
    g = kerr_metric(r0, th0)
    norm_sq = n_local @ g @ n_local
    n_local = n_local / np.sqrt(abs(norm_sq))
    
    k_contra = u + n_local
    
    y0 = np.array([t0, r0, th0, ph0,
                   k_contra[0], k_contra[1], k_contra[2], k_contra[3]], dtype=float)
    

    p_cov = g @ k_contra
    E_cons = -p_cov[0]
    Lz = p_cov[3]
    
    cos_th = np.cos(th0)
    sin_th = np.sin(th0)
    if abs(sin_th) < 1e-10:
        sin_th = 1e-10
    Q = (p_cov[2]**2) + (cos_th**2) * ((Lz**2)/(sin_th**2) - (a**2 * E_cons**2))
    
    consts = {'E_cons': float(E_cons), 'Lz': float(Lz), 'Q': float(Q)}
    
    return y0, consts


def radial_potential(r, E, Lz, Q, a_local=a, M_local=M):
    Delta = r*r - 2.0*M_local*r + a_local*a_local
    term1 = E * (r*r + a_local*a_local) - a_local * Lz
    term1 = term1**2
    term2 = Delta * ( Q + (Lz - a_local*E)**2 )
    return term1 - term2


def find_radial_root_bisection(r_lo, r_hi, E, Lz, Q, a_local=a, M_local=M, tol=1e-8, maxiter=50):
    Rlo = radial_potential(r_lo, E, Lz, Q, a_local, M_local)
    Rhi = radial_potential(r_hi, E, Lz, Q, a_local, M_local)
    if Rlo == 0.0:
        return r_lo
    if Rhi == 0.0:
        return r_hi
    if Rlo * Rhi > 0:
        return None
    for _ in range(maxiter):
        rm = 0.5*(r_lo + r_hi)
        Rm = radial_potential(rm, E, Lz, Q, a_local, M_local)
        if abs(Rm) < tol:
            return rm
        if Rlo * Rm < 0:
            r_hi, Rhi = rm, Rm
        else:
            r_lo, Rlo = rm, Rm
    return 0.5*(r_lo + r_hi)

def classify_photons(results, r_obs=R_OBS):
    r_h = M + np.sqrt(max(0.0, M*M - a*a))
    r_escape_thresh = 1.5 * r_obs  # choose a large enough radius
    for entry in results:
        traj = entry['traj']
        rs = traj[:,1]
        if np.any(rs <= r_h * 1.001):
            entry['outcome'] = 'plunged'
        elif rs[-1] >= r_escape_thresh:
            entry['outcome'] = 'escaped'
        else:
            entry['outcome'] = 'other'
    return results


def simulate_photons(N_photons=N_PHOTONS, img_extent=IMG_EXTENT):
    n = int(np.ceil(np.sqrt(N_photons)))
    alphas = np.linspace(-img_extent, img_extent, n)
    betas  = np.linspace(-img_extent, img_extent, n)
    
    coords = []
    for b in betas:
        for a_val in alphas:
            if len(coords) < N_photons:
                coords.append((a_val, b))

    results = []
    for (alpha, beta) in coords:
        y0, consts = image_plane_to_initial_state(alpha, beta)
        E  = consts['E_cons']
        Lz = consts['Lz']
        Q  = consts['Q']
        r_initial = y0[1]
        r_turn = find_radial_root_bisection(1.001*M, r_initial, E, Lz, Q)
        entry = {'alpha': alpha, 'beta': beta, 'consts': consts}
        if r_turn is not None:
            entry['predicted_turn'] = r_turn

        entry['traj'] = integrate_rk45(lambda l, yy: geodesic_equations(l, yy, M, a),
                                    y0, 0.0, LAMBDA_MAX)
        results.append(entry)

    return results


def check_null_constraint(traj, tol=1e-6):
    maxv = 0.0
    for row in traj:
        r = float(row[1]); th = float(row[2])
        k = np.array([row[4], row[5], row[6], row[7]], dtype=float)
        g = kerr_metric(r, th)
        s = float(k @ (g @ k))
        maxv = max(maxv, abs(s))
    return (maxv <= tol, maxv)


def check_conserved_quantities(traj, consts, tol_rel=1e-4):
    E0 = consts.get('E_cons', None)
    Lz0 = consts.get('Lz', None)
    if E0 is None or Lz0 is None:
        return (False, np.inf)
    max_drift = 0.0
    for row in traj:
        r = float(row[1]); th = float(row[2])
        k = np.array([row[4], row[5], row[6], row[7]], dtype=float)
        p_cov = kerr_metric(r, th) @ k
        E = -float(p_cov[0])
        Lz = float(p_cov[3])
        drift = 0.0
        if E0 != 0:
            drift = max(drift, abs((E - E0) / E0))
        if Lz0 != 0:
            drift = max(drift, abs((Lz - Lz0) / (Lz0 if Lz0 != 0 else 1.0)))
        max_drift = max(max_drift, drift)
    return (max_drift <= tol_rel, max_drift)


def detect_reemergence(traj, M_local=M, a_local=a):
    r_h = M_local + math.sqrt(max(0.0, M_local*M_local - a_local*a_local))
    rs = traj[:,1]
    dipped = np.any(rs <= r_h * 1.001)
    if not dipped:
        return False
    idx = np.where(rs <= r_h * 1.001)[0]
    if idx.size == 0:
        return False
    first_dip = int(idx[0])
    if np.any(rs[first_dip+1:] > r_h * 1.001):
        return True
    return False


def force_mark_plunged(entry):
    entry['outcome'] = 'plunged'
    traj = entry.get('traj')
    if traj is None or len(traj) == 0:
        return entry
    r_h = M + math.sqrt(max(0.0, M*M - a*a))
    rs = traj[:,1]
    idx = np.where(rs <= r_h * 1.001)[0]
    if idx.size > 0:
        cut_at = int(idx[0]) + 1
        entry['traj'] = traj[:cut_at].copy()
    return entry


def validate_and_fix_entry(entry, null_tol=1e-5, cons_tol=1e-3):
    traj = entry.get('traj')
    if traj is None or len(traj) == 0:
        return entry, 'discarded', {'reason': 'no_traj'}
    report = {}
    ok_null, max_null = check_null_constraint(traj, tol=null_tol)
    report['max_null_violation'] = max_null
    ok_cons, max_cons = check_conserved_quantities(traj, entry.get('consts', {}), tol_rel=cons_tol)
    report['max_cons_drift'] = max_cons
    reem = detect_reemergence(traj)
    report['reemergence'] = reem

    if reem:
        entry = force_mark_plunged(entry)
        return entry, 'fixed_plunged', report
    if not ok_null or not ok_cons:
        if (max_null < 10.0 * null_tol) and (max_cons < 10.0 * cons_tol):
            return entry, 'ok_but_warn', report
        else:
            return entry, 'discarded', report
    return entry, 'ok', report


def find_asymptotic_index(traj):
    return -1 


def asymptotic_sky_direction(pt):
    return {'theta': np.pi/2, 'phi': 0.0}


def redshift_and_intensity(E_cons, k_contra, r, theta):
    return 1.0, 0.0, 1.0, float(E_cons)


def default_background_sampler(theta, phi):
    return np.array([0.0, 0.0, 0.0], dtype=float)


def apply_redshift_rgb(rgb, g_factor):
    intensity_scale = max(0.0, g_factor**3)
    return np.clip(rgb * intensity_scale, 0.0,1.0)


def build_image_from_results(results,width,height, img_extent=IMG_EXTENT, background_sampler=default_background_sampler,supersamples=1,fill_black_for_plunged=True):
    img = np.zeros((height,width,3),dtype=np.float32)
    coords = [(float(r['alpha']), float(r['beta'])) for r in results]
    alphas = sorted(set([c[0] for c in coords]))
    betas  = sorted(set([c[1] for c in coords]))
    keymap = {(float(e['alpha']), float(e['beta'])): e for e in results}
    
    def alpha_to_x(a): return int(np.clip(((a + img_extent)/(2*img_extent)) * (width-1),0, width-1))
    def beta_to_y(b): return int(np.clip((1 - ((b+img_extent) / (2*img_extent)))*(height-1), 0, height -1))
    
    count = np.zeros((height, width), dtype=int)
    accum = np.zeros((height, width, 3), dtype=float)

    for entry in results:
        x = alpha_to_x(entry['alpha'])
        y = beta_to_y(entry['beta'])
        outcome = entry.get('outcome', 'other')
        if outcome == 'plunged':
            color = np.array([0.0,0.0,0.0], dtype=float) if fill_black_for_plunged else np.zeros(3)
        elif outcome == 'escaped':
            color = default_background_sampler(np.pi/2,0.0)
        else:
            color = np.array([0.05,0.08,0.2], dtype=float)
        accum[y,x,:] += color
        count[y,x] += 1
    
    nonzero = count > 0
    img[nonzero,:,:] = (accum[nonzero,:,:].reshape(-1,3) / count[nonzero].reshape(-1,1)).reshape(accum[nonzero,:,:].shape)
    img = np.clip(img, 0.0, 1.0)
    return img

def bl_to_cartesian(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return np.array([x,y,z])


if __name__ == "__main__":
    print("Simulating", N_PHOTONS, "photons with IMG_EXTENT=", IMG_EXTENT)
    results = simulate_photons(N_PHOTONS)
    
    def classify_and_report(results, r_h=None, r_obs=R_OBS, r_escape_thresh=0.9*R_OBS):
        if r_h is None:
            r_h = M + np.sqrt(max(0.0, M*M - a*a))
        stats = {'plunged':0, 'escaped':0, 'other':0}
        classified = []
        for res in results:
            traj = res['traj']
            r = traj[:,1]
            min_r = r.min()
            final_r = r[-1]
            if min_r <= r_h * 1.001:
                outcome = 'plunged'
            elif final_r >= r_escape_thresh:
                outcome = 'escaped'
            else:
                outcome = 'other'
            stats[outcome] += 1
            classified.append((res, outcome))
        print("Outcome counts:", stats)
        return classified

    classified = classify_and_report(results)
    
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    r_h = M + np.sqrt(max(0.0, M*M - a*a))
    ax.add_artist(plt.Circle((0,0), r_h, color='k', label='Event Horizon'))
    
    colors = {'plunged':'tab:red', 'escaped':'tab:green', 'other':'tab:blue'}
    labels_added = set()
    
    for res, outcome in classified:
        traj = res['traj']
        r = traj[:,1]
        phi = traj[:,3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        mask = (r < PLOT_RADIUS)
        label = outcome.capitalize() if outcome not in labels_added else None
        if label:
            labels_added.add(outcome)
        plt.plot(x[mask], y[mask], linewidth=0.8, color=colors[outcome], alpha=0.8, label=label)
    
    plt.xlabel("x (M)")
    plt.ylabel("y (M)")
    plt.title(f"Photon geodesics colored by outcome (a={a})")
    plt.axis("equal")
    plt.xlim(-PLOT_RADIUS, PLOT_RADIUS)
    plt.ylim(-PLOT_RADIUS, PLOT_RADIUS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()