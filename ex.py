import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import time
from tqdm import tqdm
from mpi4py import MPI
import sys
sys.stdout.reconfigure(encoding='utf-8')



M = 1.0           
a = 0.9     


N_PHOTONS = 1000
LAMBDA_MAX = 200.0
H_INIT = 0.1        
RTOL = 1e-10       
ATOL = 1e-13       
H_MIN = 1e-8        
H_MAX = 0.5       


R_OBS = 50.0      
THETA_OBS = np.pi/2 
PHI_OBS = 0.0     


IMG_EXTENT = 25.0   
WIDTH = 100       
HEIGHT = 100       


ACCRETION_INNER = 4.0  
ACCRETION_OUTER = 20.0 
DISK_THICKNESS = 0.5 

def kerr_metric(r, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    sigma = r*r + (a * cos_theta)**2
    delta = r*r - 2.0*M*r + a*a
    if np.abs(delta) < 1e-12:
        delta = np.sign(delta) * 1e-12 if delta != 0 else 1e-12

    g = np.zeros((4, 4), dtype=float)

    g[0, 0] = -(1.0 - (2.0*M*r) / sigma)
    g[0, 3] = g[3, 0] = -(2.0*M*r*a*np.sin(theta)**2) / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = ((r*r + a*a)**2 - a*a*delta*np.sin(theta)**2) / sigma * np.sin(theta)**2

    return g

def kerr_metric_complex(r, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    sigma = r*r + (a * cos_theta)**2
    delta = r*r - 2.0*M*r + a*a

    g = np.zeros((4, 4), dtype=complex)

    g[0, 0] = -(1.0 - (2.0*M*r) / sigma)
    g[0, 3] = g[3, 0] = -(2.0*M*r*a*np.sin(theta)**2) / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = ((r*r + a*a)**2 - a*a*delta*np.sin(theta)**2) / sigma * np.sin(theta)**2

    return g

def christoffel_symbols(r, theta, h=1e-8):
    g0 = kerr_metric(r, theta)
    g_inv = np.linalg.pinv(g0)

    partial = np.zeros((4, 4, 4), dtype=float)
    g_r_complex = kerr_metric_complex(r + 1j*h, theta)
    partial[1, :, :] = np.imag(g_r_complex) / h

    g_t_complex = kerr_metric_complex(r, theta + 1j*h)
    partial[2, :, :] = np.imag(g_t_complex) / h

    Gamma = np.zeros((4, 4, 4), dtype=float)
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                s = 0.0
                for nu in range(4):
                    pa = partial[alpha, nu, beta] if alpha in (1, 2) else 0.0
                    pb = partial[beta, nu, alpha] if beta in (1, 2) else 0.0
                    pn = partial[nu, alpha, beta] if nu in (1, 2) else 0.0
                    s += g_inv[mu, nu] * (pa + pb - pn)
                Gamma[mu, alpha, beta] = 0.5 * s

    return Gamma

def geodesic_equations(lmbda, y):
    t, r, th, ph, kt, kr, kth, kph = y

    r_h = M + np.sqrt(max(0.0, M*M - a*a))

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
    B_low = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    k1 = f(lmbda, y)
    k2 = f(lmbda + c2*h, y + h * a21 * k1)
    k3 = f(lmbda + c3*h, y + h * (a31*k1 + a32*k2))
    k4 = f(lmbda + c4*h, y + h * (a41*k1 + a42*k2 + a43*k3))
    k5 = f(lmbda + c5*h, y + h * (a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(lmbda + c6*h, y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

    y5 = y + h * (B_high[0]*k1 + B_high[2]*k3 + B_high[3]*k4 + B_high[4]*k5 + B_high[5]*k6)

    k7 = f(lmbda + h, y5)

    y_high = y + h * (B_high[0]*k1 + B_high[2]*k3 + B_high[3]*k4 +
                      B_high[4]*k5 + B_high[5]*k6 + B_high[6]*k7)
    y_low = y + h * (B_low[0]*k1 + B_low[2]*k3 + B_low[3]*k4 +
                     B_low[4]*k5 + B_low[5]*k6 + B_low[6]*k7)

    err_vec = y_high - y_low
    err = np.linalg.norm(err_vec)

    return y_high, err

def integrate_geodesic(y0, l_max=LAMBDA_MAX):
    y = y0.copy()
    l = 0.0
    h = H_INIT
    traj = [y.copy()]
    steps = 0
    max_steps = 100000

    r_h = M + np.sqrt(max(0.0, M*M - a*a))

    initial_phi = y0[3]
    total_phi_change = 0.0
    last_phi = initial_phi

    while l < l_max and steps < max_steps:
        y_new, err = rk45_step(geodesic_equations, l, y, h)

        tol = ATOL + RTOL * np.linalg.norm(y)

        accept = (err <= tol) or (h <= H_MIN)

        if accept:
            y = y_new
            l += h

            current_phi = y[3]
            dphi = current_phi - last_phi
            if dphi > np.pi:
                dphi -= 2*np.pi
            elif dphi < -np.pi:
                dphi += 2*np.pi
            total_phi_change += dphi
            last_phi = current_phi
            traj.append(y.copy())
            steps += 1

            r_current = y[1]

            if r_current <= r_h * 1.001:
                break

            if r_current > R_OBS * 3.0:
                break

            if abs(total_phi_change) > 6 * np.pi:
                break

        if err == 0.0:
            fac = 2.0
        else:
            fac = 0.9 * (tol / err)**0.25
            fac = max(0.2, min(2.0, fac))

        h = np.clip(h * fac, H_MIN, H_MAX)

        if l + h > l_max:
            h = l_max - l
            if h <= 0:
                break

    return np.array(traj)

def image_plane_to_initial_state(alpha, beta, r_obs=R_OBS, theta_obs=THETA_OBS):
    t0, r0, th0, ph0 = 0.0, r_obs, theta_obs, PHI_OBS

    g = kerr_metric(r0, th0)

    fov_angle = IMG_EXTENT / R_OBS

    theta_angle = beta / R_OBS
    phi_angle = alpha / R_OBS

    E = 1.0

    cos_theta_local = np.cos(theta_angle)
    sin_theta_local = np.sin(theta_angle)
    cos_phi_local = np.cos(phi_angle)
    sin_phi_local = np.sin(phi_angle)

    kt = E
    kr = -E * cos_theta_local * cos_phi_local
    kth = E * sin_theta_local / r0
    kph = E * sin_phi_local / (r0 * np.sin(th0))

    k_contra = np.array([kt, kr, kth, kph])

    k_cov = g @ k_contra
    norm = k_contra @ k_cov

    y0 = np.array([t0, r0, th0, ph0, kt, kr, kth, kph])

    return y0, {'E': E, 'Lz': k_cov[3]}

def classify_photon_with_ring(traj):
    r = traj[:, 1]
    r_h = M + np.sqrt(max(0.0, M*M - a*a))

    min_r = r.min()
    final_r = r[-1]

    phi = traj[:, 3]
    total_rotation = 0.0
    for i in range(1, len(phi)):
        dphi = phi[i] - phi[i-1]
        if dphi > np.pi:
            dphi -= 2*np.pi
        elif dphi < -np.pi:
            dphi += 2*np.pi
        total_rotation += dphi

    total_rotations = abs(total_rotation) / (2 * np.pi)
    if total_rotations > 1.5 and min_r > r_h * 1.2 and min_r < r_h * 2.5:
        return 'photon_ring'
    elif min_r <= r_h * 1.05:
        return 'plunged'
    elif final_r >= R_OBS * 1.5:
        return 'escaped'
    else:
        return 'disk'

def check_disk_intersection_improved(traj):
    intersections = []

    for i in range(1, len(traj)):
        r = traj[i, 1]
        theta_curr = traj[i, 2]
        theta_prev = traj[i-1, 2]

        crossed_equator = (theta_prev - np.pi/2) * (theta_curr - np.pi/2) < 0

        near_equator = abs(theta_curr - np.pi/2) < DISK_THICKNESS

        if (crossed_equator or near_equator) and ACCRETION_INNER <= r <= ACCRETION_OUTER:
            intersections.append(i)

    return intersections

def disk_emission_with_doppler(r, phi, idx, traj):
    r_norm = (r - ACCRETION_INNER) / (ACCRETION_OUTER - ACCRETION_INNER)
    r_norm = np.clip(r_norm, 0.0, 1.0)

    temperature_profile = 1.0 - 0.6 * r_norm

    n_spirals = 3
    spiral_phase = n_spirals * phi - 2.0 * np.log(r / ACCRETION_INNER)
    spiral_pattern = 0.5 + 0.3 * np.cos(spiral_phase)

    turbulence_seed = np.sin(r * 17.3 + phi * 23.7) * np.cos(r * 31.1 - phi * 19.4)
    turbulence = 0.7 + 0.3 * turbulence_seed

    small_turbulence = 0.5 * np.sin(r * 50.0 + phi * 40.0) * np.cos(r * 60.0 - phi * 55.0)
    turbulence += 0.15 * small_turbulence

    shear_pattern = np.sin(phi * 8.0 + r * 5.0)
    shear_effect = 0.85 + 0.15 * shear_pattern

    radial_modulation = 1.0 + 0.2 * np.sin(r * 10.0)

    structure = spiral_pattern * turbulence * shear_effect * radial_modulation
    structure = np.clip(structure, 0.3, 1.3)

    v_disk = np.sqrt(M / r)
    v_disk = np.clip(v_disk, 0.0, 0.99)

    if idx < len(traj):
        kph = traj[idx, 7]
        kr = traj[idx, 5]
    else:
        kph = 0.0
        kr = 0.0

    if kph < 0:
        cos_theta = v_disk
    else:
        cos_theta = -v_disk

    gamma = 1.0 / np.sqrt(1.0 - v_disk**2)
    doppler_factor = 1.0 / (gamma * (1.0 - v_disk * cos_theta))

    beaming = doppler_factor ** 3.0

    base_emission = temperature_profile * structure
    emission = base_emission * beaming

    emission = np.clip(emission, 0.0, 2.5)

    if kph < 0:
        color = np.array([1.0, 0.8, 0.55])
    else:
        color = np.array([1.0, 0.7, 0.4])
    color = color / np.max(color)

    if r_norm < 0.3:
        white_hot = np.array([1.0, 0.95, 0.9])
        color = 0.4 * color + 0.6 * white_hot
    elif r_norm < 0.5:
        t = (r_norm - 0.3) / 0.2
        yellow_hot = np.array([1.0, 0.9, 0.7])
        color = (1.0 - t) * (0.4 * color + 0.6 * yellow_hot) + t * (0.6 * color + 0.4 * yellow_hot)
    else:
        color = 0.7 * color + 0.3 * np.array([1.0, 0.85, 0.7])

    color = np.clip(color, 0.0, 1.0)

    return emission, color

def photon_ring_emission(traj, min_radius_idx):
    r = traj[:, 1]
    min_r = r.min()

    r_h = M + np.sqrt(max(0.0, M*M - a*a))
    proximity = (min_r - r_h) / (r_h * 0.5)
    proximity = np.clip(proximity, 0.0, 1.0)

    base_intensity = 2.0 * (1.0 - proximity ** 0.5)

    phi = traj[:, 3]
    total_rotation = 0.0
    for i in range(1, len(phi)):
        dphi = phi[i] - phi[i-1]
        if dphi > np.pi:
            dphi -= 2*np.pi
        elif dphi < -np.pi:
            dphi += 2*np.pi
        total_rotation += dphi

    num_orbits = abs(total_rotation) / (2 * np.pi)
    orbit_boost = np.log(num_orbits + 1.0)

    intensity = base_intensity * orbit_boost
    intensity = np.clip(intensity, 0.0, 1.0)

    blue_component = 0.5 + 0.5 * (num_orbits / 3.0)
    blue_component = np.clip(blue_component, 0.0, 1.0)

    color = np.array([
        0.1 + 0.2 * (1.0 - proximity),
        0.3 + 0.3 * (1.0 - proximity),
        blue_component
    ])

    return intensity, color

def compute_redshift(traj, idx):
    if idx >= len(traj):
        return 1.0

    r = traj[idx, 1]
    theta = traj[idx, 2]

    z_grav = np.sqrt(1.0 - 2.0 * M / r)

    v_rot = np.sqrt(M / r) if r > 2*M else 0.5
    gamma = 1.0 / np.sqrt(1.0 - v_rot**2)

    return z_grav * gamma

def integrate_intensity_advanced_with_ring(result):
    traj = result['traj']

    r = traj[:, 1]
    r_h = M + np.sqrt(max(0.0, M*M - a*a))

    classification = classify_photon_with_ring(traj)

    if np.min(r) <= r_h * 1.001:
        return 0.0, np.array([0.0, 0.0, 0.0])

    if classification == 'photon_ring':
        min_r_idx = np.argmin(r)
        intensity, color = photon_ring_emission(traj, min_r_idx)
        return intensity, color

    intersections = check_disk_intersection_improved(traj)

    if len(intersections) == 0:
        return 0.005, np.array([0.03, 0.03, 0.08])

    idx = intersections[0]
    r_disk = traj[idx, 1]
    phi = traj[idx, 3]

    emission, color = disk_emission_with_doppler(r_disk, phi, idx, traj)

    if r_disk > 2*M:
        g_factor = np.sqrt(1.0 - 2.0*M/r_disk)
    else:
        g_factor = 0.3

    intensity = emission * (g_factor ** 4)

    r_isco = 6.0 * M
    if r_disk < r_isco:
        isco_boost = 1.0 + 0.3 * (1.0 - r_disk / r_isco)
        intensity *= isco_boost

    return intensity, color

def build_image_advanced_with_ring(results):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    intensity_map = np.zeros((HEIGHT, WIDTH), dtype=float)
    color_map = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

    print("Computing intensities with lensing and photon ring...")
    for result in tqdm(results, desc="Processing"):
        alpha = result['alpha']
        beta = result['beta']

        x = int((alpha + IMG_EXTENT) / (2 * IMG_EXTENT) * (WIDTH - 1))
        y = int((1 - (beta + IMG_EXTENT) / (2 * IMG_EXTENT)) * (HEIGHT - 1))

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            intensity, color = integrate_intensity_advanced_with_ring(result)
            intensity_map[y, x] = intensity
            color_map[y, x] = color

    max_intensity = np.max(intensity_map)
    if max_intensity <= 0:
        max_intensity = 1.0

    non_zero_intensities = intensity_map[intensity_map > 0]
    if len(non_zero_intensities) > 0:
        percentile_95 = np.percentile(non_zero_intensities, 95)
        print(f"Max intensity: {max_intensity:.4f}, 95th percentile: {percentile_95:.4f}")
    else:
        percentile_95 = max_intensity
        print(f"Max intensity: {max_intensity:.4f}")

    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = intensity_map[y, x]
            base_color = color_map[y, x]

            if intensity > 0:
                normalized = intensity / max_intensity

                if normalized > 0.5:
                    scaled = 0.5 + 0.5 * np.log10(1 + (normalized - 0.5) * 2 * 99) / 2.0
                elif normalized > 0.1:
                    scaled = np.log10(1 + normalized * 99) / 2.0
                else:
                    scaled = normalized * 1.5

                scaled = np.clip(scaled, 0.0, 1.0)
                img[y, x] = base_color * scaled
            else:
                img[y, x] = [0.0, 0.0, 0.0]

    img = np.clip(img, 0.0, 1.0)

    img = img ** (1.0 / 2.3)

    return img

def map_intensity_to_color(intensity, max_intensity=1.0):
    if intensity <= 0:
        return np.array([0.0, 0.0, 0.0])

    normalized = intensity / max_intensity
    normalized = np.clip(normalized, 0.0, 1.0)

    if normalized > 0:
        scaled = (np.log10(1 + normalized * 99) / 2.0)
    else:
        scaled = 0.0

    scaled = np.clip(scaled, 0.0, 1.0)

    if scaled < 0.2:
        t = scaled * 5
        r = t * 0.3
        g = 0.0
        b = 0.0
    elif scaled < 0.4:
        t = (scaled - 0.2) * 5
        r = 0.3 + t * 0.4
        g = t * 0.1
        b = 0.0
    elif scaled < 0.6:
        t = (scaled - 0.4) * 5
        r = 0.7 + t * 0.3
        g = 0.1 + t * 0.4
        b = t * 0.05
    elif scaled < 0.8:
        t = (scaled - 0.6) * 5
        r = 1.0
        g = 0.5 + t * 0.4
        b = 0.05 + t * 0.15
    else:
        t = (scaled - 0.8) * 5
        r = 1.0
        g = 0.9 + t * 0.1
        b = 0.2 + t * 0.8

    return np.array([r, g, b])

def build_image(results):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    intensity_map = np.zeros((HEIGHT, WIDTH), dtype=float)

    print("Computing intensities...")
    for result in tqdm(results, desc="Computing"):
        alpha = result['alpha']
        beta = result['beta']

        x = int((alpha + IMG_EXTENT) / (2 * IMG_EXTENT) * (WIDTH - 1))
        y = int((1 - (beta + IMG_EXTENT) / (2 * IMG_EXTENT)) * (HEIGHT - 1))

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            intensity, _ = integrate_intensity_advanced_with_ring(result)
            intensity_map[y, x] = intensity

    max_intensity = np.max(intensity_map)
    if max_intensity <= 0:
        max_intensity = 1.0

    print(f"Max intensity: {max_intensity:.4f}")

    print("Mapping colors...")
    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = intensity_map[y, x]
            color = map_intensity_to_color(intensity, max_intensity)
            img[y, x] = color

    img = np.clip(img, 0.0, 1.0)
    img = img ** (1.0 / 2.2)

    return img

def add_bloom(img, radius=5, strength=0.2):
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    bright_threshold = 0.6
    bloom_mask = np.max(img, axis=2) > bright_threshold

    if not np.any(bloom_mask):
        return img

    bloom_img = img.copy()
    bloom_img[~bloom_mask] = 0

    bloom_uint8 = (bloom_img * 255).astype(np.uint8)
    bloom_pil = Image.fromarray(bloom_uint8)

    bloom_blurred = bloom_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    result = Image.blend(pil_img, bloom_blurred, alpha=strength)

    return np.array(result) / 255.0


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def worker_trace(alpha_beta_list):
    results_local = []
    for alpha, beta in alpha_beta_list:
        y0, consts = image_plane_to_initial_state(alpha, beta)
        traj = integrate_geodesic(y0)
        results_local.append({
            'alpha': alpha,
            'beta': beta,
            'traj': traj,
            'consts': consts
        })
    return results_local

def distribute_rays_and_trace(n_photons):

    n_side = int(np.sqrt(n_photons))
    if n_side < 1:
        n_side = 1
    alphas = np.linspace(-IMG_EXTENT, IMG_EXTENT, n_side)
    betas = np.linspace(-IMG_EXTENT, IMG_EXTENT, n_side)
    all_pairs = [(alpha, beta) for alpha in alphas for beta in betas]

    # Split the list into `size` chunks as evenly as possible
    if rank == 0:
        chunks = []
        base = len(all_pairs) // size
        rem = len(all_pairs) % size
        start = 0
        for i in range(size):
            extra = 1 if i < rem else 0
            end = start + base + extra
            chunks.append(all_pairs[start:end])
            start = end
    else:
        chunks = None

    my_chunk = comm.scatter(chunks, root=0)

    local_results = worker_trace(my_chunk)

    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        results = [item for sublist in gathered for item in sublist]
        return results
    else:
        return None

def main_mpi():
    if rank == 0:
        print("=" * 70)
        print("BLACK HOLE RENDERER - MPI PARALLEL (mpi4py)")
        print("=" * 70)
        print(f"Running with MPI size = {size}")
        print(f"Black hole parameters: M={M}, a={a}")
        print(f"Observer position: r={R_OBS}, θ={THETA_OBS:.2f}")
        print(f"Image size: {WIDTH}x{HEIGHT}")
        print(f"Field of view: ±{IMG_EXTENT}M")
        print("=" * 70)

    comm.Barrier()
    start_time = time.time()

    results = distribute_rays_and_trace(N_PHOTONS)

    sim_time = time.time() - start_time
    if rank == 0:
        print(f"\nRay tracing completed in {sim_time:.2f} seconds (wall time)")
        outcomes = {'plunged': 0, 'escaped': 0, 'disk': 0, 'photon_ring': 0}
        disk_hits = 0

        for result in results:
            outcome = classify_photon_with_ring(result['traj'])
            outcomes[outcome] += 1

            intersections = check_disk_intersection_improved(result['traj'])
            if len(intersections) > 0:
                disk_hits += 1

        print("\nPhoton outcomes:")
        for outcome, count in outcomes.items():
            percentage = 100 * count / len(results)
            print(f"  {outcome}: {count} ({percentage:.1f}%)")
        print(f"  Disk intersections: {disk_hits} ({100*disk_hits/len(results):.1f}%)")

        print("\nBuilding image with enhanced features...")
        img = build_image_advanced_with_ring(results)
        img = add_bloom(img, radius=4, strength=0.18)

        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img.save('black_hole_enhanced_mpi.png')
        print("\n✓ Image saved as 'black_hole_enhanced_mpi.png'")

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Kerr Black Hole - Enhanced Rendering (MPI)', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig('black_hole_display_enhanced_mpi.png', dpi=150, bbox_inches='tight')
        plt.show()

    # finalize
    comm.Barrier()
    if rank == 0:
        print("\nMPI run complete.")

if __name__ == "__main__":
    main_mpi()