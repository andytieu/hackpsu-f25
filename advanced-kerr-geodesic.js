/**
 * Advanced Kerr Geodesic Solver
 * Implementation of the Python geodesic solver for accurate photon paths
 */

class AdvancedKerrGeodesic {
    constructor(mass, spin) {
        this.M = mass;
        this.a = spin;
    }
    /**
     * Calculate Kerr metric components
     */
    kerrMetric(r, theta) {
        const sigma = r * r + (this.a * Math.cos(theta)) ** 2;
        const delta = r * r - 2.0 * this.M * r + this.a * this.a;
        
        // Handle case where delta is very close to zero
        const delta_adj = Math.abs(delta) < 1e-12 ? Math.sign(delta) * 1e-12 : delta;
        
        const g = {
            0: { 0: -(1.0 - (2.0 * this.M * r) / sigma) },
            1: { 1: sigma / delta_adj },
            2: { 2: sigma },
            3: { 3: ((r * r + this.a * this.a) + (2.0 * this.M * r * this.a * this.a * Math.sin(theta) ** 2) / sigma) * Math.sin(theta) ** 2 }
        };
        
        g[0][3] = g[3][0] = -(2.0 * this.M * r * this.a * Math.sin(theta) ** 2) / sigma;
        
        return g;
    }
    
    /**
     * Calculate Christoffel symbols numerically
     */

    christoffelSymbols(r, theta, h = 1e-8) {
        const g0 = this.kerrMetric(r, theta);
        const g_inv = this.inverseMetric(g0);
        
        // Numerical derivatives using complex step
        const partial = {};
        
        // Derivative with respect to r
        const g_r_plus = this.kerrMetric(r + h, theta);
        const g_r_minus = this.kerrMetric(r - h, theta);
        partial[1] = this.numericalDerivative(g_r_plus, g_r_minus, h);
        
        // Derivative with respect to theta
        const g_t_plus = this.kerrMetric(r, theta + h);
        const g_t_minus = this.kerrMetric(r, theta - h);
        partial[2] = this.numericalDerivative(g_t_plus, g_t_minus, h);
        
        // Calculate Christoffel symbols
        const Gamma = {};
        for (let mu = 0; mu < 4; mu++) {
            for (let alpha = 0; alpha < 4; alpha++) {
                for (let beta = 0; beta < 4; beta++) {
                    let s = 0.0;
                    for (let nu = 0; nu < 4; nu++) {
                        const pa = partial[alpha] && partial[alpha][`${nu},${beta}`] || 0.0;
                        const pb = partial[beta] && partial[beta][`${nu},${alpha}`] || 0.0;
                        const pn = partial[nu] && partial[nu][`${alpha},${beta}`] || 0.0;
                        s += g_inv[mu][nu] * (pa + pb - pn);
                    }
                    if (!Gamma[mu]) Gamma[mu] = {};
                    if (!Gamma[mu][alpha]) Gamma[mu][alpha] = {};
                    Gamma[mu][alpha][beta] = 0.5 * s;
                }
            }
        }
        
        return Gamma;
    }
    
    /**
     * Calculate numerical derivative
     */
    numericalDerivative(g_plus, g_minus, h) {
        const deriv = {};
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                const key = `${i},${j}`;
                deriv[key] = (this.getMetricValue(g_plus, i, j) - this.getMetricValue(g_minus, i, j)) / (2 * h);
            }
        }
        return deriv;
    }
    
    /**
     * Get metric value at indices
     */
    getMetricValue(g, i, j) {
        return g[i] && g[i][j] || 0.0;
    }
    
    /**
     * Calculate inverse metric
     */
    inverseMetric(g) {
        // For simplicity, return diagonal approximation
        const g_inv = {};
        for (let i = 0; i < 4; i++) {
            g_inv[i] = {};
            for (let j = 0; j < 4; j++) {
                const val = this.getMetricValue(g, i, j);
                g_inv[i][j] = (i === j && val !== 0) ? 1.0 / val : 0.0;
            }
        }
        return g_inv;
    }
    
    /**
     * Geodesic equations
     */
    geodesicEquations(lambda, y) {
        const t = y[0];
        const r = y[1];
        const th = y[2];
        const ph = y[3];
        const kt = y[4];
        const kr = y[5];
        const kth = y[6];
        const kph = y[7];
        
        // Check if inside event horizon
        const r_h = this.M + Math.sqrt(Math.max(0.0, this.M * this.M - this.a * this.a));
        if (r <= r_h + 1e-8 || r <= 0.0) {
            return new Array(8).fill(0.0);
        }
        
        const Gamma = this.christoffelSymbols(r, th);
        const dx = [kt, kr, kth, kph];
        const k = dx;
        const dk = new Array(4).fill(0.0);
        
        for (let mu = 0; mu < 4; mu++) {
            let s = 0.0;
            for (let alpha = 0; alpha < 4; alpha++) {
                for (let beta = 0; beta < 4; beta++) {
                    s += (Gamma[mu] && Gamma[mu][alpha] && Gamma[mu][alpha][beta] || 0.0) * k[alpha] * k[beta];
                }
            }
            dk[mu] = -s;
        }
        
        return [...dx, ...dk];
    }
    
    /**
     * RK45 integration step
     */
    rk45Step(f, lambda, y, h) {
        const c2 = 1/5;
        const c3 = 3/10;
        const c4 = 4/5;
        const c5 = 8/9;
        const c6 = 1.0;
        
        const k1 = f(lambda, y);
        const k2 = f(lambda + c2*h, this.addVectors(y, this.scaleVector(k1, h * 1/5)));
        const k3 = f(lambda + c3*h, this.addVectors(y, this.scaleVector(k1, h * 3/40), this.scaleVector(k2, h * 9/40)));
        const k4 = f(lambda + c4*h, this.addVectors(y, this.scaleVector(k1, h * 44/45), 
                   this.scaleVector(k2, h * -56/15), this.scaleVector(k3, h * 32/9)));
        const k5 = f(lambda + c5*h, this.addVectors(y, this.scaleVector(k1, h * 19372/6561),
                   this.scaleVector(k2, h * -25360/2187), this.scaleVector(k3, h * 64448/6561),
                   this.scaleVector(k4, h * -212/729)));
        const k6 = f(lambda + c6*h, this.addVectors(y, this.scaleVector(k1, h * 9017/3168),
                   this.scaleVector(k2, h * -355/33), this.scaleVector(k3, h * 46732/5247),
                   this.scaleVector(k4, h * 49/176), this.scaleVector(k5, h * -5103/18656)));
        
        // High-order estimate
        const B_high = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0];
        const y_high = this.addVectors(y, 
            this.scaleVector(k1, h * B_high[0]),
            this.scaleVector(k3, h * B_high[2]),
            this.scaleVector(k4, h * B_high[3]),
            this.scaleVector(k5, h * B_high[4]),
            this.scaleVector(k6, h * B_high[5]));
        
        const k7 = f(lambda + h, y_high);
        const y_high_final = this.addVectors(y_high, this.scaleVector(k7, h * B_high[6]));
        
        // Low-order estimate for error calculation
        const B_low = [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40];
        const y_low = this.addVectors(y, 
            this.scaleVector(k1, h * B_low[0]),
            this.scaleVector(k3, h * B_low[2]),
            this.scaleVector(k4, h * B_low[3]),
            this.scaleVector(k5, h * B_low[4]),
            this.scaleVector(k6, h * B_low[5]),
            this.scaleVector(k7, h * B_low[6]));
        
        const err_vec = this.subtractVectors(y_high_final, y_low);
        const err = this.vectorNorm(err_vec);
        
        return { y: y_high_final, err: err };
    }
    
    /**
     * Integrate geodesic using RK45
     */
    integrateRK45(f, y0, l0, l_max, h_init = 0.1, h_min = 1e-8, h_max = 0.1, 
                    rtol = 1e-10, atol = 1e-13) {
        let y = [...y0];
        let l = l0;
        let h = h_init;
        const traj = [y];
        let steps = 0;
        
        while (l < l_max && steps < 500000) {
            const result = this.rk45Step(f, l, y, h);
            const tol = atol + rtol * this.vectorNorm(y);
            const accept = (result.err <= tol) || (h <= h_min);
            
            if (accept) {
                y = result.y;
                l += h;
                traj.push([...y]);
                steps++;
                
                const r_current = y[1];
                const r_h = this.M + Math.sqrt(Math.max(0.0, this.M * this.M - this.a * this.a));
                if (r_current <= r_h * 1.01) break;
                if (r_current > 50 * 1.5) break;
            }
            
            if (result.err === 0.0) {
                h = Math.min(h_max, h * 5.0);
            } else {
                const fac = 0.9 * Math.pow(tol / result.err, 0.25);
                h = Math.max(h_min, Math.min(h_max, h * Math.max(0.2, Math.min(5.0, fac))));
            }
            
            if (l + h > l_max) {
                h = l_max - l;
                if (h <= 0) break;
            }
        }
        
        return traj;
    }
    
    /**
     * Helper functions for vector operations
     */
    addVectors(...vectors) {
        const result = new Array(vectors[0].length).fill(0);
        for (const v of vectors) {
            for (let i = 0; i < result.length; i++) {
                result[i] += v[i];
            }
        }
        return result;
    }
    
    subtractVectors(a, b) {
        const result = new Array(a.length);
        for (let i = 0; i < result.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }
    
    scaleVector(v, scalar) {
        const result = new Array(v.length);
        for (let i = 0; i < result.length; i++) {
            result[i] = v[i] * scalar;
        }
        return result;
    }
    
    vectorNorm(v) {
        let sum = 0;
        for (const x of v) {
            sum += x * x;
        }
        return Math.sqrt(sum);
    }
}

/**
 * Simplified interface for photon geodesic integration
 */
class PhotonGeodesicIntegrator {
    constructor(position, direction, mass, spin) {
        this.geodesic = new AdvancedKerrGeodesic(mass, spin);
        this.position = position;
        this.direction = direction;
        this.mass = mass;
        this.spin = spin;
    }
    
    /**
     * Integrate photon path
     */
    integrate(maxSteps = 1000, dt = 0.05) {
        // Convert Three.js vectors to geodesic coordinates
        const r = this.position.length();
        const theta = Math.acos(this.position.y / r);
        const phi = Math.atan2(this.position.z, this.position.x);
        
        // Initial state: [t, r, theta, phi, dt/dlambda, dr/dlambda, dtheta/dlambda, dphi/dlambda]
        const y0 = [0.0, r, theta, phi, 1.0, this.direction.x, this.direction.y, this.direction.z];
        
        // Integrate geodesic
        const traj = this.geodesic.integrateRK45(
            (l, y) => this.geodesic.geodesicEquations(l, y),
            y0,
            0.0,
            maxSteps * dt,
            dt,
            1e-8,
            dt
        );
        
        // Convert back to Three.js coordinates
        const lastState = traj[traj.length - 1];
        const newR = lastState[1];
        const newTheta = lastState[2];
        const newPhi = lastState[3];
        
        const newPosition = new THREE.Vector3(
            newR * Math.sin(newTheta) * Math.cos(newPhi),
            newR * Math.cos(newTheta),
            newR * Math.sin(newTheta) * Math.sin(newPhi)
        );
        
        return newPosition;
    }
}
