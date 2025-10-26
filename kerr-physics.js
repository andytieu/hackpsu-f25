/**
 * Kerr Black Hole Physics
 * Implements Kerr metric for rotating black holes with frame dragging
 */

class KerrPhysics extends RelativisticPhysics {
    /**
     * Calculate Kerr metric components in Boyer-Lindquist coordinates
     * @param {number} r - Radial coordinate
     * @param {number} theta - Polar angle
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter (0 ≤ |a| ≤ M)
     * @returns {Object} Kerr metric components
     */
    static kerrMetric(r, theta, mass, spin) {
        const M = mass;
        const a = spin;
        const rho2 = r * r + a * a * Math.cos(theta) * Math.cos(theta);
        const delta = r * r - 2 * M * r + a * a;
        const sigma = (r * r + a * a) * (r * r + a * a) - delta * a * a * Math.sin(theta) * Math.sin(theta);
        
        return {
            g_tt: -(1 - 2 * M * r / rho2),
            g_tr: 0,
            g_tphi: -2 * M * a * r * Math.sin(theta) * Math.sin(theta) / rho2,
            g_rr: rho2 / delta,
            g_theta: rho2,
            g_phi: sigma * Math.sin(theta) * Math.sin(theta) / rho2,
            rho2: rho2,
            delta: delta,
            sigma: sigma
        };
    }
    
    /**
     * Calculate Kerr event horizon radius
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Event horizon radius
     */
    static kerrEventHorizon(mass, spin) {
        const M = mass;
        const a = spin;
        return M + Math.sqrt(M * M - a * a);
    }
    
    /**
     * Calculate Kerr ergosphere radius
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @param {number} theta - Polar angle
     * @returns {number} Ergosphere radius
     */
    static kerrErgosphere(mass, spin, theta) {
        const M = mass;
        const a = spin;
        return M + Math.sqrt(M * M - a * a * Math.sin(theta) * Math.sin(theta));
    }
    
    /**
     * Calculate Kerr ISCO (Innermost Stable Circular Orbit)
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @param {boolean} prograde - True for prograde orbits
     * @returns {number} ISCO radius
     */
    static kerrISCO(mass, spin, prograde = true) {
        const M = mass;
        const a = spin;
        const sign = prograde ? 1 : -1;
        
        const Z1 = 1 + Math.pow(1 - a * a / (M * M), 1/3) * 
                   (Math.pow(1 + a / M, 1/3) + Math.pow(1 - a / M, 1/3));
        const Z2 = Math.sqrt(3 * a * a / (M * M) + Z1 * Z1);
        
        const r_isco = M * (3 + Z2 - sign * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)));
        return r_isco;
    }
    
    /**
     * Calculate Kerr photon sphere radius
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @param {boolean} prograde - True for prograde orbits
     * @returns {number} Photon sphere radius
     */
    static kerrPhotonSphere(mass, spin, prograde = true) {
        const M = mass;
        const a = spin;
        const sign = prograde ? 1 : -1;
        
        const r_photon = 2 * M * (1 + Math.cos(2/3 * Math.acos(-sign * a / M)));
        return r_photon;
    }
    
    /**
     * Calculate frame dragging angular velocity
     * @param {number} r - Radial coordinate
     * @param {number} theta - Polar angle
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Frame dragging angular velocity
     */
    static frameDraggingAngularVelocity(r, theta, mass, spin) {
        const M = mass;
        const a = spin;
        const rho2 = r * r + a * a * Math.cos(theta) * Math.cos(theta);
        const sigma = (r * r + a * a) * (r * r + a * a) - (r * r - 2 * M * r + a * a) * a * a * Math.sin(theta) * Math.sin(theta);
        
        return 2 * M * a * r / sigma;
    }
    
    /**
     * Calculate Kerr gravitational force with frame dragging
     * @param {THREE.Mesh} photon - The photon object
     * @param {Object} params - Physics parameters including spin
     * @returns {THREE.Vector3} Kerr gravitational force
     */
    static calculateKerrForce(photon, params) {
        const position = photon.position;
        const distance = position.length();
        const mass = params.mass;
        const spin = params.spin || 0;
        
        // Convert Cartesian to spherical coordinates
        const r = distance;
        const theta = Math.acos(position.y / r);
        const phi = Math.atan2(position.z, position.x);
        
        // Check if inside event horizon
        const r_horizon = this.kerrEventHorizon(mass, spin);
        if (r <= r_horizon) {
            const direction = position.clone().normalize();
            return direction.multiplyScalar(-10.0); // Strong capture force
        }
        
        // Kerr metric components
        const metric = this.kerrMetric(r, theta, mass, spin);
        
        // Calculate relativistic force with frame dragging
        const direction = position.clone().normalize();
        const forceMagnitude = (mass * params.gravitationalConstant) / (r * r);
        
        // Frame dragging correction
        const frameDraggingFactor = 1 + Math.abs(spin) / mass;
        const relativisticFactor = 1 / Math.sqrt(-metric.g_tt);
        
        // Add frame dragging torque
        const frameDragForce = new THREE.Vector3();
        if (Math.abs(spin) > 0) {
            const omega = this.frameDraggingAngularVelocity(r, theta, mass, spin);
            const torque = new THREE.Vector3(0, omega, 0);
            frameDragForce.crossVectors(position, torque);
            frameDragForce.multiplyScalar(0.1); // Scale factor
        }
        
        const baseForce = direction.multiplyScalar(-forceMagnitude * relativisticFactor * frameDraggingFactor);
        return baseForce.add(frameDragForce);
    }
    
    /**
     * Calculate Kerr orbital velocity with frame dragging
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @param {boolean} prograde - True for prograde orbits
     * @returns {number} Orbital velocity
     */
    static kerrOrbitalVelocity(position, mass, spin, prograde = true) {
        const r = position.length();
        const theta = Math.acos(position.y / r);
        
        if (r <= this.kerrEventHorizon(mass, spin)) {
            return 0; // No stable orbits inside event horizon
        }
        
        // Kerr orbital velocity with frame dragging
        const v_newtonian = Math.sqrt(mass / r);
        const frameDraggingCorrection = 1 + Math.abs(spin) / mass;
        const relativisticFactor = Math.sqrt(1 - 2 * mass / r);
        
        return v_newtonian * relativisticFactor * frameDraggingCorrection;
    }
    
    /**
     * Check if position is inside Kerr event horizon
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {boolean} True if inside event horizon
     */
    static isInsideEventHorizon(position, mass, spin) {
        const distance = position.length();
        const r_horizon = this.kerrEventHorizon(mass, spin);
        return distance <= r_horizon;
    }
    
    /**
     * Check if position is inside Kerr ergosphere
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {boolean} True if inside ergosphere
     */
    static isInsideErgosphere(position, mass, spin) {
        const r = position.length();
        const theta = Math.acos(position.y / r);
        const r_ergosphere = this.kerrErgosphere(mass, spin, theta);
        return r <= r_ergosphere;
    }
    
    /**
     * Calculate Kerr redshift with frame dragging
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Redshift factor
     */
    static kerrRedshift(position, velocity, mass, spin) {
        const r = position.length();
        const theta = Math.acos(position.y / r);
        const metric = this.kerrMetric(r, theta, mass, spin);
        
        // Gravitational redshift
        const gravitationalRedshift = Math.sqrt(-metric.g_tt) - 1;
        
        // Doppler shift from frame dragging
        const omega = this.frameDraggingAngularVelocity(r, theta, mass, spin);
        const dopplerShift = omega * r * Math.sin(theta) / RelativisticPhysics.C;
        
        return gravitationalRedshift + dopplerShift;
    }
    
    /**
     * Calculate Carter constant for Kerr geodesics
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Carter constant
     */
    static carterConstant(position, velocity, mass, spin) {
        const r = position.length();
        const theta = Math.acos(position.y / r);
        const phi = Math.atan2(position.z, position.x);
        
        const metric = this.kerrMetric(r, theta, mass, spin);
        
        // Simplified Carter constant calculation
        const p_theta = r * velocity.y;
        const p_phi = r * Math.sin(theta) * velocity.z;
        
        return p_theta * p_theta + Math.sin(theta) * Math.sin(theta) * p_phi * p_phi;
    }
    
    /**
     * Calculate Kerr escape velocity
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Escape velocity
     */
    static kerrEscapeVelocity(position, mass, spin) {
        const r = position.length();
        const theta = Math.acos(position.y / r);
        
        if (r <= this.kerrEventHorizon(mass, spin)) {
            return Infinity; // Cannot escape from inside event horizon
        }
        
        // Kerr escape velocity with frame dragging correction
        const v_newtonian = Math.sqrt(2 * mass / r);
        const frameDraggingFactor = 1 + Math.abs(spin) / mass;
        const relativisticFactor = Math.sqrt(1 - 2 * mass / r);
        
        return v_newtonian * relativisticFactor * frameDraggingFactor;
    }
}
