/**
 * Relativistic Physics Utilities
 * Implements Schwarzschild metric and relativistic calculations
 */

class RelativisticPhysics {
    // Physical constants (in geometric units: c = G = 1)
    static C = 1.0; // Speed of light
    static G = 1.0; // Gravitational constant
    
    /**
     * Calculate Schwarzschild radius
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Schwarzschild radius
     */
    static schwarzschildRadius(mass) {
        return 2 * mass; // r_s = 2GM/c², with c = G = 1
    }
    
    /**
     * Calculate Schwarzschild metric components
     * @param {number} r - Radial coordinate
     * @param {number} mass - Mass of the gravitational object
     * @returns {Object} Metric components
     */
    static schwarzschildMetric(r, mass) {
        const rs = this.schwarzschildRadius(mass);
        const f = 1 - rs / r; // f(r) = 1 - 2M/r
        
        return {
            g_tt: -f,           // Time-time component
            g_rr: 1 / f,        // Radial-radial component
            g_theta: r * r,     // Theta-theta component
            g_phi: r * r        // Phi-phi component
        };
    }
    
    /**
     * Calculate time dilation factor
     * @param {number} r - Radial coordinate
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Time dilation factor dτ/dt
     */
    static timeDilation(r, mass) {
        const rs = this.schwarzschildRadius(mass);
        return Math.sqrt(1 - rs / r);
    }
    
    /**
     * Calculate gravitational redshift
     * @param {number} r - Radial coordinate
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Redshift factor
     */
    static gravitationalRedshift(r, mass) {
        const rs = this.schwarzschildRadius(mass);
        return Math.sqrt(1 - rs / r) - 1;
    }
    
    /**
     * Calculate ISCO (Innermost Stable Circular Orbit) radius
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} ISCO radius
     */
    static iscoRadius(mass) {
        return 6 * mass; // r_isco = 6GM/c²
    }
    
    /**
     * Calculate photon sphere radius
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Photon sphere radius
     */
    static photonSphereRadius(mass) {
        return 3 * mass; // r_photon = 3GM/c²
    }
    
    /**
     * Check if position is inside event horizon
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the gravitational object
     * @returns {boolean} True if inside event horizon
     */
    static isInsideEventHorizon(position, mass) {
        const distance = position.length();
        const rs = this.schwarzschildRadius(mass);
        return distance <= rs;
    }
    
    /**
     * Calculate relativistic gravitational force using Schwarzschild metric
     * @param {THREE.Mesh} photon - The photon object
     * @param {Object} params - Physics parameters
     * @returns {THREE.Vector3} Relativistic gravitational force
     */
    static calculateRelativisticForce(photon, params) {
        const distance = photon.position.length();
        const mass = params.mass;
        const rs = this.schwarzschildRadius(mass);
        
        // Check if inside event horizon
        if (distance <= rs) {
            // Inside event horizon - strong inward force
            const direction = photon.position.clone().normalize();
            return direction.multiplyScalar(-10.0); // Strong capture force
        }
        
        // Outside event horizon - relativistic correction
        const direction = photon.position.clone().normalize();
        const f = 1 - rs / distance;
        
        // Relativistic force magnitude
        const forceMagnitude = (mass * params.gravitationalConstant) / (distance * distance);
        const relativisticFactor = 1 / Math.sqrt(f); // Relativistic correction
        
        return direction.multiplyScalar(-forceMagnitude * relativisticFactor);
    }
    
    /**
     * Calculate relativistic orbital velocity
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Orbital velocity
     */
    static relativisticOrbitalVelocity(position, mass) {
        const distance = position.length();
        const rs = this.schwarzschildRadius(mass);
        
        if (distance <= rs) {
            return 0; // No stable orbits inside event horizon
        }
        
        // Relativistic orbital velocity
        const v_newtonian = Math.sqrt(mass / distance);
        const relativisticFactor = Math.sqrt(1 - rs / distance);
        
        return v_newtonian * relativisticFactor;
    }
    
    /**
     * Calculate gravitational lensing deflection angle
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Deflection angle in radians
     */
    static gravitationalLensingDeflection(position, mass) {
        const distance = position.length();
        const rs = this.schwarzschildRadius(mass);
        
        // Einstein deflection angle: α = 4GM/rc²
        return 4 * mass / distance;
    }
    
    /**
     * Apply relativistic velocity limit
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} maxSpeed - Maximum speed (default: speed of light)
     * @returns {THREE.Vector3} Limited velocity vector
     */
    static limitRelativisticVelocity(velocity, maxSpeed = this.C) {
        const speed = velocity.length();
        if (speed > maxSpeed) {
            return velocity.normalize().multiplyScalar(maxSpeed);
        }
        return velocity;
    }
    
    /**
     * Calculate relativistic momentum
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Rest mass
     * @returns {THREE.Vector3} Relativistic momentum
     */
    static relativisticMomentum(velocity, mass = 0) {
        const speed = velocity.length();
        const gamma = 1 / Math.sqrt(1 - speed * speed / (this.C * this.C));
        return velocity.multiplyScalar(gamma * mass);
    }
    
    /**
     * Calculate proper time step
     * @param {number} coordinateTimeStep - Coordinate time step
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Proper time step
     */
    static properTimeStep(coordinateTimeStep, position, mass) {
        const timeDilationFactor = this.timeDilation(position.length(), mass);
        return coordinateTimeStep * timeDilationFactor;
    }
    
    /**
     * Calculate escape velocity
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of the gravitational object
     * @returns {number} Escape velocity
     */
    static escapeVelocity(position, mass) {
        const distance = position.length();
        const rs = this.schwarzschildRadius(mass);
        
        if (distance <= rs) {
            return Infinity; // Cannot escape from inside event horizon
        }
        
        // Relativistic escape velocity
        const v_newtonian = Math.sqrt(2 * mass / distance);
        const relativisticFactor = Math.sqrt(1 - rs / distance);
        
        return v_newtonian * relativisticFactor;
    }
}
