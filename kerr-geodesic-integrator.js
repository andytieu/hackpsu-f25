/**
 * Kerr Geodesic Integrator
 * Solves Kerr geodesic equations with frame dragging effects
 */

class KerrGeodesicIntegrator {
    /**
     * Fourth-order Runge-Kutta integration for Kerr geodesic equations
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @param {number} dt - Time step
     * @returns {Object} New position and velocity
     */
    static rk4KerrIntegration(position, velocity, mass, spin, dt) {
        // Calculate acceleration using Kerr force
        const acceleration = this.calculateKerrAcceleration(position, velocity, mass, spin);
        
        // RK4 coefficients
        const k1_pos = velocity.clone();
        const k1_vel = acceleration.clone();
        
        const k2_pos = velocity.clone().add(acceleration.clone().multiplyScalar(dt * 0.5));
        const k2_vel = this.calculateKerrAcceleration(
            position.clone().add(k1_pos.clone().multiplyScalar(dt * 0.5)),
            k2_pos,
            mass,
            spin
        );
        
        const k3_pos = velocity.clone().add(k2_vel.clone().multiplyScalar(dt * 0.5));
        const k3_vel = this.calculateKerrAcceleration(
            position.clone().add(k2_pos.clone().multiplyScalar(dt * 0.5)),
            k3_pos,
            mass,
            spin
        );
        
        const k4_pos = velocity.clone().add(k3_vel.clone().multiplyScalar(dt));
        const k4_vel = this.calculateKerrAcceleration(
            position.clone().add(k3_pos.clone().multiplyScalar(dt)),
            k4_pos,
            mass,
            spin
        );
        
        // Combine coefficients
        const newPosition = position.clone().add(
            k1_pos.add(k2_pos.multiplyScalar(2))
                  .add(k3_pos.multiplyScalar(2))
                  .add(k4_pos)
                  .multiplyScalar(dt / 6)
        );
        
        const newVelocity = velocity.clone().add(
            k1_vel.add(k2_vel.multiplyScalar(2))
                  .add(k3_vel.multiplyScalar(2))
                  .add(k4_vel)
                  .multiplyScalar(dt / 6)
        );
        
        return {
            position: newPosition,
            velocity: newVelocity
        };
    }
    
    /**
     * Calculate Kerr acceleration with frame dragging
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @returns {THREE.Vector3} Acceleration vector
     */
    static calculateKerrAcceleration(position, velocity, mass, spin) {
        const distance = position.length();
        const theta = Math.acos(position.y / distance);
        
        // Check if inside event horizon
        const r_horizon = KerrPhysics.kerrEventHorizon(mass, spin);
        if (distance <= r_horizon) {
            const direction = position.clone().normalize();
            return direction.multiplyScalar(-10.0);
        }
        
        // Kerr metric components
        const metric = KerrPhysics.kerrMetric(distance, theta, mass, spin);
        
        // Calculate relativistic acceleration
        const direction = position.clone().normalize();
        const accelerationMagnitude = mass / (distance * distance);
        
        // Frame dragging correction
        const frameDraggingFactor = 1 + Math.abs(spin) / mass;
        const relativisticFactor = 1 / Math.sqrt(-metric.g_tt);
        
        // Base gravitational acceleration
        const baseAcceleration = direction.multiplyScalar(-accelerationMagnitude * relativisticFactor * frameDraggingFactor);
        
        // Add frame dragging torque
        const frameDragAcceleration = new THREE.Vector3();
        if (Math.abs(spin) > 0) {
            const omega = KerrPhysics.frameDraggingAngularVelocity(distance, theta, mass, spin);
            
            // Coriolis-like force from frame dragging
            const coriolisForce = new THREE.Vector3();
            coriolisForce.crossVectors(velocity, new THREE.Vector3(0, omega, 0));
            coriolisForce.multiplyScalar(2); // Coriolis factor
            
            // Centrifugal-like force
            const centrifugalForce = new THREE.Vector3();
            centrifugalForce.crossVectors(
                new THREE.Vector3(0, omega, 0),
                new THREE.Vector3(0, omega, 0).cross(position)
            );
            
            frameDragAcceleration.add(coriolisForce).add(centrifugalForce);
            frameDragAcceleration.multiplyScalar(0.1); // Scale factor
        }
        
        return baseAcceleration.add(frameDragAcceleration);
    }
    
    /**
     * Adaptive step size control for Kerr spacetime
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @param {number} baseDt - Base time step
     * @returns {number} Adaptive time step
     */
    static adaptiveKerrStepSize(position, velocity, mass, spin, baseDt) {
        const distance = position.length();
        const r_horizon = KerrPhysics.kerrEventHorizon(mass, spin);
        const r_ergosphere = KerrPhysics.kerrErgosphere(mass, spin, Math.acos(position.y / distance));
        
        // Reduce step size near event horizon
        if (distance < 2 * r_horizon) {
            return baseDt * 0.05;
        }
        
        // Reduce step size near ergosphere
        if (distance < 2 * r_ergosphere) {
            return baseDt * 0.1;
        }
        
        // Reduce step size for high velocities
        const speed = velocity.length();
        if (speed > 0.8) { // Near speed of light
            return baseDt * 0.3;
        }
        
        // Reduce step size for high spin
        if (Math.abs(spin) > 0.8 * mass) {
            return baseDt * 0.5;
        }
        
        return baseDt;
    }
    
    /**
     * Check for numerical stability in Kerr spacetime
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @returns {boolean} True if stable
     */
    static isKerrStable(position, velocity, mass, spin) {
        const distance = position.length();
        const speed = velocity.length();
        
        // Check for NaN or infinite values
        if (!isFinite(distance) || !isFinite(speed)) {
            return false;
        }
        
        // Check for reasonable values
        if (distance < 0.01 || speed > 2.0) {
            return false;
        }
        
        // Check if inside event horizon (should be captured)
        const r_horizon = KerrPhysics.kerrEventHorizon(mass, spin);
        if (distance <= r_horizon) {
            return false; // Should be captured
        }
        
        // Check for extreme spin effects
        if (Math.abs(spin) > mass) {
            return false; // Invalid spin parameter
        }
        
        return true;
    }
    
    /**
     * Integrate Kerr geodesic with stability checks
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @param {number} dt - Time step
     * @returns {Object} New position and velocity, or null if unstable
     */
    static integrateKerrGeodesic(position, velocity, mass, spin, dt) {
        const adaptiveDt = this.adaptiveKerrStepSize(position, velocity, mass, spin, dt);
        
        const result = this.rk4KerrIntegration(position, velocity, mass, spin, adaptiveDt);
        
        if (!this.isKerrStable(result.position, result.velocity, mass, spin)) {
            return null; // Unstable integration
        }
        
        return result;
    }
    
    /**
     * Calculate Kerr-specific orbital precession
     * @param {THREE.Vector3} position - Position vector
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @returns {number} Precession rate
     */
    static kerrPrecessionRate(position, mass, spin) {
        const distance = position.length();
        const theta = Math.acos(position.y / distance);
        
        // Lense-Thirring precession
        const omega_lt = KerrPhysics.frameDraggingAngularVelocity(distance, theta, mass, spin);
        
        // Schwarzschild precession
        const omega_schwarzschild = 3 * mass / (distance * distance);
        
        return omega_lt + omega_schwarzschild;
    }
}
