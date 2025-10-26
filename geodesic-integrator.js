/**
 * Geodesic Integrator
 * Solves relativistic geodesic equations using Runge-Kutta methods
 */

class GeodesicIntegrator {
    /**
     * Fourth-order Runge-Kutta integration for geodesic equations
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} dt - Time step
     * @returns {Object} New position and velocity
     */
    static rk4Integration(position, velocity, mass, dt) {
        // Calculate acceleration using relativistic force
        const acceleration = this.calculateAcceleration(position, velocity, mass);
        
        // RK4 coefficients
        const k1_pos = velocity.clone();
        const k1_vel = acceleration.clone();
        
        const k2_pos = velocity.clone().add(acceleration.clone().multiplyScalar(dt * 0.5));
        const k2_vel = this.calculateAcceleration(
            position.clone().add(k1_pos.clone().multiplyScalar(dt * 0.5)),
            k2_pos,
            mass
        );
        
        const k3_pos = velocity.clone().add(k2_vel.clone().multiplyScalar(dt * 0.5));
        const k3_vel = this.calculateAcceleration(
            position.clone().add(k2_pos.clone().multiplyScalar(dt * 0.5)),
            k3_pos,
            mass
        );
        
        const k4_pos = velocity.clone().add(k3_vel.clone().multiplyScalar(dt));
        const k4_vel = this.calculateAcceleration(
            position.clone().add(k3_pos.clone().multiplyScalar(dt)),
            k4_pos,
            mass
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
     * Calculate acceleration from relativistic force
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @param {number} mass - Mass of gravitational object
     * @returns {THREE.Vector3} Acceleration vector
     */
    static calculateAcceleration(position, velocity, mass) {
        const distance = position.length();
        const rs = RelativisticPhysics.schwarzschildRadius(mass);
        
        if (distance <= rs) {
            // Inside event horizon - strong inward acceleration
            const direction = position.clone().normalize();
            return direction.multiplyScalar(-10.0);
        }
        
        // Relativistic gravitational acceleration
        const direction = position.clone().normalize();
        const f = 1 - rs / distance;
        const accelerationMagnitude = mass / (distance * distance);
        const relativisticFactor = 1 / Math.sqrt(f);
        
        return direction.multiplyScalar(-accelerationMagnitude * relativisticFactor);
    }
    
    /**
     * Adaptive step size control
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} baseDt - Base time step
     * @returns {number} Adaptive time step
     */
    static adaptiveStepSize(position, velocity, mass, baseDt) {
        const distance = position.length();
        const rs = RelativisticPhysics.schwarzschildRadius(mass);
        
        // Reduce step size near event horizon
        if (distance < 2 * rs) {
            return baseDt * 0.1;
        }
        
        // Reduce step size for high velocities
        const speed = velocity.length();
        if (speed > 0.8) { // Near speed of light
            return baseDt * 0.5;
        }
        
        return baseDt;
    }
    
    /**
     * Check for numerical stability
     * @param {THREE.Vector3} position - Position vector
     * @param {THREE.Vector3} velocity - Velocity vector
     * @returns {boolean} True if stable
     */
    static isStable(position, velocity) {
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
        
        return true;
    }
    
    /**
     * Integrate geodesic with stability checks
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} dt - Time step
     * @returns {Object} New position and velocity, or null if unstable
     */
    static integrateGeodesic(position, velocity, mass, dt) {
        const adaptiveDt = this.adaptiveStepSize(position, velocity, mass, dt);
        
        const result = this.rk4Integration(position, velocity, mass, adaptiveDt);
        
        if (!this.isStable(result.position, result.velocity)) {
            return null; // Unstable integration
        }
        
        return result;
    }
}
