/**
 * Physics utilities for gravitational calculations
 */

class PhysicsUtils {
    /**
     * Calculate gravitational force using inverse square law
     * @param {THREE.Mesh} photon - The photon object
     * @param {Object} gravityParams - Gravity parameters (mass, gravitationalConstant)
     * @returns {THREE.Vector3} Gravitational force vector
     */
    static calculateGravitationalForce(photon, gravityParams) {
        const distance = photon.position.length();
        const direction = photon.position.clone().normalize();

        // Gravitational force magnitude (inverse square law)
        const forceMagnitude = (gravityParams.mass * gravityParams.gravitationalConstant) / (distance * distance);

        // Apply force in direction toward the gravitational object
        const force = direction.multiplyScalar(-forceMagnitude);

        return force;
    }

    /**
     * Calculate orbital force for circular motion
     * @param {THREE.Vector3} position - Current position
     * @param {number} orbitalSpeed - Orbital speed factor
     * @returns {THREE.Vector3} Orbital force vector
     */
    static calculateOrbitalForce(position, orbitalSpeed) {
        return new THREE.Vector3(
            -position.z * orbitalSpeed,
            0,
            position.x * orbitalSpeed
        );
    }

    /**
     * Limit velocity to prevent runaway acceleration
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} maxSpeed - Maximum allowed speed
     * @returns {THREE.Vector3} Limited velocity vector
     */
    static limitVelocity(velocity, maxSpeed = 0.4) {
        if (velocity.length() > maxSpeed) {
            return velocity.normalize().multiplyScalar(maxSpeed);
        }
        return velocity;
    }
    /**
     * Add vertical oscillation to position
     * @param {THREE.Vector3} position - Current position
     * @param {number} angle - Base angle for oscillation
     * @param {number} amplitude - Oscillation amplitude
     * @returns {number} New Y position with oscillation
     */
    static addVerticalOscillation(position, angle, amplitude = 0.001) {
        return position.y + Math.sin(Date.now() * 0.001 + angle) * amplitude;
    }
}
