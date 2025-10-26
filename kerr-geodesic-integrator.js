/**
 * Kerr Geodesic Integrator
 * Uses advanced geodesic solver for accurate photon paths
 */

class KerrGeodesicIntegrator {
    /**
     * Integrate Kerr geodesic using advanced geodesic solver
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} velocity - Current velocity
     * @param {number} mass - Mass of gravitational object
     * @param {number} spin - Spin parameter
     * @param {number} dt - Time step
     * @returns {Object} New position and velocity
     */
    static integrateKerrGeodesic(position, velocity, mass, spin, dt) {
        // Use advanced geodesic solver
        const geodesic = new AdvancedKerrGeodesic(mass, spin);
        
        // Convert Three.js vectors to geodesic coordinates
        const r = position.length();
        const theta = Math.acos(position.y / r);
        const phi = Math.atan2(position.z, position.x);
        
        // Initial state: [t, r, theta, phi, dt/dlambda, dr/dlambda, dtheta/dlambda, dphi/dlambda]
        const y0 = [0.0, r, theta, phi, 1.0, velocity.x, velocity.y, velocity.z];
        
        // Integrate geodesic using advanced solver
        const f = (lambda, y) => geodesic.geodesicEquations(lambda, y);
        const traj = geodesic.integrateRK45(f, y0, 0.0, dt, dt, 1e-8, dt);
        
        // Get final state
        const lastState = traj[traj.length - 1];
        const newR = lastState[1];
        const newTheta = lastState[2];
        const newPhi = lastState[3];
        
        // Convert back to Cartesian coordinates
        const newPosition = new THREE.Vector3(
            newR * Math.sin(newTheta) * Math.cos(newPhi),
            newR * Math.cos(newTheta),
            newR * Math.sin(newTheta) * Math.sin(newPhi)
        );
        
        // Calculate new velocity from geodesic
        const newVelocity = new THREE.Vector3(
            lastState[5],
            lastState[6],
            lastState[7]
        ).normalize().multiplyScalar(velocity.length());
        
        return {
            position: newPosition,
            velocity: newVelocity
        };
    }
}