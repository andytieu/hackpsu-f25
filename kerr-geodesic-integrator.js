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
        
        // Convert Three.js vectors to Boyer-Lindquist coordinates (spherical)
        const r = position.length();
        
        // Performance optimization: skip geodesic integration if too far from black hole
        if (r > 20 * mass) {
            // Use simple linear update for far-away photons
            const newPosition = position.clone().add(velocity.clone().multiplyScalar(dt));
            return {
                position: newPosition,
                velocity: velocity.clone()
            };
        }
        
        const theta = Math.acos(position.y / r);  // Colatitude
        const phi = Math.atan2(position.z, position.x);  // Azimuth
        
        // Convert Cartesian velocity to Boyer-Lindquist velocity
        // For photon geodesics, we need proper 4-velocity components
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        
        // Convert velocity from Cartesian (x,y,z) to (r, theta, phi) components
        const vr = (velocity.x * sinTheta * cosPhi + 
                   velocity.y * cosTheta + 
                   velocity.z * sinTheta * sinPhi);
        
        const vtheta = (velocity.x * cosTheta * cosPhi - 
                       velocity.y * sinTheta + 
                       velocity.z * cosTheta * sinPhi) / r;
        
        const vphi = (-velocity.x * sinPhi + velocity.z * cosPhi) / (r * sinTheta);
        
        // Initial state: [t, r, theta, phi, dt/dlambda, dr/dlambda, dtheta/dlambda, dphi/dlambda]
        const y0 = [0.0, r, theta, phi, 1.0, vr, vtheta, vphi];
        
        // Integrate geodesic using advanced solver
        // Use a balanced lambda_max for performance and visual clarity
        const f = (lambda, y) => geodesic.geodesicEquations(lambda, y);
        const l_max = Math.max(0.15, dt * 20); // Optimized for performance
        const traj = geodesic.integrateRK45(f, y0, 0.0, l_max, dt * 0.5, 1e-6, 0.1);
        
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
        
        // Get velocity in Boyer-Lindquist coordinates
        const newVr = lastState[5];      // dr/dlambda
        const newVtheta = lastState[6];   // dtheta/dlambda
        const newVphi = lastState[7];    // dphi/dlambda
        
        // Convert Boyer-Lindquist velocity to Cartesian velocity
        const sinThetaNew = Math.sin(newTheta);
        const cosThetaNew = Math.cos(newTheta);
        const sinPhiNew = Math.sin(newPhi);
        const cosPhiNew = Math.cos(newPhi);
        
        // Transform from spherical to Cartesian velocity
        const vx = newVr * sinThetaNew * cosPhiNew + 
                  newVtheta * newR * cosThetaNew * cosPhiNew - 
                  newVphi * newR * sinThetaNew * sinPhiNew;
        
        const vy = newVr * cosThetaNew - 
                  newVtheta * newR * sinThetaNew;
        
        const vz = newVr * sinThetaNew * sinPhiNew + 
                  newVtheta * newR * cosThetaNew * sinPhiNew + 
                  newVphi * newR * sinThetaNew * cosPhiNew;
        
        const newVelocity = new THREE.Vector3(vx, vy, vz);
        
        return {
            position: newPosition,
            velocity: newVelocity
        };
    }
}