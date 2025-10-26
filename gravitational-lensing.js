/**
 * Gravitational Lensing Physics
 * Implements light bending effects around black holes
 */

class GravitationalLensing {
    /**
     * Calculate gravitational lensing deflection angle
     * @param {number} impactParameter - Distance of closest approach
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {number} Deflection angle in radians
     */
    static deflectionAngle(impactParameter, mass, spin) {
        const rs = KerrPhysics.kerrEventHorizon(mass, spin);
        
        // Schwarzschild deflection angle (simplified)
        // For small angles: α ≈ 4GM/(c²b) where b is impact parameter
        const deflection = (4 * mass) / impactParameter;
        
        // Add Kerr correction for frame dragging
        const kerrCorrection = 1 + Math.abs(spin) / mass;
        
        return deflection * kerrCorrection;
    }
    
    /**
     * Calculate Einstein radius for gravitational lensing
     * @param {number} mass - Mass of the black hole
     * @param {number} sourceDistance - Distance to source
     * @param {number} lensDistance - Distance to lens (black hole)
     * @returns {number} Einstein radius
     */
    static einsteinRadius(mass, sourceDistance, lensDistance) {
        const rs = KerrPhysics.kerrEventHorizon(mass, 0); // Use Schwarzschild for simplicity
        const dls = sourceDistance - lensDistance;
        const ds = sourceDistance;
        const dl = lensDistance;
        
        return Math.sqrt((4 * mass * dls * dl) / ds);
    }
    
    /**
     * Check if a light ray will be captured by the black hole
     * @param {THREE.Vector3} position - Current position
     * @param {THREE.Vector3} direction - Light ray direction
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {boolean} True if captured
     */
    static isRayCaptured(position, direction, mass, spin) {
        const distance = position.length();
        const r_horizon = KerrPhysics.kerrEventHorizon(mass, spin);
        const r_photon = KerrPhysics.kerrPhotonSphere(mass, spin, true);
        
        // If inside photon sphere, likely to be captured
        if (distance <= r_photon) {
            return true;
        }
        
        // Check if trajectory leads to event horizon
        const velocity = direction.clone().normalize();
        const radialVelocity = position.clone().normalize().dot(velocity);
        
        // If moving toward black hole with high velocity, likely captured
        if (radialVelocity < -0.5 && distance < r_photon * 2) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Calculate lensed position of a background star
     * @param {THREE.Vector3} starPosition - Original star position
     * @param {THREE.Vector3} blackHolePosition - Black hole position
     * @param {number} mass - Mass of the black hole
     * @param {number} spin - Spin parameter
     * @returns {THREE.Vector3} Lensed star position
     */
    static lensStarPosition(starPosition, blackHolePosition, mass, spin) {
        const relativePosition = starPosition.clone().sub(blackHolePosition);
        const distance = relativePosition.length();
        
        if (distance < 0.1) return starPosition; // Avoid division by zero
        
        // Calculate impact parameter (perpendicular distance)
        const impactParameter = distance;
        const deflectionAngle = this.deflectionAngle(impactParameter, mass, spin);
        
        // Apply deflection perpendicular to the line of sight
        const deflectionDirection = new THREE.Vector3();
        deflectionDirection.crossVectors(relativePosition, new THREE.Vector3(0, 1, 0));
        deflectionDirection.normalize();
        
        // Scale deflection by distance
        const deflectionMagnitude = deflectionAngle * distance * 0.1; // Scale factor
        
        const lensedPosition = starPosition.clone().add(
            deflectionDirection.multiplyScalar(deflectionMagnitude)
        );
        
        return lensedPosition;
    }
    
    /**
     * Calculate magnification factor for lensed light
     * @param {THREE.Vector3} starPosition - Original star position
     * @param {THREE.Vector3} blackHolePosition - Black hole position
     * @param {number} mass - Mass of the black hole
     * @returns {number} Magnification factor
     */
    static calculateMagnification(starPosition, blackHolePosition, mass) {
        const distance = starPosition.distanceTo(blackHolePosition);
        const einsteinRadius = this.einsteinRadius(mass, 100, 50); // Example distances
        
        // Magnification increases as star approaches Einstein radius
        const magnification = 1 + (einsteinRadius / distance) * 2;
        
        return Math.min(magnification, 10); // Cap magnification
    }
    
    /**
     * Generate Einstein ring positions
     * @param {THREE.Vector3} blackHolePosition - Black hole position
     * @param {number} mass - Mass of the black hole
     * @param {number} numRings - Number of ring points
     * @returns {Array} Array of ring positions
     */
    static generateEinsteinRing(blackHolePosition, mass, numRings = 32) {
        const einsteinRadius = this.einsteinRadius(mass, 100, 50);
        const ringPositions = [];
        
        for (let i = 0; i < numRings; i++) {
            const angle = (i / numRings) * Math.PI * 2;
            const x = blackHolePosition.x + Math.cos(angle) * einsteinRadius;
            const z = blackHolePosition.z + Math.sin(angle) * einsteinRadius;
            const y = blackHolePosition.y;
            
            ringPositions.push(new THREE.Vector3(x, y, z));
        }
        
        return ringPositions;
    }
    
    /**
     * Check if a star is in perfect alignment for Einstein ring
     * @param {THREE.Vector3} starPosition - Star position
     * @param {THREE.Vector3} blackHolePosition - Black hole position
     * @param {THREE.Vector3} observerPosition - Observer position
     * @param {number} mass - Mass of the black hole
     * @returns {boolean} True if perfect alignment
     */
    static isPerfectAlignment(starPosition, blackHolePosition, observerPosition, mass) {
        const starToBH = starPosition.clone().sub(blackHolePosition);
        const observerToBH = observerPosition.clone().sub(blackHolePosition);
        
        // Check if star, black hole, and observer are collinear
        const crossProduct = starToBH.clone().cross(observerToBH);
        const alignmentTolerance = 0.1;
        
        return crossProduct.length() < alignmentTolerance;
    }
    
    /**
     * Calculate multiple image positions for lensed star
     * @param {THREE.Vector3} starPosition - Original star position
     * @param {THREE.Vector3} blackHolePosition - Black hole position
     * @param {number} mass - Mass of the black hole
     * @returns {Array} Array of image positions
     */
    static calculateMultipleImages(starPosition, blackHolePosition, mass) {
        const images = [];
        const distance = starPosition.distanceTo(blackHolePosition);
        const einsteinRadius = this.einsteinRadius(mass, 100, 50);
        
        if (distance < einsteinRadius * 2) {
            // Primary image (slightly displaced)
            const primaryDisplacement = einsteinRadius * einsteinRadius / distance;
            const primaryDirection = starPosition.clone().sub(blackHolePosition).normalize();
            const primaryImage = starPosition.clone().add(
                primaryDirection.multiplyScalar(primaryDisplacement)
            );
            images.push(primaryImage);
            
            // Secondary image (on opposite side)
            const secondaryImage = starPosition.clone().add(
                primaryDirection.multiplyScalar(-primaryDisplacement)
            );
            images.push(secondaryImage);
        }
        
        return images;
    }
}
