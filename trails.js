/**
 * Trail management utilities for photon trails
 */

class TrailManager {
    /**
     * Create a trail for a photon
     * @param {THREE.Mesh} photon - The photon object
     * @param {THREE.Scene} scene - Three.js scene
     * @param {Array} trailArray - Array to store trail references
     * @param {Object} options - Trail options (color, opacity, maxPoints)
     */
    static createTrail(photon, scene, trailArray, options = {}) {
        const {
            color = 0xffff00,
            opacity = 0.3,
            maxPoints = 50
        } = options;

        const trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity
        });
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        trail.userData = { photon: photon, maxPoints: maxPoints };
        trailArray.push(trail);
        scene.add(trail);
    }

    /**
     * Update trail geometry with new position
     * @param {THREE.Mesh} photon - The photon object
     * @param {Array} trailArray - Array containing trail references
     */
    static updateTrail(photon, trailArray) {
        const trail = trailArray.find(t => t.userData.photon === photon);
        if (!trail) return;

        const trailData = trail.userData;
        trailData.photon.userData.trail.push(photon.position.clone());

        // Limit trail length
        if (trailData.photon.userData.trail.length > trailData.maxPoints) {
            trailData.photon.userData.trail.shift();
        }

        // Update trail geometry
        if (trailData.photon.userData.trail.length > 1) {
            const positions = new Float32Array(trailData.photon.userData.trail.length * 3);
            trailData.photon.userData.trail.forEach((point, index) => {
                positions[index * 3] = point.x;
                positions[index * 3 + 1] = point.y;
                positions[index * 3 + 2] = point.z;
            });

            trail.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            trail.geometry.attributes.position.needsUpdate = true;
        }
    }

    /**
     * Clear trail data for a photon
     * @param {THREE.Mesh} photon - The photon object
     */
    static clearTrail(photon) {
        photon.userData.trail = [];
    }
}
