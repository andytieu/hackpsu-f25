/**
 * Background Starfield Manager
 * Creates and manages lensed background stars
 */

class BackgroundStarfield {
    constructor(scene, blackHolePosition, mass, spin) {
        this.scene = scene;
        this.blackHolePosition = blackHolePosition;
        this.mass = mass;
        this.spin = spin;
        this.stars = [];
        this.lensedStars = [];
        this.einsteinRings = [];
        
        this.createStarfield();
    }
    
    /**
     * Create a procedural starfield
     */
    createStarfield() {
        const numStars = 2000;
        const starfieldRadius = 20;
        
        for (let i = 0; i < numStars; i++) {
            // Generate random star positions on a sphere
            const phi = Math.random() * Math.PI * 2;
            const theta = Math.acos(2 * Math.random() - 1);
            const radius = starfieldRadius + Math.random() * 20;
            
            const x = radius * Math.sin(theta) * Math.cos(phi);
            const y = radius * Math.sin(theta) * Math.sin(phi);
            const z = radius * Math.cos(theta);
            
            const starPosition = new THREE.Vector3(x, y, z);
            
            // Create star geometry
            const starGeometry = new THREE.SphereGeometry(0.02, 4, 4);
            const starMaterial = new THREE.MeshBasicMaterial({
                color: new THREE.Color().setHSL(Math.random() * 0.1 + 0.5, 0.8, 0.8),
                emissive: new THREE.Color().setHSL(Math.random() * 0.1 + 0.5, 0.8, 0.3)
            });
            
            const star = new THREE.Mesh(starGeometry, starMaterial);
            star.position.copy(starPosition);
            star.userData = {
                originalPosition: starPosition.clone(),
                brightness: Math.random() * 0.5 + 0.5
            };
            
            this.stars.push(star);
            this.scene.add(star);
        }
        
        this.updateLensing();
    }
    
    /**
     * Update gravitational lensing effects
     */
    updateLensing() {
        this.stars.forEach(star => {
            const originalPos = star.userData.originalPosition;
            
            // Calculate lensed position
            const lensedPosition = GravitationalLensing.lensStarPosition(
                originalPos,
                this.blackHolePosition,
                this.mass,
                this.spin
            );
            
            // Update star position
            star.position.copy(lensedPosition);
            
            // Calculate magnification
            const magnification = GravitationalLensing.calculateMagnification(
                originalPos,
                this.blackHolePosition,
                this.mass
            );
            
            // Update brightness based on magnification
            const baseBrightness = star.userData.brightness;
            const magnifiedBrightness = baseBrightness * magnification;
            
            // Update star material
            star.material.emissiveIntensity = magnifiedBrightness;
            star.material.opacity = Math.min(magnifiedBrightness, 1.0);
            star.material.transparent = true;
            
            // Scale star size based on magnification
            const scale = Math.min(1 + magnification * 0.5, 3);
            star.scale.setScalar(scale);
        });
        
        this.updateEinsteinRings();
    }
    
    /**
     * Update Einstein ring visualization
     */
    updateEinsteinRings() {
        // Remove existing rings
        this.einsteinRings.forEach(ring => {
            this.scene.remove(ring);
        });
        this.einsteinRings = [];
        
        // Generate new Einstein ring
        const ringPositions = GravitationalLensing.generateEinsteinRing(
            this.blackHolePosition,
            this.mass,
            64
        );
        
        // Create ring geometry
        const ringGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(ringPositions.length * 3);
        
        ringPositions.forEach((pos, index) => {
            positions[index * 3] = pos.x;
            positions[index * 3 + 1] = pos.y;
            positions[index * 3 + 2] = pos.z;
        });
        
        ringGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const ringMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.3
        });
        
        const einsteinRing = new THREE.LineLoop(ringGeometry, ringMaterial);
        this.einsteinRings.push(einsteinRing);
        this.scene.add(einsteinRing);
    }
    
    /**
     * Update black hole parameters
     */
    updateBlackHole(mass, spin) {
        this.mass = mass;
        this.spin = spin;
        this.updateLensing();
    }
    
    /**
     * Update black hole position
     */
    updateBlackHolePosition(position) {
        this.blackHolePosition.copy(position);
        this.updateLensing();
    }
    
    /**
     * Toggle starfield visibility
     */
    setVisible(visible) {
        this.stars.forEach(star => {
            star.visible = visible;
        });
        this.einsteinRings.forEach(ring => {
            ring.visible = visible;
        });
    }
    
    /**
     * Add multiple image visualization for a specific star
     */
    addMultipleImages(starIndex) {
        if (starIndex >= this.stars.length) return;
        
        const star = this.stars[starIndex];
        const originalPos = star.userData.originalPosition;
        
        const images = GravitationalLensing.calculateMultipleImages(
            originalPos,
            this.blackHolePosition,
            this.mass
        );
        
        images.forEach((imagePos, index) => {
            const imageGeometry = new THREE.SphereGeometry(0.03, 6, 6);
            const imageMaterial = new THREE.MeshBasicMaterial({
                color: 0xffff00,
                emissive: 0x222200,
                transparent: true,
                opacity: 0.7
            });
            
            const image = new THREE.Mesh(imageGeometry, imageMaterial);
            image.position.copy(imagePos);
            image.userData = { isMultipleImage: true, parentStar: star };
            
            this.scene.add(image);
            this.lensedStars.push(image);
        });
    }
    
    /**
     * Remove multiple image visualizations
     */
    removeMultipleImages() {
        this.lensedStars.forEach(star => {
            this.scene.remove(star);
        });
        this.lensedStars = [];
    }
    
    /**
     * Clean up resources
     */
    dispose() {
        this.stars.forEach(star => {
            star.geometry.dispose();
            star.material.dispose();
            this.scene.remove(star);
        });
        
        this.einsteinRings.forEach(ring => {
            ring.geometry.dispose();
            ring.material.dispose();
            this.scene.remove(ring);
        });
        
        this.lensedStars.forEach(star => {
            star.geometry.dispose();
            star.material.dispose();
            this.scene.remove(star);
        });
    }
}
