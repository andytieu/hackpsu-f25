/**
 * Accretion Disk with Ray Tracing and Halos
 * Creates realistic accretion disk with gravitational effects
 */

class AccretionDiskRenderer {
    constructor(scene, blackHole, lightRaySystem) {
        this.scene = scene;
        this.blackHole = blackHole;
        this.lightRaySystem = null; // Not used anymore
        this.diskGeometry = null;
        this.diskMaterial = null;
        this.diskMesh = null;
        this.innermostRadius = 1.5;
        this.outermostRadius = 10;
        this.diskHeight = 0.5;
        this.segments = 64;
        
        // Halo parameters
        this.halo1 = null;
        this.halo2 = null;
        this.halo3 = null;
        
        this.createDisk();
        this.createHalos();
    }
    
    /**
     * Create accretion disk geometry
     */
    createDisk() {
        // Create disk as a torus geometry
        const torusGeometry = new THREE.TorusGeometry(
            (this.innermostRadius + this.outermostRadius) / 2,
            (this.outermostRadius - this.innermostRadius) / 2,
            this.segments,
            64
        );
        
        // Create material with temperature-based coloring
        const diskMaterial = new THREE.MeshLambertMaterial({
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8,
            emissive: 0x000000,
            emissiveIntensity: 0.5
        });
        
        this.diskMesh = new THREE.Mesh(torusGeometry, diskMaterial);
        this.diskMesh.rotation.x = Math.PI / 2;
        
        // Add vertex colors for temperature gradient
        const colors = [];
        const position = torusGeometry.attributes.position;
        
        for (let i = 0; i < position.count; i++) {
            const x = position.getX(i);
            const y = position.getY(i);
            const z = position.getZ(i);
            
            const distance = Math.sqrt(x * x + y * y + z * z);
            const temperature = this.calculateTemperature(distance);
            const color = this.temperatureToColor(temperature);
            
            colors.push(color.r, color.g, color.b);
        }
        
        torusGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        diskMaterial.vertexColors = true;
        
        this.scene.add(this.diskMesh);
    }
    
    /**
     * Create halo effects around the disk
     */
    createHalos() {
        // Inner halo (hot, blue-white)
        const halo1Geometry = new THREE.RingGeometry(
            this.innermostRadius,
            this.innermostRadius + 0.5,
            64
        );
        const halo1Material = new THREE.MeshBasicMaterial({
            color: 0x88ccff,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        this.halo1 = new THREE.Mesh(halo1Geometry, halo1Material);
        this.halo1.rotation.x = Math.PI / 2;
        this.scene.add(this.halo1);
        
        // Middle halo (warm, yellow-white)
        const halo2Geometry = new THREE.RingGeometry(
            this.innermostRadius + 2,
            this.innermostRadius + 3,
            64
        );
        const halo2Material = new THREE.MeshBasicMaterial({
            color: 0xffffaa,
            transparent: true,
            opacity: 0.4,
            side: THREE.DoubleSide
        });
        this.halo2 = new THREE.Mesh(halo2Geometry, halo2Material);
        this.halo2.rotation.x = Math.PI / 2;
        this.scene.add(this.halo2);
        
        // Outer halo (cool, red-orange)
        const halo3Geometry = new THREE.RingGeometry(
            this.innermostRadius + 4,
            this.innermostRadius + 5,
            64
        );
        const halo3Material = new THREE.MeshBasicMaterial({
            color: 0xff8844,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        this.halo3 = new THREE.Mesh(halo3Geometry, halo3Material);
        this.halo3.rotation.x = Math.PI / 2;
        this.scene.add(this.halo3);
    }
    
    /**
     * Calculate temperature at given distance
     */
    calculateTemperature(distance) {
        const innerTemp = 1.0;
        const outerTemp = 0.1;
        
        // Temperature decreases with radius (T ∝ r^(-3/4))
        const tempRatio = Math.pow(distance / this.innermostRadius, -0.75);
        return Math.max(outerTemp, Math.min(innerTemp, tempRatio * innerTemp));
    }
    
    /**
     * Convert temperature to color
     */
    temperatureToColor(temperature) {
        if (temperature > 0.8) {
            return new THREE.Color(0.8, 0.9, 1.0); // Blue-white
        } else if (temperature > 0.6) {
            return new THREE.Color(1.0, 1.0, 0.9); // White
        } else if (temperature > 0.4) {
            return new THREE.Color(1.0, 0.8, 0.4); // Yellow
        } else if (temperature > 0.2) {
            return new THREE.Color(1.0, 0.5, 0.2); // Orange
        } else {
            return new THREE.Color(0.9, 0.3, 0.2); // Red
        }
    }
    
    /**
     * Update disk rotation
     */
    updateRotation() {
        if (this.diskMesh) {
            this.diskMesh.rotation.z += 0.01;
        }
        if (this.halo1) {
            this.halo1.rotation.z += 0.015;
        }
        if (this.halo2) {
            this.halo2.rotation.z += 0.012;
        }
        if (this.halo3) {
            this.halo3.rotation.z += 0.008;
        }
    }
    
    /**
     * Check if ray intersects with disk
     */
    isRayInDisk(ray) {
        const distance = ray.position.length();
        const height = Math.abs(ray.position.y);
        
        return distance >= this.innermostRadius && 
               distance <= this.outermostRadius && 
               height <= this.diskHeight;
    }
    
    /**
     * Get disk properties at position
     */
    getDiskProperties(position) {
        if (!this.isRayInDisk({ position: position })) return null;
        
        const distance = position.length();
        const temperature = this.calculateTemperature(distance);
        const color = this.temperatureToColor(temperature);
        
        return {
            temperature: temperature,
            color: color,
            normal: new THREE.Vector3(0, 1, 0)
        };
    }
    
    /**
     * Toggle disk visibility
     */
    setVisible(visible) {
        if (this.diskMesh) this.diskMesh.visible = visible;
        if (this.halo1) this.halo1.visible = visible;
        if (this.halo2) this.halo2.visible = visible;
        if (this.halo3) this.halo3.visible = visible;
    }
    
    /**
     * Dispose resources
     */
    dispose() {
        if (this.diskMesh) {
            this.scene.remove(this.diskMesh);
            this.diskMesh.geometry.dispose();
            this.diskMesh.material.dispose();
        }
        if (this.halo1) {
            this.scene.remove(this.halo1);
            this.halo1.geometry.dispose();
            this.halo1.material.dispose();
        }
        if (this.halo2) {
            this.scene.remove(this.halo2);
            this.halo2.geometry.dispose();
            this.halo2.material.dispose();
        }
        if (this.halo3) {
            this.scene.remove(this.halo3);
            this.halo3.geometry.dispose();
            this.halo3.material.dispose();
        }
    }
}
