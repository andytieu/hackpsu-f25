/**
 * Light Ray System for Interstellar-Style Gravitational Lensing
 * Creates clustered light rays that bend around black holes
 */

class LightRaySystem {
    constructor(scene, camera, blackHole) {
        this.scene = scene;
        this.camera = camera;
        this.blackHole = blackHole;
        this.rays = [];
        this.rayTrails = [];
        this.maxRays = 200;
        this.rayLength = 100;
        this.stepSize = 0.05;
        this.trailLength = 50;
        this.rayThickness = 0.02;
        
        // Clustering parameters
        this.clusters = {
            primary: { count: 80, target: 'center', spread: 0.1, color: 0x88ccff },
            secondary: { count: 60, target: 'photon-sphere', spread: 0.2, color: 0xffffff },
            tertiary: { count: 40, target: 'disk-edge', spread: 0.3, color: 0xffff88 },
            escape: { count: 20, target: 'escape', spread: 0.4, color: 0xff8888 }
        };
        
        // Performance settings
        this.updateFrequency = 1; // Update every frame
        this.regenerationRate = 0.02; // 2% chance per frame
    }
    
    /**
     * Generate all ray clusters
     */
    generateRayClusters() {
        this.clearRays();
        
        Object.keys(this.clusters).forEach(clusterName => {
            const cluster = this.clusters[clusterName];
            this.generateCluster(clusterName, cluster);
        });
        
        console.log(`Generated ${this.rays.length} light rays`);
    }
    
    /**
     * Generate rays for a specific cluster
     */
    generateCluster(clusterName, cluster) {
        const cameraPosition = this.camera.position.clone();
        const cameraDirection = new THREE.Vector3();
        this.camera.getWorldDirection(cameraDirection);
        
        for (let i = 0; i < cluster.count; i++) {
            const ray = this.createRay(clusterName, cluster, cameraPosition, cameraDirection);
            this.rays.push(ray);
            this.createRayTrail(ray);
        }
    }
    
    /**
     * Create a single ray
     */
    createRay(clusterName, cluster, cameraPosition, cameraDirection) {
        // Calculate ray direction based on cluster target
        let rayDirection = this.calculateRayDirection(clusterName, cluster, cameraDirection);
        
        // Add some randomness for natural clustering
        const randomFactor = cluster.spread;
        rayDirection.x += (Math.random() - 0.5) * randomFactor;
        rayDirection.y += (Math.random() - 0.5) * randomFactor;
        rayDirection.z += (Math.random() - 0.5) * randomFactor;
        rayDirection.normalize();
        
        const ray = {
            position: cameraPosition.clone(),
            direction: rayDirection,
            energy: 1.0,
            color: new THREE.Color(cluster.color),
            trail: [],
            active: true,
            bounces: 0,
            maxBounces: 2,
            cluster: clusterName,
            age: 0
        };
        
        return ray;
    }
    
    /**
     * Calculate ray direction based on cluster target
     */
    calculateRayDirection(clusterName, cluster, cameraDirection) {
        const direction = new THREE.Vector3();
        
        switch (clusterName) {
            case 'primary':
                // Rays aimed at black hole center
                direction.set(0, 0, 0).sub(this.camera.position).normalize();
                break;
                
            case 'secondary':
                // Rays aimed at photon sphere
                const photonSphereRadius = 3.0; // Approximate photon sphere
                const angle = Math.random() * Math.PI * 2;
                direction.set(
                    Math.cos(angle) * photonSphereRadius,
                    0,
                    Math.sin(angle) * photonSphereRadius
                ).sub(this.camera.position).normalize();
                break;
                
            case 'tertiary':
                // Rays aimed at accretion disk edges
                const diskRadius = 4.0;
                const diskAngle = Math.random() * Math.PI * 2;
                const diskHeight = (Math.random() - 0.5) * 0.5;
                direction.set(
                    Math.cos(diskAngle) * diskRadius,
                    diskHeight,
                    Math.sin(diskAngle) * diskRadius
                ).sub(this.camera.position).normalize();
                break;
                
            case 'escape':
                // Rays aimed away from black hole
                direction.copy(cameraDirection);
                break;
                
            default:
                direction.copy(cameraDirection);
        }
        
        return direction;
    }
    
    /**
     * Create visual trail for a ray
     */
    createRayTrail(ray) {
        // Create a thick ray using tube geometry
        const trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.MeshBasicMaterial({
            color: ray.color,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        const trail = new THREE.Mesh(trailGeometry, trailMaterial);
        trail.userData = { ray: ray, isThickRay: true };
        this.rayTrails.push(trail);
        this.scene.add(trail);
    }
    
    /**
     * Update all rays
     */
    updateRays() {
        this.rays.forEach(ray => {
            if (ray.active) {
                this.traceRay(ray);
                this.updateRayTrail(ray);
                ray.age++;
            }
        });
        
        // Regenerate rays periodically
        if (Math.random() < this.regenerationRate) {
            this.regenerateRays();
        }
    }
    
    /**
     * Trace a single ray through spacetime
     */
    traceRay(ray) {
        // Store current position in trail
        ray.trail.push(ray.position.clone());
        
        // Limit trail length
        if (ray.trail.length > this.trailLength) {
            ray.trail.shift();
        }
        
        // Calculate next position using geodesic integration
        const nextPosition = this.integrateRayPath(ray);
        
        if (nextPosition) {
            ray.position.copy(nextPosition);
            
            // Check for interactions
            this.checkRayInteractions(ray);
            
            // Check if ray escapes or gets captured
            if (this.isRayCaptured(ray) || this.isRayEscaped(ray)) {
                ray.active = false;
            }
        } else {
            ray.active = false;
        }
    }
    
    /**
     * Integrate ray path using geodesic equations
     */
    integrateRayPath(ray) {
        const position = ray.position.clone();
        const direction = ray.direction.clone();
        const mass = this.blackHole.userData.mass || 1.0;
        const spin = this.blackHole.userData.spin || 0.5;
        
        // Use Kerr geodesic integration
        const result = KerrGeodesicIntegrator.integrateKerrGeodesic(
            position,
            direction,
            mass,
            spin,
            this.stepSize
        );
        
        if (result) {
            // Update ray direction
            ray.direction.copy(result.velocity.normalize());
            
            // Apply relativistic effects
            this.applyRelativisticEffects(ray, position, result.position);
            
            return result.position;
        }
        
        return null;
    }
    
    /**
     * Apply relativistic effects to ray
     */
    applyRelativisticEffects(ray, oldPosition, newPosition) {
        const distance = newPosition.length();
        const mass = this.blackHole.userData.mass || 1.0;
        
        // Gravitational redshift
        const redshift = 1 / Math.sqrt(1 - 2 * mass / distance);
        ray.energy *= redshift;
        
        // Update ray color based on energy
        this.updateRayColor(ray);
        
        // Limit energy to prevent extreme values
        ray.energy = Math.min(ray.energy, 5.0);
    }
    
    /**
     * Update ray color based on energy
     */
    updateRayColor(ray) {
        const energy = Math.min(ray.energy, 3.0);
        
        if (energy > 2.0) {
            ray.color.setHSL(0.6, 1.0, 0.8); // Blue-white
        } else if (energy > 1.5) {
            ray.color.setHSL(0.1, 0.8, 0.9); // White
        } else if (energy > 1.0) {
            ray.color.setHSL(0.15, 1.0, 0.7); // Yellow
        } else {
            ray.color.setHSL(0.0, 1.0, 0.5); // Red
        }
    }
    
    /**
     * Check for ray interactions
     */
    checkRayInteractions(ray) {
        // Check for accretion disk interaction
        const diskIntersection = this.checkDiskIntersection(ray);
        if (diskIntersection) {
            this.handleDiskIntersection(ray, diskIntersection);
        }
        
        // Check for photon sphere interaction
        if (this.isNearPhotonSphere(ray)) {
            this.handlePhotonSphereInteraction(ray);
        }
    }
    
    /**
     * Check if ray intersects with accretion disk
     */
    checkDiskIntersection(ray) {
        const diskRadius = 3.0;
        const diskHeight = 0.3;
        
        const distance = ray.position.length();
        const height = Math.abs(ray.position.y);
        
        if (distance < diskRadius && height < diskHeight) {
            return {
                position: ray.position.clone(),
                normal: new THREE.Vector3(0, Math.sign(ray.position.y), 0),
                temperature: this.calculateDiskTemperature(distance)
            };
        }
        
        return null;
    }
    
    /**
     * Handle ray intersection with accretion disk
     */
    handleDiskIntersection(ray, intersection) {
        // Calculate scattering
        const scatterDirection = this.calculateScattering(ray, intersection);
        ray.direction.copy(scatterDirection);
        
        // Update ray energy based on disk temperature
        const temperature = intersection.temperature;
        ray.energy *= (1 + temperature * 0.2);
        
        // Update ray color based on temperature
        ray.color.setHSL(0.1, 1.0, temperature);
        
        ray.bounces++;
        if (ray.bounces >= ray.maxBounces) {
            ray.active = false;
        }
    }
    
    /**
     * Check if ray is near photon sphere
     */
    isNearPhotonSphere(ray) {
        const distance = ray.position.length();
        const photonSphereRadius = 3.0;
        return Math.abs(distance - photonSphereRadius) < 0.5;
    }
    
    /**
     * Handle photon sphere interaction
     */
    handlePhotonSphereInteraction(ray) {
        // Rays near photon sphere get deflected
        const deflection = 0.1;
        ray.direction.x += (Math.random() - 0.5) * deflection;
        ray.direction.y += (Math.random() - 0.5) * deflection;
        ray.direction.z += (Math.random() - 0.5) * deflection;
        ray.direction.normalize();
        
        // Increase energy due to gravitational focusing
        ray.energy *= 1.1;
    }
    
    /**
     * Calculate scattering direction
     */
    calculateScattering(ray, intersection) {
        const normal = intersection.normal;
        const incident = ray.direction.clone();
        
        // Simple reflection with some randomness
        const reflected = incident.clone().reflect(normal);
        
        // Add randomness for realistic scattering
        const randomFactor = 0.2;
        reflected.x += (Math.random() - 0.5) * randomFactor;
        reflected.y += (Math.random() - 0.5) * randomFactor;
        reflected.z += (Math.random() - 0.5) * randomFactor;
        
        return reflected.normalize();
    }
    
    /**
     * Calculate disk temperature at given radius
     */
    calculateDiskTemperature(radius) {
        const innerRadius = 1.5;
        const outerRadius = 3.0;
        
        if (radius < innerRadius) return 1.0;
        if (radius > outerRadius) return 0.1;
        
        return 1.0 - (radius - innerRadius) / (outerRadius - innerRadius);
    }
    
    /**
     * Check if ray is captured by black hole
     */
    isRayCaptured(ray) {
        const distance = ray.position.length();
        const mass = this.blackHole.userData.mass || 1.0;
        const spin = this.blackHole.userData.spin || 0.5;
        
        const eventHorizon = KerrPhysics.kerrEventHorizon(mass, spin);
        return distance <= eventHorizon;
    }
    
    /**
     * Check if ray has escaped
     */
    isRayEscaped(ray) {
        const distance = ray.position.length();
        return distance > this.rayLength;
    }
    
    /**
     * Update ray trail visualization
     */
    updateRayTrail(ray) {
        const trail = this.rayTrails.find(t => t.userData.ray === ray);
        if (!trail || ray.trail.length < 2) return;
        
        // Create thick ray using tube geometry
        if (ray.trail.length >= 2) {
            // Create a smooth curve through the trail points
            const curve = new THREE.CatmullRomCurve3(ray.trail);
            
            // Create tube geometry with thickness
            const tubeGeometry = new THREE.TubeGeometry(curve, ray.trail.length * 2, this.rayThickness, 8, false);
            
            // Dispose old geometry
            trail.geometry.dispose();
            trail.geometry = tubeGeometry;
            
            // Update material
            trail.material.color.copy(ray.color);
            trail.material.opacity = Math.min(ray.energy * 0.4, 0.8);
        }
    }
    
    /**
     * Regenerate rays periodically
     */
    regenerateRays() {
        // Remove old inactive rays
        this.rays = this.rays.filter(ray => ray.active);
        
        // Generate new rays to maintain count
        const currentCount = this.rays.length;
        const targetCount = this.maxRays;
        
        if (currentCount < targetCount) {
            const needed = targetCount - currentCount;
            this.generateAdditionalRays(needed);
        }
    }
    
    /**
     * Generate additional rays
     */
    generateAdditionalRays(count) {
        const cameraPosition = this.camera.position.clone();
        const cameraDirection = new THREE.Vector3();
        this.camera.getWorldDirection(cameraDirection);
        
        for (let i = 0; i < count; i++) {
            // Randomly select cluster
            const clusterNames = Object.keys(this.clusters);
            const clusterName = clusterNames[Math.floor(Math.random() * clusterNames.length)];
            const cluster = this.clusters[clusterName];
            
            const ray = this.createRay(clusterName, cluster, cameraPosition, cameraDirection);
            this.rays.push(ray);
            this.createRayTrail(ray);
        }
    }
    
    /**
     * Clear all rays
     */
    clearRays() {
        this.rays.forEach(ray => {
            ray.trail = [];
        });
        
        this.rayTrails.forEach(trail => {
            this.scene.remove(trail);
            trail.geometry.dispose();
            trail.material.dispose();
        });
        
        this.rays = [];
        this.rayTrails = [];
    }
    
    /**
     * Set ray thickness
     */
    setRayThickness(thickness) {
        this.rayThickness = thickness;
        // Regenerate all trails with new thickness
        this.rays.forEach(ray => {
            this.updateRayTrail(ray);
        });
    }
    
    /**
     * Set ray count
     */
    setRayCount(count) {
        this.maxRays = count;
        this.generateRayClusters();
    }
    
    /**
     * Toggle ray visibility
     */
    setVisible(visible) {
        this.rayTrails.forEach(trail => {
            trail.visible = visible;
        });
    }
    
    /**
     * Update black hole reference
     */
    updateBlackHole(blackHole) {
        this.blackHole = blackHole;
    }
    
    /**
     * Get ray statistics
     */
    getStats() {
        const activeRays = this.rays.filter(ray => ray.active).length;
        const totalRays = this.rays.length;
        
        return {
            activeRays,
            totalRays,
            trails: this.rayTrails.length
        };
    }
}
