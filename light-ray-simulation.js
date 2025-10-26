/**
 * Schwarzschild Black Hole Light Ray Simulation
 * Implements the physics from Track A instructions with Three.js
 */

class SchwarzschildLightRaySimulation {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        // Schwarzschild parameters (geometric units: G = c = 1)
        this.physics = {
            mass: 1.0,                    // Black hole mass M
            observerDistance: 10.0,        // Observer distance r_obs
            fieldOfView: 60.0,            // Field of view in degrees
            criticalImpactParameter: 0,   // b_crit = √27M
            eventHorizon: 0,               // r = 2M
            photonSphere: 0,              // r = 3M
            capturedRays: 0,               // Count of captured rays
            totalRays: 50                  // Total number of light rays
        };
        
        // Light ray objects
        this.lightRays = [];
        this.rayTrails = [];
        this.backgroundStars = [];
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupCamera();
        this.createRedSphere();
        this.createBackground();
        this.createLightRays();
        this.setupControls();
        this.setupEventListeners();
        this.updatePhysics();
        this.animate();
        
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('container').appendChild(this.renderer.domElement);
    }
    
    setupCamera() {
        this.camera.position.set(0, 2, this.physics.observerDistance);
        this.camera.lookAt(0, 0, 0);
    }
    
    createRedSphere() {
        // Red spherical object
        const sphereGeometry = new THREE.SphereGeometry(this.physics.mass, 32, 32);
        const sphereMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000, // Bright red
            transparent: true,
            opacity: 0.9
        });
        
        this.redSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.redSphere.castShadow = true;
        this.scene.add(this.redSphere);
        
        // Add a subtle glow effect
        const glowGeometry = new THREE.SphereGeometry(this.physics.mass * 1.1, 32, 32);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0xff4444,
            transparent: true,
            opacity: 0.2,
            side: THREE.BackSide
        });
        
        this.glowSphere = new THREE.Mesh(glowGeometry, glowMaterial);
        this.scene.add(this.glowSphere);
    }
    
    
    createBackground() {
        // Create procedural starfield
        const starGeometry = new THREE.BufferGeometry();
        const starCount = 1000;
        const positions = new Float32Array(starCount * 3);
        const colors = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount; i++) {
            const i3 = i * 3;
            
            // Random positions in a large sphere
            const radius = 50 + Math.random() * 100;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);
            
            // Random star colors (white to blue-white)
            const colorIntensity = 0.5 + Math.random() * 0.5;
            colors[i3] = colorIntensity;
            colors[i3 + 1] = colorIntensity;
            colors[i3 + 2] = colorIntensity + Math.random() * 0.3;
        }
        
        starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        starGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const starMaterial = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        
        this.starfield = new THREE.Points(starGeometry, starMaterial);
        this.scene.add(this.starfield);
    }
    
    // Schwarzschild physics functions
    calculateImpactParameter(r0, psi) {
        // b = (r0 * sin(psi)) / sqrt(1 - 2M/r0)
        const M = this.physics.mass;
        return (r0 * Math.sin(psi)) / Math.sqrt(1 - 2 * M / r0);
    }
    
    weakFieldDeflection(b) {
        // α(b) ≈ 4M/b + 15πM²/4b²
        const M = this.physics.mass;
        const bClamped = Math.max(b, this.physics.criticalImpactParameter + 0.05);
        return 4 * M / bClamped + (15 * Math.PI * M * M) / (4 * bClamped * bClamped);
    }
    
    isRayCaptured(b) {
        return b < this.physics.criticalImpactParameter;
    }
    
    createLightRays() {
        this.lightRays = [];
        this.rayTrails = [];
        
        const raysPerRing = 10;
        const rings = 5;
        
        for (let ring = 0; ring < rings; ring++) {
            const radius = this.physics.observerDistance + (ring + 1) * 2;
            
            for (let i = 0; i < raysPerRing; i++) {
                const angle = (i / raysPerRing) * Math.PI * 2;
                
                // Create light ray geometry (elongated cylinder)
                const rayGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.5, 8);
                const rayMaterial = new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    emissive: 0x222222,
                    transparent: true,
                    opacity: 0.9
                });
                
                const ray = new THREE.Mesh(rayGeometry, rayMaterial);
                
                // Position ray
                ray.position.x = Math.cos(angle) * radius;
                ray.position.z = Math.sin(angle) * radius;
                ray.position.y = (Math.random() - 0.5) * 2;
                
                // Calculate impact parameter
                const psi = Math.atan2(ray.position.z, ray.position.x);
                const impactParameter = this.calculateImpactParameter(radius, psi);
                
                // Store ray data
                ray.userData = {
                    originalPosition: ray.position.clone(),
                    impactParameter: impactParameter,
                    isCaptured: this.isRayCaptured(impactParameter),
                    deflection: this.weakFieldDeflection(impactParameter),
                    radius: radius,
                    angle: angle,
                    speed: 0.02 + Math.random() * 0.01
                };
                
                // Orient ray
                ray.rotation.z = Math.PI / 2;
                
                this.lightRays.push(ray);
                this.scene.add(ray);
                
                // Create trail
                this.createRayTrail(ray);
            }
        }
        
        this.updateCapturedCount();
    }
    
    createRayTrail(ray) {
        const trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.LineBasicMaterial({
            color: ray.userData.isCaptured ? 0xff0000 : 0xffffff,
            transparent: true,
            opacity: 0.3
        });
        
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        trail.userData = {
            ray: ray,
            points: [],
            maxPoints: 100
        };
        
        this.rayTrails.push(trail);
        this.scene.add(trail);
    }
    
    updateLightRays() {
        this.lightRays.forEach(ray => {
            const userData = ray.userData;
            
            if (userData.isCaptured) {
                // Captured rays spiral into black hole
                const time = Date.now() * 0.001;
                const spiralRadius = userData.radius * Math.exp(-time * 0.1);
                const spiralAngle = userData.angle + time * 2;
                
                ray.position.x = Math.cos(spiralAngle) * spiralRadius;
                ray.position.z = Math.sin(spiralAngle) * spiralRadius;
                ray.position.y *= Math.exp(-time * 0.05);
                
                // Fade out as approaching red sphere
                const distanceToCenter = Math.sqrt(
                    ray.position.x * ray.position.x + 
                    ray.position.z * ray.position.z + 
                    ray.position.y * ray.position.y
                );
                const fadeFactor = Math.max(0, (distanceToCenter - this.physics.mass) / 2);
                ray.material.opacity = fadeFactor;
                
                // Remove if too close to red sphere
                if (distanceToCenter < this.physics.mass * 1.1) {
                    ray.visible = false;
                }
            } else {
                // Escaped rays follow deflected path
                const time = Date.now() * 0.001;
                const deflection = userData.deflection;
                const newAngle = userData.angle + deflection + time * userData.speed;
                
                ray.position.x = Math.cos(newAngle) * userData.radius;
                ray.position.z = Math.sin(newAngle) * userData.radius;
                
                // Add slight vertical oscillation
                ray.position.y = (Math.random() - 0.5) * 2 + Math.sin(time * 2) * 0.5;
            }
            
            // Update trail
            this.updateRayTrail(ray);
        });
    }
    
    updateRayTrail(ray) {
        const trail = this.rayTrails.find(t => t.userData.ray === ray);
        if (!trail) return;
        
        const trailData = trail.userData;
        const position = ray.position.clone();
        
        // Add current position to trail
        trailData.points.push(position);
        
        // Limit trail length
        if (trailData.points.length > trailData.maxPoints) {
            trailData.points.shift();
        }
        
        // Update trail geometry
        if (trailData.points.length > 1) {
            const positions = new Float32Array(trailData.points.length * 3);
            trailData.points.forEach((point, index) => {
                positions[index * 3] = point.x;
                positions[index * 3 + 1] = point.y;
                positions[index * 3 + 2] = point.z;
            });
            
            trail.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            trail.geometry.attributes.position.needsUpdate = true;
        }
    }
    
    updateCapturedCount() {
        this.physics.capturedRays = this.lightRays.filter(ray => ray.userData.isCaptured).length;
        this.updateParameterDisplay();
    }
    
    updatePhysics() {
        const M = this.physics.mass;
        this.physics.criticalImpactParameter = Math.sqrt(27) * M;
        this.physics.eventHorizon = 2 * M;
        this.physics.photonSphere = 3 * M;
        
        // Update red sphere geometry
        if (this.redSphere) {
            this.redSphere.geometry.dispose();
            this.redSphere.geometry = new THREE.SphereGeometry(this.physics.mass, 32, 32);
        }
        
        if (this.glowSphere) {
            this.glowSphere.geometry.dispose();
            this.glowSphere.geometry = new THREE.SphereGeometry(this.physics.mass * 1.1, 32, 32);
        }
        
        this.updateParameterDisplay();
    }
    
    updateParameterDisplay() {
        document.getElementById('mass-value').textContent = this.physics.mass.toFixed(1);
        document.getElementById('distance-value').textContent = this.physics.observerDistance.toFixed(1);
        document.getElementById('rays-value').textContent = this.lightRays.length;
        document.getElementById('fov-value').textContent = this.physics.fieldOfView.toFixed(0) + '°';
        
        document.getElementById('horizon-value').textContent = this.physics.mass.toFixed(1);
        document.getElementById('photon-value').textContent = this.physics.eventHorizon.toFixed(1);
        document.getElementById('critical-value').textContent = this.physics.criticalImpactParameter.toFixed(1);
        document.getElementById('captured-value').textContent = this.physics.capturedRays;
    }
    
    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 50;
        this.controls.maxPolarAngle = Math.PI;
    }
    
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowUp':
                    this.physics.mass = Math.min(5.0, this.physics.mass + 0.1);
                    this.updatePhysics();
                    break;
                case 'ArrowDown':
                    this.physics.mass = Math.max(0.1, this.physics.mass - 0.1);
                    this.updatePhysics();
                    break;
                case 'ArrowLeft':
                    this.physics.observerDistance = Math.max(5.0, this.physics.observerDistance - 0.5);
                    this.camera.position.z = this.physics.observerDistance;
                    break;
                case 'ArrowRight':
                    this.physics.observerDistance = Math.min(30.0, this.physics.observerDistance + 0.5);
                    this.camera.position.z = this.physics.observerDistance;
                    break;
                case ' ':
                    e.preventDefault();
                    this.resetSimulation();
                    break;
            }
        });
    }
    
    resetSimulation() {
        // Remove existing rays
        this.lightRays.forEach(ray => this.scene.remove(ray));
        this.rayTrails.forEach(trail => this.scene.remove(trail));
        
        // Recreate rays
        this.createLightRays();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Rotate starfield slowly
        if (this.starfield) {
            this.starfield.rotation.y += 0.0005;
        }
        
        // Rotate red sphere slowly
        if (this.redSphere) {
            this.redSphere.rotation.y += 0.01;
            this.redSphere.rotation.x += 0.005;
        }
        
        // Update light rays
        this.updateLightRays();
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Initialize simulation when page loads
window.addEventListener('load', () => {
    new SchwarzschildLightRaySimulation();
});
