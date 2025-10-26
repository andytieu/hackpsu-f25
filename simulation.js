/**
 * Gravitational Red Sphere Simulation
 * Main simulation class for photons orbiting around a massive object
 */

class GravitationalSphereSimulation {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        
        // Gravitational parameters
        this.gravityParams = {
            mass: 1.0,
            gravitationalConstant: 0.1,
            maxPhotons: 15, // Reduced for first cluster
            maxPhotons2: 15 // Second cluster
        };
        
        // Arrays for photons and trails
        this.photons = [];
        this.photonTrails = [];
        this.photons2 = []; // Second cluster
        this.photonTrails2 = []; // Second cluster trails
        
        // Orbital camera parameters
        this.cameraRadius = 8;
        this.cameraAngleX = Math.PI / 3; // 60 degrees
        this.cameraAngleY = 0;
        this.mouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.autoRotate = false;
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.createGravitationalObject();
        this.setupLighting();
        this.createPhotons();
        this.createPhotons2();
        this.setupCamera();
        this.setupOrbitalCamera();
        this.setupEventListeners();
        this.animate();
    }
    
    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        document.body.appendChild(this.renderer.domElement);
    }
    
    createGravitationalObject() {
        // Create red sphere (gravitational object)
        const sphereGeometry = new THREE.SphereGeometry(1.2, 32, 32);
        const sphereMaterial = new THREE.MeshLambertMaterial({
            color: 0xff0000,
            emissive: 0x220000
        });
        this.gravitationalObject = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.gravitationalObject);
    }
    
    setupLighting() {
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);
    }
    
    createPhotons() {
        for (let i = 0; i < this.gravityParams.maxPhotons; i++) {
            // Random starting position around the sphere
            const angle = (i / this.gravityParams.maxPhotons) * Math.PI * 2;
            const radius = 2.25 + Math.random() * 3; // Decreased by factor of 0.75
            const height = (Math.random() - 0.5) * 1.5; // Decreased by factor of 0.75

            // Create photon geometry (small sphere)
            const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
            const photonMaterial = new THREE.MeshBasicMaterial({
                color: 0xffff00,
                emissive: 0x222200
            });
            const photon = new THREE.Mesh(photonGeometry, photonMaterial);

            // Set initial position
            photon.position.x = Math.cos(angle) * radius;
            photon.position.z = Math.sin(angle) * radius;
            photon.position.y = height;

            // Store photon data
            photon.userData = {
                angle: angle,
                radius: radius,
                height: height,
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.08,
                    (Math.random() - 0.5) * 0.04,
                    (Math.random() - 0.5) * 0.08
                ),
                orbitalSpeed: 0.04 + Math.random() * 0.08,
                trail: []
            };

            this.photons.push(photon);
            this.scene.add(photon);

            // Create trail
            this.createPhotonTrail(photon);
        }
    }
    
    createPhotons2() {
        for (let i = 0; i < this.gravityParams.maxPhotons2; i++) {
            // Create photons in a perpendicular orbital plane (around Y-axis)
            const angle = (i / this.gravityParams.maxPhotons2) * Math.PI * 2;
            const radius = 2.5 + Math.random() * 2.5; // Similar radius range
            const height = (Math.random() - 0.5) * 0.8; // Smaller height variation

            // Create photon geometry (small sphere) - different color for second cluster
            const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
            const photonMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ffff, // Cyan color for second cluster
                emissive: 0x002222
            });
            const photon = new THREE.Mesh(photonGeometry, photonMaterial);

            // Set initial position in perpendicular plane (Y-Z plane)
            photon.position.x = height; // Use height as X coordinate
            photon.position.y = Math.cos(angle) * radius; // Y becomes the orbital plane
            photon.position.z = Math.sin(angle) * radius; // Z remains orbital

            // Store photon data
            photon.userData = {
                angle: angle,
                radius: radius,
                height: height,
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.06, // Different velocity range
                    (Math.random() - 0.5) * 0.08,
                    (Math.random() - 0.5) * 0.06
                ),
                orbitalSpeed: 0.03 + Math.random() * 0.06, // Different orbital speed
                trail: []
            };

            this.photons2.push(photon);
            this.scene.add(photon);

            // Create trail for second cluster
            this.createPhotonTrail2(photon);
        }
    }
    
    createPhotonTrail(photon) {
        TrailManager.createTrail(photon, this.scene, this.photonTrails, {
            color: 0xffff00,
            opacity: 0.3,
            maxPoints: 50
        });
    }
    
    createPhotonTrail2(photon) {
        TrailManager.createTrail(photon, this.scene, this.photonTrails2, {
            color: 0x00ffff, // Cyan trails for second cluster
            opacity: 0.3,
            maxPoints: 50
        });
    }
    
    calculateGravitationalForce(photon) {
        return PhysicsUtils.calculateGravitationalForce(photon, this.gravityParams);
    }
    
    updatePhotons() {
        // Update first cluster (yellow photons)
        this.photons.forEach(photon => {
            const userData = photon.userData;

            // Calculate gravitational force
            const gravitationalForce = this.calculateGravitationalForce(photon);

            // Apply gravitational force to velocity
            userData.velocity.add(gravitationalForce);

            // Apply velocity to position
            photon.position.add(userData.velocity);

            // Add orbital motion (X-Z plane)
            const orbitalForce = PhysicsUtils.calculateOrbitalForce(photon.position, userData.orbitalSpeed);
            userData.velocity.add(orbitalForce);

            // Limit velocity to prevent runaway acceleration
            userData.velocity = PhysicsUtils.limitVelocity(userData.velocity, 0.4);

            // Add slight vertical oscillation
            photon.position.y = PhysicsUtils.addVerticalOscillation(photon.position, userData.angle, 0.001);
            TrailManager.updateTrail(photon, this.photonTrails);

            // Reset photon if it gets too far or too close
            const distance = photon.position.length();
            if (distance > 11.25 || distance < 1.5) {
                this.resetPhoton(photon);
            }
        });

        // Update second cluster (cyan photons)
        this.photons2.forEach(photon => {
            const userData = photon.userData;

            // Calculate gravitational force
            const gravitationalForce = this.calculateGravitationalForce(photon);

            // Apply gravitational force to velocity
            userData.velocity.add(gravitationalForce);

            // Apply velocity to position
            photon.position.add(userData.velocity);

            // Add orbital motion (Y-Z plane) - perpendicular to first cluster
            const orbitalForce = new THREE.Vector3(
                -photon.position.y * userData.orbitalSpeed,
                photon.position.x * userData.orbitalSpeed,
                0
            );
            userData.velocity.add(orbitalForce);

            // Limit velocity to prevent runaway acceleration
            userData.velocity = PhysicsUtils.limitVelocity(userData.velocity, 0.4);

            // Add slight oscillation in X direction
            photon.position.x += Math.sin(Date.now() * 0.001 + userData.angle) * 0.001;
            TrailManager.updateTrail(photon, this.photonTrails2);

            // Reset photon if it gets too far or too close
            const distance = photon.position.length();
            if (distance > 11.25 || distance < 1.5) {
                this.resetPhoton2(photon);
            }
        });
    }
    
    
    resetPhoton(photon) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 3 + Math.random() * 2.25; 
        const height = (Math.random() - 0.5) * 1.5;

        photon.position.x = Math.cos(angle) * radius;
        photon.position.z = Math.sin(angle) * radius;
        photon.position.y = height;

        photon.userData.velocity.set(
            (Math.random() - 0.5) * 0.08,
            (Math.random() - 0.5) * 0.04,
            (Math.random() - 0.5) * 0.08
        );
        TrailManager.clearTrail(photon);
    }
    
    resetPhoton2(photon) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 3 + Math.random() * 2.25;
        const height = (Math.random() - 0.5) * 0.8;

        // Reset position in perpendicular plane (Y-Z plane)
        photon.position.x = height;
        photon.position.y = Math.cos(angle) * radius;
        photon.position.z = Math.sin(angle) * radius;

        photon.userData.velocity.set(
            (Math.random() - 0.5) * 0.06,
            (Math.random() - 0.5) * 0.08,
            (Math.random() - 0.5) * 0.06
        );
        TrailManager.clearTrail(photon);
    }
    
    setupCamera() {
        // Initial camera position (will be overridden by orbital camera)
        this.camera.position.set(0, 3, 8);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupOrbitalCamera() {
        // Set initial camera position using spherical coordinates
        this.updateCameraPosition();
        
        // Add mouse event listeners for orbital control
        this.renderer.domElement.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('mouseup', () => this.onMouseUp());
        this.renderer.domElement.addEventListener('wheel', (e) => this.onWheel(e));
        
        // Touch events for mobile
        this.renderer.domElement.addEventListener('touchstart', (e) => this.onTouchStart(e));
        this.renderer.domElement.addEventListener('touchmove', (e) => this.onTouchMove(e));
        this.renderer.domElement.addEventListener('touchend', () => this.onTouchEnd());
    }
    
    // Orbital camera methods
    updateCameraPosition() {
        const x = this.cameraRadius * Math.sin(this.cameraAngleX) * Math.cos(this.cameraAngleY);
        const y = this.cameraRadius * Math.cos(this.cameraAngleX);
        const z = this.cameraRadius * Math.sin(this.cameraAngleX) * Math.sin(this.cameraAngleY);
        
        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 0, 0);
    }
    
    onMouseDown(event) {
        this.mouseDown = true;
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
        event.preventDefault();
    }
    
    onMouseMove(event) {
        if (!this.mouseDown) return;
        
        const deltaX = event.clientX - this.lastMouseX;
        const deltaY = event.clientY - this.lastMouseY;
        
        this.cameraAngleY -= deltaX * 0.01;
        this.cameraAngleX += deltaY * 0.01;
        
        // Clamp vertical angle to prevent camera flipping
        this.cameraAngleX = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraAngleX));
        
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
        
        event.preventDefault();
    }
    
    onMouseUp() {
        this.mouseDown = false;
    }
    
    onWheel(event) {
        const delta = event.deltaY;
        this.cameraRadius += delta * 0.1;
        this.cameraRadius = Math.max(2, Math.min(20, this.cameraRadius));
        event.preventDefault();
    }
    
    onTouchStart(event) {
        if (event.touches.length === 1) {
            this.mouseDown = true;
            this.lastMouseX = event.touches[0].clientX;
            this.lastMouseY = event.touches[0].clientY;
        }
    }
    
    onTouchMove(event) {
        if (!this.mouseDown || event.touches.length !== 1) return;
        
        const deltaX = event.touches[0].clientX - this.lastMouseX;
        const deltaY = event.touches[0].clientY - this.lastMouseY;
        
        this.cameraAngleY -= deltaX * 0.01;
        this.cameraAngleX += deltaY * 0.01;
        
        this.cameraAngleX = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraAngleX));
        
        this.lastMouseX = event.touches[0].clientX;
        this.lastMouseY = event.touches[0].clientY;
    }
    
    onTouchEnd() {
        this.mouseDown = false;
    }
    
    setupEventListeners() {
        UIManager.setupResizeHandler(this.camera, this.renderer);

        UIManager.setupKeyboardControls(this.gravityParams);
    }
    
    updateUI() {
        UIManager.updateUI({
            photonCount: this.photons.length + this.photons2.length,
            gravitationalConstant: this.gravityParams.gravitationalConstant
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());

        // Update orbital camera position
        this.updateCameraPosition();

        // Rotate gravitational object
        this.gravitationalObject.rotation.y += 0.005;
        this.gravitationalObject.rotation.x += 0.002;

        // Update photon physics
        this.updatePhotons();

        // Update UI
        this.updateUI();

        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize simulation when page loads
window.addEventListener('load', () => {
    new GravitationalSphereSimulation();
});
