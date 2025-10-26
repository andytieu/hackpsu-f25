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
            maxPhotons: 30
        };
        
        // Arrays for photons and trails
        this.photons = [];
        this.photonTrails = [];
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.createGravitationalObject();
        this.setupLighting();
        this.createPhotons();
        this.setupCamera();
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
    
    createPhotonTrail(photon) {
        TrailManager.createTrail(photon, this.scene, this.photonTrails, {
            color: 0xffff00,
            opacity: 0.3,
            maxPoints: 50
        });
    }
    
    calculateGravitationalForce(photon) {
        return PhysicsUtils.calculateGravitationalForce(photon, this.gravityParams);
    }
    
    updatePhotons() {
        this.photons.forEach(photon => {
            const userData = photon.userData;

            // Calculate gravitational force
            const gravitationalForce = this.calculateGravitationalForce(photon);

            // Apply gravitational force to velocity
            userData.velocity.add(gravitationalForce);

            // Apply velocity to position
            photon.position.add(userData.velocity);

            // Add orbital motion
            const orbitalForce = PhysicsUtils.calculateOrbitalForce(photon.position, userData.orbitalSpeed);
            userData.velocity.add(orbitalForce);

            // Limit velocity to prevent runaway acceleration
            userData.velocity = PhysicsUtils.limitVelocity(userData.velocity, 0.4);

            // Add slight vertical oscillation
            photon.position.y = PhysicsUtils.addVerticalOscillation(photon.position, userData.angle, 0.001);

            // Update trail
            TrailManager.updateTrail(photon, this.photonTrails);

            // Reset photon if it gets too far or too close
            const distance = photon.position.length();
            if (distance > 11.25 || distance < 1.5) { // Adjusted by factor of 0.75
                this.resetPhoton(photon);
            }
        });
    }
    
    
    resetPhoton(photon) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 3 + Math.random() * 2.25; // Decreased by factor of 0.75
        const height = (Math.random() - 0.5) * 1.5; // Decreased by factor of 0.75

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
    
    setupCamera() {
        // Position camera
        this.camera.position.set(0, 3, 8);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupEventListeners() {
        // Handle window resize
        UIManager.setupResizeHandler(this.camera, this.renderer);

        // Keyboard controls
        UIManager.setupKeyboardControls(this.gravityParams);
    }
    
    updateUI() {
        UIManager.updateUI({
            photonCount: this.photons.length,
            gravitationalConstant: this.gravityParams.gravitationalConstant
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());

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
