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
            mass: 3.0,
            gravitationalConstant: 0.1,
            maxPhotons: 200, 
            maxPhotons2: 200,
            useRelativisticPhysics: true, // Enable Kerr physics
            showEventHorizon: true, // Show event horizon
            showErgosphere: true, // Show ergosphere
            spin: 0.5 // Kerr spin parameter (-1 to 1)
        };
        
        // Arrays for photons and trails
        this.photons = [];
        this.photonTrails = [];
        this.photons2 = []; // Second cluster
        this.photonTrails2 = []; // Second cluster trails
        
        // Background starfield for gravitational lensing
        this.backgroundStarfield = null;
        
        // Accretion disk renderer
        this.accretionDiskRenderer = null;
        
        // Spacetime curvature grid
        this.spacetimeGrid = null;
        
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
        this.createBackgroundStarfield();
        this.createAccretionDiskRenderer();
        this.createSpacetimeGrid();
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
        // Create invisible gravitational object for physics calculations
        this.gravitationalObject = new THREE.Object3D();
        this.gravitationalObject.userData = {
            mass: this.gravityParams.mass,
            spin: this.gravityParams.spin
        };
        this.scene.add(this.gravitationalObject);
        
        // Create event horizon visualization
        if (this.gravityParams.showEventHorizon) {
            this.createEventHorizon();
        }
    }
    
    createEventHorizon() {
        const rs = KerrPhysics.kerrEventHorizon(this.gravityParams.mass, this.gravityParams.spin);
        
        // Create event horizon sphere
        const eventHorizonGeometry = new THREE.SphereGeometry(rs, 32, 32);
        const eventHorizonMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        
        this.eventHorizon = new THREE.Mesh(eventHorizonGeometry, eventHorizonMaterial);
        this.scene.add(this.eventHorizon);
        
        // Create ergosphere visualization
        if (this.gravityParams.showErgosphere) {
            this.createErgosphere();
        }
        
        // Create Kerr ISCO visualization
        const iscoRadius = KerrPhysics.kerrISCO(this.gravityParams.mass, this.gravityParams.spin, true);
        const iscoGeometry = new THREE.SphereGeometry(iscoRadius, 32, 32);
        const iscoMaterial = new THREE.MeshBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 0.05,
            wireframe: true
        });
        
        this.isco = new THREE.Mesh(iscoGeometry, iscoMaterial);
        this.scene.add(this.isco);
        
        // Create Kerr photon sphere visualization
        const photonSphereRadius = KerrPhysics.kerrPhotonSphere(this.gravityParams.mass, this.gravityParams.spin, true);
        const photonSphereGeometry = new THREE.SphereGeometry(photonSphereRadius, 32, 32);
        const photonSphereMaterial = new THREE.MeshBasicMaterial({
            color: 0x444444,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });
        
        this.photonSphere = new THREE.Mesh(photonSphereGeometry, photonSphereMaterial);
        this.scene.add(this.photonSphere);
    }
    
    createErgosphere() {
        // Don't create ergosphere if spin is 0 (no rotation)
        if (Math.abs(this.gravityParams.spin) < 0.01) {
            return;
        }
        
        // Create ergosphere as a distorted sphere
        const ergosphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        
        // Modify vertices to create ergosphere shape
        const vertices = ergosphereGeometry.attributes.position.array;
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
            
            const r = Math.sqrt(x * x + y * y + z * z);
            const theta = Math.acos(y / r);
            
            const ergosphereRadius = KerrPhysics.kerrErgosphere(this.gravityParams.mass, this.gravityParams.spin, theta);
            const scale = ergosphereRadius / r;
            
            vertices[i] *= scale;
            vertices[i + 1] *= scale;
            vertices[i + 2] *= scale;
        }
        
        ergosphereGeometry.attributes.position.needsUpdate = true;
        
        const ergosphereMaterial = new THREE.MeshBasicMaterial({
            color: 0x440044,
            transparent: true,
            opacity: 0.3,
            wireframe: true,
            side: THREE.DoubleSide
        });
        
        this.ergosphere = new THREE.Mesh(ergosphereGeometry, ergosphereMaterial);
        this.scene.add(this.ergosphere);
    }
    
    createBackgroundStarfield() {
        // Create background starfield for gravitational lensing
        this.backgroundStarfield = new BackgroundStarfield(
            this.scene,
            new THREE.Vector3(0, 0, 0), // Black hole position
            this.gravityParams.mass,
            this.gravityParams.spin
        );
    }
    
    createAccretionDiskRenderer() {
        // Create accretion disk with halos
        this.accretionDiskRenderer = new AccretionDiskRenderer(this.scene, this.gravitationalObject, null);
    }
    
    createSpacetimeGrid() {
        // Create spacetime curvature grid
        this.spacetimeGrid = new SpacetimeCurvatureGrid(this.scene, this.gravitationalObject);
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

            // Set initial position around black hole in orbital plane
            photon.position.x = Math.cos(angle) * radius;
            photon.position.z = Math.sin(angle) * radius;
            photon.position.y = height;
            
            // Calculate proper initial velocity for Kerr orbit (tangential to orbit)
            const tangentDirection = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle));
            const orbitalVelocity = Math.sqrt(this.gravityParams.mass / radius); // Circular orbit velocity
            const initialVelocity = tangentDirection.multiplyScalar(orbitalVelocity);
            
            // Store photon data
            photon.userData = {
                angle: angle,
                radius: radius,
                height: height,
                velocity: initialVelocity,
                orbitalSpeed: orbitalVelocity,
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
            
            // Calculate proper initial velocity for Kerr orbit in perpendicular plane
            const tangentDirection = new THREE.Vector3(0, -Math.sin(angle), Math.cos(angle));
            const orbitalVelocity = Math.sqrt(this.gravityParams.mass / radius); // Circular orbit velocity
            const initialVelocity = tangentDirection.multiplyScalar(orbitalVelocity);
            
            // Store photon data
            photon.userData = {
                angle: angle,
                radius: radius,
                height: height,
                velocity: initialVelocity,
                orbitalSpeed: orbitalVelocity,
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
        if (this.gravityParams.useRelativisticPhysics) {
            return KerrPhysics.calculateKerrForce(photon, this.gravityParams);
        } else {
            return PhysicsUtils.calculateGravitationalForce(photon, this.gravityParams);
        }
    }
    
    updatePhotons() {
        // Update first cluster (yellow photons)
        this.photons.forEach(photon => {
            const userData = photon.userData;

            // Check if photon is captured by event horizon
            if (KerrPhysics.isInsideEventHorizon(photon.position, this.gravityParams.mass, this.gravityParams.spin)) {
                this.resetPhoton(photon);
                return;
            }

            if (this.gravityParams.useRelativisticPhysics) {
                // Use Kerr geodesic integration
                const result = KerrGeodesicIntegrator.integrateKerrGeodesic(
                    photon.position,
                    userData.velocity,
                    this.gravityParams.mass,
                    this.gravityParams.spin,
                    0.016 // ~60 FPS
                );
                
                if (result) {
                    photon.position.copy(result.position);
                    userData.velocity.copy(result.velocity);
                    
                    // Apply relativistic velocity limit
                    userData.velocity = RelativisticPhysics.limitRelativisticVelocity(userData.velocity);
                } else {
                    // Unstable integration - reset photon
                    this.resetPhoton(photon);
                    return;
                }
            } else {
                // Classical physics (original code)
                const gravitationalForce = this.calculateGravitationalForce(photon);
                userData.velocity.add(gravitationalForce);
                photon.position.add(userData.velocity);
                
                const orbitalForce = PhysicsUtils.calculateOrbitalForce(photon.position, userData.orbitalSpeed);
                userData.velocity.add(orbitalForce);
                userData.velocity = PhysicsUtils.limitVelocity(userData.velocity, 0.4);
                photon.position.y = PhysicsUtils.addVerticalOscillation(photon.position, userData.angle, 0.001);
            }

            TrailManager.updateTrail(photon, this.photonTrails);

            // Reset photon if it gets too far
            const distance = photon.position.length();
            if (distance > 20) {
                this.resetPhoton(photon);
            }
        });

        // Update second cluster (cyan photons)
        this.photons2.forEach(photon => {
            const userData = photon.userData;

            // Check if photon is captured by event horizon
            if (KerrPhysics.isInsideEventHorizon(photon.position, this.gravityParams.mass, this.gravityParams.spin)) {
                this.resetPhoton2(photon);
                return;
            }

            if (this.gravityParams.useRelativisticPhysics) {
                // Use Kerr geodesic integration
                const result = KerrGeodesicIntegrator.integrateKerrGeodesic(
                    photon.position,
                    userData.velocity,
                    this.gravityParams.mass,
                    this.gravityParams.spin,
                    0.016 // ~60 FPS
                );
                
                if (result) {
                    photon.position.copy(result.position);
                    userData.velocity.copy(result.velocity);
                    
                    // Apply relativistic velocity limit
                    userData.velocity = RelativisticPhysics.limitRelativisticVelocity(userData.velocity);
                } else {
                    // Unstable integration - reset photon
                    this.resetPhoton2(photon);
                    return;
                }
            } else {
                // Classical physics (original code)
                const gravitationalForce = this.calculateGravitationalForce(photon);
                userData.velocity.add(gravitationalForce);
                photon.position.add(userData.velocity);
                
                const orbitalForce = new THREE.Vector3(
                    -photon.position.y * userData.orbitalSpeed,
                    photon.position.x * userData.orbitalSpeed,
                    0
                );
                userData.velocity.add(orbitalForce);
                userData.velocity = PhysicsUtils.limitVelocity(userData.velocity, 0.4);
                photon.position.x += Math.sin(Date.now() * 0.001 + userData.angle) * 0.001;
            }

            TrailManager.updateTrail(photon, this.photonTrails2);

            // Reset photon if it gets too far
            const distance = photon.position.length();
            if (distance > 20) {
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
        
        // Setup slider event listeners
        this.setupSliderControls();
    }
    
    setupSliderControls() {
        // Photon count slider
        const photonSlider = document.getElementById('photon-count-slider');
        const photonValue = document.getElementById('photon-count-value');
        
        photonSlider.addEventListener('input', (e) => {
            const newCount = parseInt(e.target.value);
            photonValue.textContent = newCount;
            this.updatePhotonCount(newCount);
        });
        
        // Gravity slider
        const gravitySlider = document.getElementById('gravity-slider');
        const gravityValue = document.getElementById('gravity-value');
        
        gravitySlider.addEventListener('input', (e) => {
            const newGravity = parseFloat(e.target.value);
            gravityValue.textContent = newGravity.toFixed(2);
            this.gravityParams.gravitationalConstant = newGravity;
        });
        
        // Mass slider
        const massSlider = document.getElementById('mass-slider');
        const massValue = document.getElementById('mass-value');
        
        massSlider.addEventListener('input', (e) => {
            const newMass = parseFloat(e.target.value);
            massValue.textContent = newMass.toFixed(1);
            this.gravityParams.mass = newMass;
            this.gravitationalObject.userData.mass = newMass;
            this.updateEventHorizon();
        });
        
        // Spin slider
        const spinSlider = document.getElementById('spin-slider');
        const spinValue = document.getElementById('spin-value');
        
        spinSlider.addEventListener('input', (e) => {
            const newSpin = parseFloat(e.target.value);
            spinValue.textContent = newSpin.toFixed(2);
            this.gravityParams.spin = newSpin;
            this.gravitationalObject.userData.spin = newSpin;
            this.updateEventHorizon();
        });
        
        // Relativistic physics toggle
        const relativisticToggle = document.getElementById('relativistic-toggle');
        relativisticToggle.addEventListener('change', (e) => {
            this.gravityParams.useRelativisticPhysics = e.target.checked;
        });
        
        // Event horizon toggle
        const eventHorizonToggle = document.getElementById('event-horizon-toggle');
        eventHorizonToggle.addEventListener('change', (e) => {
            this.gravityParams.showEventHorizon = e.target.checked;
            this.toggleEventHorizon();
        });
        
        // Ergosphere toggle
        const ergosphereToggle = document.getElementById('ergosphere-toggle');
        if (ergosphereToggle) {
            ergosphereToggle.addEventListener('change', (e) => {
                this.gravityParams.showErgosphere = e.target.checked;
                this.toggleErgosphere();
            });
        }
        
        // Background starfield toggle
        const starfieldToggle = document.getElementById('starfield-toggle');
        starfieldToggle.addEventListener('change', (e) => {
            if (this.backgroundStarfield) {
                this.backgroundStarfield.setVisible(e.target.checked);
            }
        });
        
        // Accretion disk toggle
        const accretionDiskToggle = document.getElementById('accretion-disk-toggle');
        if (accretionDiskToggle) {
            accretionDiskToggle.addEventListener('change', (e) => {
                if (this.accretionDiskRenderer) {
                    this.accretionDiskRenderer.setVisible(e.target.checked);
                }
            });
        }
        
        // Spacetime grid toggle
        const spacetimeGridToggle = document.getElementById('spacetime-grid-toggle');
        if (spacetimeGridToggle) {
            spacetimeGridToggle.addEventListener('change', (e) => {
                if (this.spacetimeGrid) {
                    this.spacetimeGrid.setVisible(e.target.checked);
                }
            });
        }
    }
    
    updateEventHorizon() {
        if (this.eventHorizon) {
            const rs = KerrPhysics.kerrEventHorizon(this.gravityParams.mass, this.gravityParams.spin);
            this.eventHorizon.geometry.dispose();
            this.eventHorizon.geometry = new THREE.SphereGeometry(rs, 32, 32);
        }
        
        if (this.photonSphere) {
            const photonSphereRadius = KerrPhysics.kerrPhotonSphere(this.gravityParams.mass, this.gravityParams.spin, true);
            this.photonSphere.geometry.dispose();
            this.photonSphere.geometry = new THREE.SphereGeometry(photonSphereRadius, 32, 32);
        }
        
        if (this.isco) {
            const iscoRadius = KerrPhysics.kerrISCO(this.gravityParams.mass, this.gravityParams.spin, true);
            this.isco.geometry.dispose();
            this.isco.geometry = new THREE.SphereGeometry(iscoRadius, 32, 32);
        }
        
        if (this.ergosphere) {
            this.scene.remove(this.ergosphere);
            this.createErgosphere();
        }
        
        // Update background starfield
        if (this.backgroundStarfield) {
            this.backgroundStarfield.updateBlackHole(this.gravityParams.mass, this.gravityParams.spin);
        }
    }
    
    toggleErgosphere() {
        if (this.gravityParams.showErgosphere) {
            if (!this.ergosphere) {
                this.createErgosphere();
            } else {
                this.ergosphere.visible = true;
            }
        } else {
            if (this.ergosphere) {
                this.ergosphere.visible = false;
            }
        }
    }
    
    toggleEventHorizon() {
        if (this.gravityParams.showEventHorizon) {
            if (!this.eventHorizon) {
                this.createEventHorizon();
            } else {
                this.eventHorizon.visible = true;
                this.photonSphere.visible = true;
                this.isco.visible = true;
            }
        } else {
            if (this.eventHorizon) {
                this.eventHorizon.visible = false;
                this.photonSphere.visible = false;
                this.isco.visible = false;
            }
        }
    }
    
    updatePhotonCount(newCount) {
        // Update both clusters to have the same number of photons
        this.gravityParams.maxPhotons = newCount;
        this.gravityParams.maxPhotons2 = newCount;
        
        // Remove excess photons if reducing count
        while (this.photons.length > newCount) {
            const photon = this.photons.pop();
            this.scene.remove(photon);
            // Remove corresponding trail
            const trailIndex = this.photonTrails.findIndex(t => t.userData.photon === photon);
            if (trailIndex !== -1) {
                this.scene.remove(this.photonTrails[trailIndex]);
                this.photonTrails.splice(trailIndex, 1);
            }
        }
        
        while (this.photons2.length > newCount) {
            const photon = this.photons2.pop();
            this.scene.remove(photon);
            // Remove corresponding trail
            const trailIndex = this.photonTrails2.findIndex(t => t.userData.photon === photon);
            if (trailIndex !== -1) {
                this.scene.remove(this.photonTrails2[trailIndex]);
                this.photonTrails2.splice(trailIndex, 1);
            }
        }
        
        // Add photons if increasing count
        while (this.photons.length < newCount) {
            this.addSinglePhoton();
        }
        
        while (this.photons2.length < newCount) {
            this.addSinglePhoton2();
        }
    }
    
    addSinglePhoton() {
        const i = this.photons.length;
        const angle = (i / this.gravityParams.maxPhotons) * Math.PI * 2;
        const radius = 2.25 + Math.random() * 3;
        const height = (Math.random() - 0.5) * 1.5;

        const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
        const photonMaterial = new THREE.MeshBasicMaterial({
            color: 0xffff00,
            emissive: 0x222200
        });
        const photon = new THREE.Mesh(photonGeometry, photonMaterial);

        photon.position.x = Math.cos(angle) * radius;
        photon.position.z = Math.sin(angle) * radius;
        photon.position.y = height;

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
        this.createPhotonTrail(photon);
    }
    
    addSinglePhoton2() {
        const i = this.photons2.length;
        const angle = (i / this.gravityParams.maxPhotons2) * Math.PI * 2;
        const radius = 2.5 + Math.random() * 2.5;
        const height = (Math.random() - 0.5) * 0.8;

        const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
        const photonMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            emissive: 0x002222
        });
        const photon = new THREE.Mesh(photonGeometry, photonMaterial);

        photon.position.x = height;
        photon.position.y = Math.cos(angle) * radius;
        photon.position.z = Math.sin(angle) * radius;

        photon.userData = {
            angle: angle,
            radius: radius,
            height: height,
            velocity: new THREE.Vector3(
                (Math.random() - 0.5) * 0.06,
                (Math.random() - 0.5) * 0.08,
                (Math.random() - 0.5) * 0.06
            ),
            orbitalSpeed: 0.03 + Math.random() * 0.06,
            trail: []
        };

        this.photons2.push(photon);
        this.scene.add(photon);
        this.createPhotonTrail2(photon);
    }
    
    updateUI() {
        UIManager.updateUI({
            photonCount: this.photons.length + this.photons2.length,
            gravitationalConstant: this.gravityParams.gravitationalConstant
        });
        
        // Update Kerr information
        const rs = KerrPhysics.kerrEventHorizon(this.gravityParams.mass, this.gravityParams.spin);
        const r_ergosphere = KerrPhysics.kerrErgosphere(this.gravityParams.mass, this.gravityParams.spin, Math.PI/2);
        const photonSphereRadius = KerrPhysics.kerrPhotonSphere(this.gravityParams.mass, this.gravityParams.spin, true);
        const iscoRadius = KerrPhysics.kerrISCO(this.gravityParams.mass, this.gravityParams.spin, true);
        
        // Calculate frame dragging rate at a reference distance
        const referenceDistance = 3.0;
        const frameDraggingRate = KerrPhysics.frameDraggingAngularVelocity(referenceDistance, Math.PI/2, this.gravityParams.mass, this.gravityParams.spin);
        
        document.getElementById('event-horizon-radius').textContent = rs.toFixed(2);
        document.getElementById('ergosphere-radius').textContent = r_ergosphere.toFixed(2);
        document.getElementById('photon-sphere-radius').textContent = photonSphereRadius.toFixed(2);
        document.getElementById('isco-radius').textContent = iscoRadius.toFixed(2);
        document.getElementById('frame-dragging-rate').textContent = frameDraggingRate.toFixed(3);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());

        // Update orbital camera position
        this.updateCameraPosition();

        // Update photon physics
        this.updatePhotons();
        
        // Update gravitational lensing effects
        if (this.backgroundStarfield) {
            this.backgroundStarfield.updateBlackHolePosition(this.gravitationalObject.position);
        }
        
        // Update accretion disk
        if (this.accretionDiskRenderer) {
            this.accretionDiskRenderer.updateRotation();
        }
        
        // Update spacetime grid
        if (this.spacetimeGrid) {
            this.spacetimeGrid.update();
        }
        
        // Update UI
        this.updateUI();

        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize simulation when page loads
window.addEventListener('load', () => {
    new GravitationalSphereSimulation();
});
