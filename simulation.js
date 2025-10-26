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
        this.cameraRadius = 15; // Increased initial camera distance for better overview
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
        const M = this.gravityParams.mass;
        const a = this.gravityParams.spin;
        
        // Calculate critical values
        const r_h = KerrPhysics.kerrEventHorizon(M, a);  // Event horizon
        const r_photon = KerrPhysics.kerrPhotonSphere(M, a);  // Photon sphere
        
        // Divide photons into three categories
        const n_total = this.gravityParams.maxPhotons;
        const n_plunging = Math.floor(n_total * 0.2);   // 20% plunging
        const n_photon_sphere = Math.floor(n_total * 0.3);  // 30% at photon sphere
        const n_distant = n_total - n_plunging - n_photon_sphere;  // 50% distant
        
        for (let i = 0; i < n_total; i++) {
            let r, theta, phi, category;
            
            // Assign photon to a category based on position
            if (i < n_plunging) {
                // Region 1: Plunging photons (just outside event horizon)
                category = 'plunging';
                r = r_h * 1.5 + Math.random() * r_h * 0.5;
                theta = Math.PI / 2;  // Equatorial plane
                phi = (i / n_plunging) * Math.PI * 2;
            } else if (i < n_plunging + n_photon_sphere) {
                // Region 2: Photon sphere region
                category = 'orbiting';
                r = r_photon + (Math.random() - 0.5) * 0.3 * M;
                theta = Math.PI / 2 + (Math.random() - 0.5) * 0.3;
                phi = ((i - n_plunging) / n_photon_sphere) * Math.PI * 2;
            } else {
                // Region 3: Distant photons for lensing effect
                category = 'lensing';
                r = 8 * M + Math.random() * 8 * M;
                theta = Math.PI / 2 + (Math.random() - 0.5) * Math.PI / 3;
                phi = ((i - n_plunging - n_photon_sphere) / n_distant) * Math.PI * 2;
            }
            
            // Convert to Cartesian coordinates
            const x = r * Math.sin(theta) * Math.cos(phi);
            const y = r * Math.cos(theta);
            const z = r * Math.sin(theta) * Math.sin(phi);
            
            // Create photon
            const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
            
            // Color based on category
            let color = 0xffff00; // Default yellow
            if (category === 'plunging') color = 0xff0000;  // Red
            else if (category === 'orbiting') color = 0xffff00;  // Yellow
            else if (category === 'lensing') color = 0x00ff00;  // Green
            
            const photonMaterial = new THREE.MeshBasicMaterial({
                color: color,
                emissive: color >> 1,
                transparent: true,
                opacity: 0.9
            });
            const photon = new THREE.Mesh(photonGeometry, photonMaterial);
            
            photon.position.set(x, y, z);
            
            // Calculate initial velocity with INWARD radial component
            const posDirection = photon.position.clone().normalize();
            const tangentDirection = new THREE.Vector3(-Math.sin(phi), 0, Math.cos(phi));
            
            // Strong inward radial component (photons fall toward black hole)
            const radialInward = posDirection.multiplyScalar(-0.8);  // 80% inward
            const tangential = tangentDirection.multiplyScalar(0.6);  // 60% tangential
            const vertical = new THREE.Vector3(0, (Math.random() - 0.5) * 0.2, 0);  // Some vertical
            
            const initialVelocity = radialInward.add(tangential).add(vertical).normalize();
            
            // Store photon data
            photon.userData = {
                category: category,
                velocity: initialVelocity,
                trail: [],
                nOrbits: 0,
                captured: false,
                escaped: false,
                closestApproach: r
            };

            this.photons.push(photon);
            this.scene.add(photon);
            this.createPhotonTrail(photon);
        }
    }
    
    createPhotons2() {
        const M = this.gravityParams.mass;
        const a = this.gravityParams.spin;
        const r_h = KerrPhysics.kerrEventHorizon(M, a);
        const r_photon = KerrPhysics.kerrPhotonSphere(M, a);
        
        const n_total = this.gravityParams.maxPhotons2;
        const n_plunging = Math.floor(n_total * 0.2);
        const n_photon_sphere = Math.floor(n_total * 0.3);
        const n_distant = n_total - n_plunging - n_photon_sphere;
        
        for (let i = 0; i < n_total; i++) {
            let r, theta, phi, category;
            
            if (i < n_plunging) {
                category = 'plunging';
                r = r_h * 1.5 + Math.random() * r_h * 0.5;
                theta = Math.PI / 2;
                phi = (i / n_plunging) * Math.PI * 2 + Math.PI; // Offset by PI for second cluster
            } else if (i < n_plunging + n_photon_sphere) {
                category = 'orbiting';
                r = r_photon + (Math.random() - 0.5) * 0.3 * M;
                theta = Math.PI / 2 + (Math.random() - 0.5) * 0.3;
                phi = ((i - n_plunging) / n_photon_sphere) * Math.PI * 2 + Math.PI;
            } else {
                category = 'lensing';
                r = 8 * M + Math.random() * 8 * M;
                theta = Math.PI / 2 + (Math.random() - 0.5) * Math.PI / 3;
                phi = ((i - n_plunging - n_photon_sphere) / n_distant) * Math.PI * 2 + Math.PI;
            }
            
            const x = r * Math.sin(theta) * Math.cos(phi);
            const y = r * Math.cos(theta);
            const z = r * Math.sin(theta) * Math.sin(phi);
            
            const photonGeometry = new THREE.SphereGeometry(0.05, 8, 6);
            let color = 0x00ffff; // Cyan for second cluster
            if (category === 'plunging') color = 0xff0000;
            else if (category === 'orbiting') color = 0xffff00;
            else if (category === 'lensing') color = 0x00ff00;
            
            const photonMaterial = new THREE.MeshBasicMaterial({
                color: color,
                emissive: color >> 1,
                transparent: true,
                opacity: 0.9
            });
            const photon = new THREE.Mesh(photonGeometry, photonMaterial);
            
            photon.position.set(x, y, z);
            
            const posDirection = photon.position.clone().normalize();
            const tangentDirection = new THREE.Vector3(-Math.sin(phi), 0, Math.cos(phi));
            
            const radialInward = posDirection.multiplyScalar(-0.8);
            const tangential = tangentDirection.multiplyScalar(0.6);
            const vertical = new THREE.Vector3(0, (Math.random() - 0.5) * 0.2, 0);
            
            const initialVelocity = radialInward.add(tangential).add(vertical).normalize();
            
            photon.userData = {
                category: category,
                velocity: initialVelocity,
                trail: [],
                nOrbits: 0,
                captured: false,
                escaped: false,
                closestApproach: r
            };
            
            this.photons2.push(photon);
            this.scene.add(photon);
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
        const M = this.gravityParams.mass;
        const r_h = KerrPhysics.kerrEventHorizon(M, this.gravityParams.spin);
        const r_photon = KerrPhysics.kerrPhotonSphere(M, this.gravityParams.spin);
        
        // Use appropriate time steps for photon geodesics
        // Smaller dt for accuracy, but not so small that photons don't move
        const adaptiveDt = this.gravityParams.mass * 0.01; // Increased time step for better visual clarity
        
        // Update first cluster
        this.photons.forEach(photon => {
            const userData = photon.userData;
            const r = photon.position.length();
            
            // Check if photon is captured by event horizon
            if (r < r_h * 1.01 || userData.captured) {
                userData.captured = true;
                photon.visible = false;
                return;
            }
            
            // Check if escaped
            if (r > 30 * M) {
                userData.escaped = true;
                // For classical physics, reset the photon to keep it in view
                if (!this.gravityParams.useRelativisticPhysics) {
                    this.resetPhoton(photon);
                    return;
                }
            }

            if (this.gravityParams.useRelativisticPhysics) {
                // Use Kerr geodesic integration
                const result = KerrGeodesicIntegrator.integrateKerrGeodesic(
                    photon.position,
                    userData.velocity,
                    this.gravityParams.mass,
                    this.gravityParams.spin,
                    adaptiveDt
                );
                
                if (result) {
                    const oldR = photon.position.length();
                    photon.position.copy(result.position);
                    userData.velocity.copy(result.velocity);
                    
                    const newR = photon.position.length();
                    
                    // Track closest approach
                    if (newR < userData.closestApproach) {
                        userData.closestApproach = newR;
                    }
                    
                    // Detect orbiting behavior (crossing phi = 0 multiple times)
                    const oldPhi = Math.atan2(photon.position.z, photon.position.x);
                    if (Math.abs(oldPhi - Math.atan2(result.position.z, result.position.x)) > Math.PI) {
                        userData.nOrbits++;
                    }
                    
                    // Apply relativistic velocity limit
                    userData.velocity = RelativisticPhysics.limitRelativisticVelocity(userData.velocity);
                    
                    // Fade out photons approaching event horizon
                    const fadeDistance = r_h * 2;
                    if (r < fadeDistance && r > r_h) {
                        const fadeFactor = (r - r_h) / (fadeDistance - r_h);
                        photon.material.opacity = fadeFactor;
                    }
                } else {
                    // Unstable integration
                    userData.captured = true;
                    photon.visible = false;
                    return;
                }
            } else {
                // Classical physics with orbital constraints
                const gravitationalForce = this.calculateGravitationalForce(photon);
                
                // Apply gravity
                userData.velocity.add(gravitationalForce);
                
                // Add orbital velocity to maintain stable orbits
                const pos = photon.position.clone();
                const perpToRadial = new THREE.Vector3(-pos.z, 0, pos.x).normalize();
                const orbitalSpeed = Math.sqrt(this.gravityParams.mass * this.gravityParams.gravitationalConstant / pos.length());
                const orbitalVelocity = perpToRadial.multiplyScalar(orbitalSpeed);
                
                // Blend velocities
                userData.velocity.lerp(orbitalVelocity, 0.5);
                
                // Update position
                photon.position.add(userData.velocity);
                
                // Limit velocity
                if (userData.velocity.length() > 2.0) {
                    userData.velocity.normalize().multiplyScalar(2.0);
                }
            }

            TrailManager.updateTrail(photon, this.photonTrails);
        });

        // Update second cluster (cyan photons)
        this.photons2.forEach(photon => {
            const userData = photon.userData;

            const r2 = photon.position.length();
            
            if (r2 < r_h * 1.01 || userData.captured) {
                userData.captured = true;
                photon.visible = false;
                return;
            }
            
            if (r2 > 30 * M) {
                userData.escaped = true;
                // For classical physics, reset the photon to keep it in view
                if (!this.gravityParams.useRelativisticPhysics) {
                    this.resetPhoton2(photon);
                    return;
                }
            }
            
            // Check if photon is captured by event horizon
            if (KerrPhysics.isInsideEventHorizon(photon.position, this.gravityParams.mass, this.gravityParams.spin)) {
                userData.captured = true;
                photon.visible = false;
                return;
            }

            if (this.gravityParams.useRelativisticPhysics) {
                // Use Kerr geodesic integration with adaptive time step
                const result = KerrGeodesicIntegrator.integrateKerrGeodesic(
                    photon.position,
                    userData.velocity,
                    this.gravityParams.mass,
                    this.gravityParams.spin,
                    adaptiveDt
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
                // Classical physics with orbital constraints
                const gravitationalForce = this.calculateGravitationalForce(photon);
                
                // Apply gravity
                userData.velocity.add(gravitationalForce);
                
                // Add orbital velocity to maintain stable orbits
                const pos = photon.position.clone();
                const perpToRadial = new THREE.Vector3(-pos.z, 0, pos.x).normalize();
                const orbitalSpeed = Math.sqrt(this.gravityParams.mass * this.gravityParams.gravitationalConstant / pos.length());
                const orbitalVelocity = perpToRadial.multiplyScalar(orbitalSpeed);
                
                // Blend velocities
                userData.velocity.lerp(orbitalVelocity, 0.5);
                
                // Update position
                photon.position.add(userData.velocity);
                
                // Limit velocity
                if (userData.velocity.length() > 2.0) {
                    userData.velocity.normalize().multiplyScalar(2.0);
                }
            }

            TrailManager.updateTrail(photon, this.photonTrails2);
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
        this.cameraRadius = Math.max(2, Math.min(100, this.cameraRadius)); // Increased max zoom distance from 20 to 100
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
        
        // Calculate proper initial velocity for Kerr photon orbit
        const tangentDirection = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle));
        const radialComponent = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)).multiplyScalar(Math.random() * 0.01);
        const verticalComponent = new THREE.Vector3(0, Math.random() * 0.1 - 0.05, 0);
        const lightSpeed = 1.0;
        const tangentialVelocity = tangentDirection.multiplyScalar(lightSpeed);
        const initialVelocity = tangentialVelocity.add(radialComponent).add(verticalComponent).normalize().multiplyScalar(lightSpeed);

        photon.userData = {
            angle: angle,
            radius: radius,
            height: height,
            velocity: initialVelocity,
            orbitalSpeed: lightSpeed,
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
        
        // Calculate proper initial velocity for Kerr photon orbit in perpendicular plane
        const tangentDirection = new THREE.Vector3(0, -Math.sin(angle), Math.cos(angle));
        const radialComponent = new THREE.Vector3(Math.random() * 0.01 - 0.005, 
                                                   Math.cos(angle) * Math.random() * 0.01, 
                                                   Math.sin(angle) * Math.random() * 0.01);
        const verticalComponent = new THREE.Vector3(Math.random() * 0.1 - 0.05, 0, 0);
        const lightSpeed = 1.0;
        const tangentialVelocity = tangentDirection.multiplyScalar(lightSpeed);
        const initialVelocity = tangentialVelocity.add(radialComponent).add(verticalComponent).normalize().multiplyScalar(lightSpeed);

        photon.userData = {
            angle: angle,
            radius: radius,
            height: height,
            velocity: initialVelocity,
            orbitalSpeed: lightSpeed,
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
