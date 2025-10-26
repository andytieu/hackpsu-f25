/**
 * Simple Red Sphere Simulation
 * Basic Three.js scene with a red spherical object
 */

class SimpleRedSphere {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupCamera();
        this.createRedSphere();
        this.setupLighting();
        this.setupControls();
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
        this.camera.position.set(0, 2, 5);
        this.camera.lookAt(0, 0, 0);
    }
    
    createRedSphere() {
        // Create red sphere geometry
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        
        // Create red material
        const sphereMaterial = new THREE.MeshLambertMaterial({
            color: 0xff0000, // Bright red
            transparent: false,
            opacity: 1.0
        });
        
        // Create sphere mesh
        this.redSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.redSphere.castShadow = true;
        this.redSphere.receiveShadow = true;
        
        // Add to scene
        this.scene.add(this.redSphere);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        this.scene.add(directionalLight);
        
        // Point light
        const pointLight = new THREE.PointLight(0xffffff, 0.5, 10);
        pointLight.position.set(-5, 5, 5);
        this.scene.add(pointLight);
    }
    
    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 20;
        this.controls.maxPolarAngle = Math.PI;
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Rotate the red sphere slowly
        if (this.redSphere) {
            this.redSphere.rotation.y += 0.01;
            this.redSphere.rotation.x += 0.005;
        }
        
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
    new SimpleRedSphere();
});
