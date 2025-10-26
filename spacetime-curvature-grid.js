/**
 * Spacetime Curvature Grid Visualization
 * Shows the "trapdoor in spacetime" effect of black holes
 */

class SpacetimeCurvatureGrid {
    constructor(scene, blackHole) {
        this.scene = scene;
        this.blackHole = blackHole;
        this.gridSize = 20;
        this.gridSegments = 20;
        this.gridMesh = null;
        this.intensity = 2.0;
        this.additionalGrids = [];
        
        this.createGrid();
    }
    
    /**
     * Create spacetime curvature grid
     */
    createGrid() {
        const geometry = new THREE.PlaneGeometry(
            this.gridSize,
            this.gridSize,
            this.gridSegments,
            this.gridSegments
        );
        
        // Create shader material for spacetime curvature visualization
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                blackHolePos: { value: new THREE.Vector3(0, 0, 0) },
                intensity: { value: this.intensity }
            },
            vertexShader: `
                uniform float time;
                uniform vec3 blackHolePos;
                uniform float intensity;
                varying vec3 vPosition;
                varying float vDistortion;
                
                void main() {
                    vPosition = position;
                    
                    // Calculate distance to black hole
                    vec2 dist = position.xz - blackHolePos.xz;
                    float r = length(dist);
                    
                    // Calculate curvature effect
                    float mass = 1.0;
                    float curvature = (2.0 * mass) / (r + 0.1);
                    curvature *= intensity;
                    
                    // Apply distortion
                    vec3 newPosition = position;
                    newPosition.y -= curvature * 0.5;
                    
                    // Store distortion for fragment shader
                    vDistortion = curvature;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vPosition;
                varying float vDistortion;
                
                void main() {
                    // Create grid pattern
                    float gridSize = 1.0;
                    vec2 grid = fract(vPosition.xz / gridSize) - 0.5;
                    
                    // Grid lines
                    float lines = abs(grid.x) < 0.02 || abs(grid.y) < 0.02;
                    
                    // Color based on distortion
                    vec3 color = mix(
                        vec3(0.1, 0.1, 0.3),
                        vec3(0.5, 0.7, 1.0),
                        vDistortion
                    );
                    
                    // Add grid lines
                    color = mix(color, vec3(1.0), lines * 0.5);
                    
                    gl_FragColor = vec4(color, 0.8);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            wireframe: false
        });
        
        this.gridMesh = new THREE.Mesh(geometry, material);
        this.gridMesh.rotation.x = -Math.PI / 2;
        this.gridMesh.position.y = -0.5;
        this.scene.add(this.gridMesh);
        
        // Create additional grid layers for depth
        this.createAdditionalGrids();
    }
    
    /**
     * Create additional grid layers
     */
    createAdditionalGrids() {
        // Second layer (slightly higher)
        const geometry2 = new THREE.PlaneGeometry(
            this.gridSize,
            this.gridSize,
            this.gridSegments,
            this.gridSegments
        );
        const material2 = new THREE.MeshBasicMaterial({
            color: 0x4466aa,
            transparent: true,
            opacity: 0.3,
            wireframe: true,
            side: THREE.DoubleSide
        });
        const grid2 = new THREE.Mesh(geometry2, material2);
        grid2.rotation.x = -Math.PI / 2;
        grid2.position.y = 0;
        this.scene.add(grid2);
        this.additionalGrids.push(grid2);
        
        // Third layer (vertical grid for 3D effect)
        const geometry3 = new THREE.PlaneGeometry(
            this.gridSize,
            this.gridSize,
            10,
            10
        );
        const material3 = new THREE.MeshBasicMaterial({
            color: 0x6677bb,
            transparent: true,
            opacity: 0.2,
            wireframe: true,
            side: THREE.DoubleSide
        });
        const grid3 = new THREE.Mesh(geometry3, material3);
        grid3.rotation.z = -Math.PI / 2;
        grid3.position.x = -this.gridSize / 2;
        this.scene.add(grid3);
        this.additionalGrids.push(grid3);
    }
    
    /**
     * Update grid animation
     */
    update() {
        if (this.gridMesh && this.gridMesh.material) {
            this.gridMesh.material.uniforms.time.value += 0.01;
            this.gridMesh.material.uniforms.intensity.value = this.intensity;
            
            // Update black hole position if it has moved
            if (this.blackHole && this.blackHole.position) {
                this.gridMesh.material.uniforms.blackHolePos.value = this.blackHole.position.clone();
            }
        }
    }
    
    /**
     * Set grid intensity
     */
    setIntensity(intensity) {
        this.intensity = intensity;
    }
    
    /**
     * Toggle grid visibility
     */
    setVisible(visible) {
        if (this.gridMesh) this.gridMesh.visible = visible;
        if (this.additionalGrids) {
            this.additionalGrids.forEach(grid => {
                grid.visible = visible;
            });
        }
    }
    
    /**
     * Dispose resources
     */
    dispose() {
        if (this.gridMesh) {
            this.scene.remove(this.gridMesh);
            this.gridMesh.geometry.dispose();
            this.gridMesh.material.dispose();
        }
        
        // Dispose additional grids
        if (this.additionalGrids) {
            this.additionalGrids.forEach(grid => {
                this.scene.remove(grid);
                grid.geometry.dispose();
                grid.material.dispose();
            });
            this.additionalGrids = [];
        }
    }
}
