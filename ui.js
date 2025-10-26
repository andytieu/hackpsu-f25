/**
 * UI management utilities
 */

class UIManager {
    /**
     * Update simulation UI with current parameters
     * @param {Object} params - Simulation parameters
     */
    static updateUI(params) {
        const photonCountElement = document.getElementById('photon-count');
        const gravityStrengthElement = document.getElementById('gravity-strength');
        
        if (photonCountElement) {
            photonCountElement.textContent = params.photonCount || 0;
        }
        
        if (gravityStrengthElement) {
            gravityStrengthElement.textContent = params.gravitationalConstant.toFixed(1);
        }
    }

    /**
     * Setup keyboard controls
     * @param {Object} gravityParams - Gravity parameters object to modify
     */
    static setupKeyboardControls(gravityParams) {
        document.addEventListener('keydown', (e) => {
            switch (e.key) {
                case 'ArrowUp':
                    gravityParams.gravitationalConstant = Math.min(0.5, gravityParams.gravitationalConstant + 0.05);
                    break;
                case 'ArrowDown':
                    gravityParams.gravitationalConstant = Math.max(0.01, gravityParams.gravitationalConstant - 0.05);
                    break;
                case 'ArrowLeft':
                    gravityParams.mass = Math.max(0.1, gravityParams.mass - 0.1);
                    break;
                case 'ArrowRight':
                    gravityParams.mass = Math.min(3.0, gravityParams.mass + 0.1);
                    break;
            }
        });
    }

    /**
     * Setup window resize handler
     * @param {THREE.Camera} camera - Three.js camera
     * @param {THREE.WebGLRenderer} renderer - Three.js renderer
     */
    static setupResizeHandler(camera, renderer) {
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
}
