import * as THREE from 'three';
import { HDRLoader } from 'three/examples/jsm/Addons.js';
import { PMREMGenerator } from 'three/src/extras/PMREMGenerator.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const loader = new HDRLoader();
loader.load('HDR_rich_multi_nebulae_2.hdr', function (texture) {
    const pmremGenerator = new PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();

    const env_map = pmremGenerator.fromEquirectangular(texture).texture;
    scene.background = env_map;
    scene.environment = env_map; // FOR REFLECTIONS

    // FREE MEMORY
    texture.dispose();
    pmremGenerator.dispose();
});

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );


const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

//const controls = new OrbitControls(camera, renderer.domElement);

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );


camera.position.z = 5;

function animate() {
  renderer.render( scene, camera );
  cube.rotation.x += 0.01;
  cube.rotation.y += 0.01;
}
renderer.setAnimationLoop( animate );