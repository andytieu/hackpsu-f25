# Transforming Photons into Black Hole Paths - Detailed Guide

## Current Problem

**What's happening now:**
- Photons orbit in circular patterns around the black hole
- They follow predictable, stable orbits
- No photons are being captured/plunged into the event horizon
- No realistic accretion disk effects
- Paths look like a planetary system, not black hole lensing

**Why this happens:**
- Photons start in circular orbital patterns (like planets)
- Initial velocities are tangential and stable
- Too far from event horizon to show relativistic effects
- No impact parameter range (all photons have similar paths)

---

## What Black Hole Photon Paths Should Look Like

### 1. **Three Types of Photon Trajectories**

#### Type A: Plunging Photons (Inner Region)
- Start inside the photon sphere (r < ~3M for non-spinning)
- Path leads directly into the event horizon
- Visible only briefly as they fall in
- Show extreme red-shifting and time dilation

#### Type B: Orbiting Photons (Photon Sphere)
- Photons that orbit at the photon sphere (r ≈ 3M)
- Can make multiple complete orbits before escaping
- Create the characteristic "photon ring" effect
- Most dramatic example of gravitational capture

#### Type C: Deflected/Escaped Photons (Outer Region)
- Start far from black hole (r > 10M)
- Path curves around the black hole (lensing)
- Some escape to infinity, some are captured
- Show gravitational deflection similar to Einstein rings

### 2. **Key Physical Features**

#### **Gravitational Lensing**
- Light paths bend around the black hole
- Creates an "Einstein ring" effect
- Background objects appear distorted and magnified
- Multiple images of the same background object

#### **Photon Capture**
- Photons approaching with impact parameter < critical value fall in
- Photons with larger impact parameters are deflected
- Critical impact parameter: ~√27 M (for Schwarzschild)
- For Kerr: depends on spin and observation angle

#### **Frame Dragging (Kerr)**
- Rotating black holes drag spacetime around them
- Photons orbit faster on prograde paths
- Creates asymmetry in photon paths
- Visible as elongated orbital patterns

---

## Detailed Implementation Steps

### **Step 1: Radically Redesign Initial Positions**

#### Current Problem:
```javascript
// Lines 208-210 in simulation.js
const angle = (i / maxPhotons) * Math.PI * 2;  // Circular ring
const radius = 2.25 + Math.random() * 3;       // Too close, circular pattern
const height = (Math.random() - 0.5) * 1.5;     // Minimal vertical variation
```

#### What to Do Instead:

**1.1: Implement Impact Parameter Distribution**
```javascript
// Replace circular rings with impact parameter grid
// Impact parameter: b = L/E (angular momentum per energy)
// Photons should have distribution in impact parameter, not radius

const M = this.gravityParams.mass;
const critical_b = Math.sqrt(27) * M;  // Critical impact parameter
const b_min = 0.5 * M;   // Very close - plunging
const b_max = 8.0 * M;   // Distant - escaping

// Create grid of impact parameters
const n_grid = Math.ceil(Math.sqrt(this.gravityParams.maxPhotons));
for (let i = 0; i < this.gravityParams.maxPhotons; i++) {
    const b = b_min + (b_max - b_min) * (i / this.gravityParams.maxPhotons);
    
    // Position photon at distance where impact parameter matters
    const r_start = 10 * M;  // Start far away
    // Angular momentum determines impact parameter
    const Lz = b * 1.0;  // Energy = 1 for null geodesics
}
```

**1.2: Use "Observer at Infinity" Approach**
```javascript
// Instead of orbiting photons, simulate rays from observer's eye
// Like the Python code does with image_plane_to_initial_state

// Observer position (camera/far away)
const r_obs = 20.0 * M;  // Observer is at r = 20M
const theta_obs = Math.PI / 2;  // Equatorial plane
const phi_obs = 0.0;

// Image plane coordinates (like camera pixel)
const alpha = (i % n_grid - n_grid/2) * 0.5;  // Horizontal angle
const beta = (Math.floor(i / n_grid) - n_grid/2) * 0.5;  // Vertical angle

// Calculate initial photon direction from image plane
const k_r = -1.0;  // Inward initial direction
const k_theta = beta / r_obs;
const k_phi = alpha / r_obs;
```

**1.3: Stratified Sampling for Photon Coverage**
```javascript
// Use stratified grid (not random) for uniform coverage
const img_extent = 4.0;  // Angular spread from center
const n_side = Math.ceil(Math.sqrt(this.gravityParams.maxPhotons));

for (let j = 0; j < n_side; j++) {
    for (let i = 0; i < n_side; i++) {
        const alpha = -img_extent + (2 * img_extent * i / (n_side - 1));
        const beta = -img_extent + (2 * img_extent * j / (n_side - 1));
        
        // Create photon with these coordinates
        // This matches ex.py's simulate_photons() approach
    }
}
```

---

### **Step 2: Implement Proper Initial Velocity Setup**

#### Current Problem:
```javascript
// Lines 225-236
const tangentDirection = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle));
const tangentialVelocity = tangentDirection.multiplyScalar(lightSpeed);
const initialVelocity = tangentialVelocity.add(radialComponent).add(verticalComponent);
```
This creates circular orbits, not geodesics toward a black hole.

#### What to Do Instead:

**2.1: Calculate from Impact Parameter**
```javascript
// Impact parameter determines initial velocity
const M = this.gravityParams.mass;
const a = this.gravityParams.spin;
const b = impactParameter;

// For photon geodesics in Kerr spacetime
// Energy and angular momentum are conserved

// Initial direction based on impact parameter
const directionVector = position.clone().normalize();
const perpendicular = new THREE.Vector3(0, 1, 0).cross(directionVector);
const k_perp = perpendicular.normalize().multiplyScalar(b / position.length());

// Initial 4-velocity components
// k^t = 1 (normalized)
// k^r = -1 (inward)
// k^theta = k_perp.y
// k^phi = k_perp.x / (r * sin(theta))
```

**2.2: Ensure Null Geodesic Condition**
```javascript
// For photons, g_{\mu\nu} k^\mu k^\nu = 0 (null geodesic)
// This must be satisfied

// Calculate metric at photon position
const metric = kerrMetric(r, theta, M, a);

// Ensure initial 4-velocity is null
const nullCondition = metric[0][0] * kt**2 + 
                      2 * metric[0][3] * kt * kphi +
                      metric[3][3] * kphi**2 +
                      metric[1][1] * kr**2 +
                      metric[2][2] * ktheta**2;
// Must be = 0 for photon
```

---

### **Step 3: Implement Realistic Distance Distribution**

#### Current Problem:
Photons start at 2.25M - 5.25M (too close, all captured)

#### What to Do Instead:

**3.1: Three Distinct Regions**
```javascript
// Region 1: Plunging photons (r < 3M) - 20%
// Region 2: Photon sphere (r ≈ 3M) - 30%
// Region 3: Distant lensing (r > 10M) - 50%

const n_plunging = Math.floor(0.2 * this.gravityParams.maxPhotons);
const n_photon_sphere = Math.floor(0.3 * this.gravityParams.maxPhotons);
const n_distant = this.gravityParams.maxPhotons - n_plunging - n_photon_sphere;

// Plunging: Start close, will fall in
for (let i = 0; i < n_plunging; i++) {
    const r = 2.0 * M + Math.random() * 1.0 * M;  // Just outside event horizon
    const angle = (i / n_plunging) * Math.PI * 2;
    // Small radial velocity inward ensures plunging
}

// Photon sphere: r = 3M exactly (for Schwarzschild)
for (let i = 0; i < n_photon_sphere; i++) {
    const r = 3.0 * M;  // Exactly at photon sphere
    const angle = (i / n_photon_sphere) * Math.PI * 2;
    // Tangential velocity with specific value
    const v_perp = Math.sqrt(M / r);  // Orbital velocity at photon sphere
}

// Distant: Lens behind black hole
for (let i = 0; i < n_distant; i++) {
    const r = 10 * M + Math.random() * 10 * M;
    const angle = (i / n_distant) * Math.PI * 2;
    // Nearly straight paths that get bent by black hole
}
```

**3.2: Convert to Boyer-Lindquist Coordinates Properly**
```javascript
// Current conversion is wrong:
// theta = Math.acos(position.y / r)  // WRONG

// Correct conversion:
// x = r sin(θ) cos(φ)
// y = r cos(θ)       <- NOTE: cos, not sin!
// z = r sin(θ) sin(φ)

// So:
const r = position.length();
const theta = Math.acos(position.y / r);  // This is actually correct!
const phi = Math.atan2(position.z, position.x);
```

---

### **Step 4: Add Photon Classification and Handling**

#### Need to Track:

**4.1: Photon Outcomes**
```javascript
photon.userData = {
    category: 'unknown',  // 'plunging', 'orbiting', 'escaping', 'captured'
    closestApproach: r,
    maxRadius: r,
    nOrbits: 0,
    captured: false
};
```

**4.2: Detect Photon Capture**
```javascript
// In updatePhotons(), before geodesic integration:
const eventHorizonRadius = KerrPhysics.kerrEventHorizon(mass, spin);
const photonSphereRadius = KerrPhysics.kerrPhotonSphere(mass, spin);

if (r < eventHorizonRadius * 1.01) {
    photon.userData.captured = true;
    photon.userData.category = 'plunging';
    // Make photon disappear or fade out
    photon.visible = false;
    return;
}

if (Math.abs(r - photonSphereRadius) < 0.1 * M) {
    photon.userData.category = 'orbiting';
    photon.userData.nOrbits++;
}
```

**4.3: Handle Escape Detection**
```javascript
// Photons that go far away should stop updating or fade
const maxDistance = 20 * M;
if (photon.position.length() > maxDistance) {
    photon.userData.category = 'escaping';
    // Stop updating this photon or mark it as done
}
```

---

### **Step 5: Correct Geodesic Integration**

#### Current Implementation Issues:

**5.1: Time Step is Too Large**
```javascript
// Current: adaptiveDt = Math.max(0.001, Math.min(0.016, mass * 0.016));
// This is WAY too large for photon geodesics near black hole

// Should be:
const adaptiveDt = Math.min(0.001, mass * 0.0001);  // Much smaller

// OR use geodesic integrator's built-in adaptive step size
const traj = geodesic.integrateRK45(f, y0, 0.0, dt, dt, 1e-8, dt);
// The 5th parameter (dt) is step size - make it adaptive based on position
```

**5.2: Add Stability Checks**
```javascript
// After geodesic integration, verify null geodesic condition
const metricAtNewPos = kerrMetric(newR, newTheta, M, a);
const k4Vector = [lastState[4], lastState[5], lastState[6], lastState[7]];

// Check: g_{\mu\nu} k^\mu k^\nu = 0
const nullCondition = calculateNullCondition(metricAtNewPos, k4Vector);
if (Math.abs(nullCondition) > 0.01) {
    console.warn('Photon geodesic violated null condition!');
    // Reset or correct
}
```

**5.3: Add Event Horizon Protection**
```javascript
// Prevent geodesic integrator from going inside event horizon
const r_h = M + Math.sqrt(Math.max(0, M*M - a*a));

// In integrateRK45:
while (l < l_max && steps < 500000) {
    const result = this.rk45Step(f, l, y, h);
    
    // Check if inside event horizon
    const r_current = result.y[1];
    if (r_current <= r_h * 1.001) {
        // Mark as captured, stop integration
        traj.push(result.y);
        break;  // Exit loop
    }
    
    // Continue integration...
}
```

---

### **Step 6: Visual Enhancements for Black Hole Physics**

**6.1: Add Photon Sphere Ring**
```javascript
// Visual indicator of photon sphere
const photonSphereRadius = KerrPhysics.kerrPhotonSphere(mass, spin);
const ringGeometry = new THREE.RingGeometry(photonSphereRadius - 0.05, photonSphereRadius + 0.05, 64);
const ringMaterial = new THREE.MeshBasicMaterial({
    color: 0xffff00,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide
});
this.scene.add(new THREE.Mesh(ringGeometry, ringMaterial));
```

**6.2: Fade Out Plunging Photons**
```javascript
// As photon approaches event horizon, fade it out
const r = photon.position.length();
const r_h = eventHorizonRadius;
const fadeStart = r_h * 2;

if (r < fadeStart) {
    const fadeFactor = (r - r_h) / (fadeStart - r_h);
    photon.material.opacity = fadeFactor;
    photon.material.emissiveIntensity = fadeFactor;
}
```

**6.3: Color Code by Outcome**
```javascript
// Plunging photons: Red
// Orbiting: Yellow/White
// Escaping: Green
// Captured: Black (invisible)

const color = photon.userData.category === 'plunging' ? 0xff0000 :
              photon.userData.category === 'orbiting' ? 0xffff00 :
              photon.userData.category === 'escaping' ? 0x00ff00 : 0xffffff;
photon.material.color.setHex(color);
```

---

### **Step 7: Performance Considerations**

**7.1: Reduce Photon Count for Geodesics**
```javascript
// Kerr geodesic integration is expensive
// Reduce photons when using relativistic physics
if (this.gravityParams.useRelativisticPhysics) {
    this.gravityParams.maxPhotons = Math.min(50, this.gravityParams.maxPhotons);
    this.gravityParams.maxPhotons2 = Math.min(50, this.gravityParams.maxPhotons2);
}
```

**7.2: Adaptive Quality**
```javascript
// Use simpler physics for far photons
const r = photon.position.length();
if (r > 10 * M) {
    // Far photons: Use simpler Schwarzschild metric
    // Use Kerr for close photons only
}
```

**7.3: Cull Inactive Photons**
```javascript
// Remove captured/escaped photons from update loop
this.photons = this.photons.filter(photon => {
    return !photon.userData.captured && 
           !photon.userData.escaped &&
           photon.visible;
});
```

---

## Summary of Changes Needed

### **Critical Changes:**
1. **Initial positions:** Use impact parameter distribution, not circular rings
2. **Initial velocities:** Calculate from impact parameter and null geodesic condition
3. **Distance distribution:** Mix close (plunging), medium (orbiting), and far (lensing) photons
4. **Outcome tracking:** Classify photons as plunging/orbiting/escaping
5. **Smaller time steps:** Use much smaller dt for geodesic integration
6. **Null geodesic enforcement:** Verify photon 4-velocity satisfies null condition
7. **Event horizon handling:** Properly detect and handle captured photons

### **Implementation Order:**
1. Fix initial position distribution (Step 1)
2. Fix initial velocity calculation (Step 2)
3. Add photon classification (Step 4)
4. Fix geodesic integration parameters (Step 5)
5. Add visual enhancements (Step 6)

This will transform the simulation from "planets orbiting a sphere" to "photons around a black hole".

