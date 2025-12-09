import React, { useState, useMemo, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  PerspectiveCamera, 
} from '@react-three/drei';
import { EffectComposer, Bloom, Vignette, Noise } from '@react-three/postprocessing';
import * as THREE from 'three';
import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';

// --- MATH & UTILS ---

const TREE_HEIGHT = 12;
const TREE_RADIUS = 4.5;
const SCATTER_RADIUS = 25;

// Helper: Random point in sphere
const randomInSphere = (radius: number) => {
  const u = Math.random();
  const v = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  const r = Math.cbrt(Math.random()) * radius;
  const sinPhi = Math.sin(phi);
  return new THREE.Vector3(
    r * sinPhi * Math.cos(theta),
    r * sinPhi * Math.sin(theta),
    r * Math.cos(phi)
  );
};

// Helper: Point in Cone (Tree volume)
const pointInCone = (h: number, r: number) => {
  const y = Math.random() * h;
  const rAtY = (r * (h - y)) / h;
  const angle = Math.random() * Math.PI * 2;
  const rad = Math.sqrt(Math.random()) * rAtY; 
  return new THREE.Vector3(
    rad * Math.cos(angle),
    y - h / 2, 
    rad * Math.sin(angle)
  );
};

// Helper: Point on Cone Surface (for ornaments)
const pointOnConeSurface = (h: number, r: number, t: number) => {
  const y = t * h - h / 2;
  const rAtY = (r * (1 - t));
  const angle = Math.random() * Math.PI * 2;
  return new THREE.Vector3(
    rAtY * Math.cos(angle),
    y,
    rAtY * Math.sin(angle)
  );
};

// Helper: Spiral path for Pearls
const getSpiralPos = (i: number, count: number, h: number, r: number) => {
  const t = i / count;
  const y = t * h - h / 2;
  const rAtY = (r * (1 - t)) + 0.2; 
  const loops = 8;
  const angle = t * Math.PI * 2 * loops;
  return new THREE.Vector3(
    rAtY * Math.cos(angle),
    y,
    rAtY * Math.sin(angle)
  );
};

// --- SHADERS ---

const FoliageMaterial = {
  uniforms: {
    uTime: { value: 0 },
    uProgress: { value: 0 },
    uColor: { value: new THREE.Color('#0B3B24') },
    uHighlight: { value: new THREE.Color('#4F7A5E') },
    uGold: { value: new THREE.Color('#FFD700') }
  },
  vertexShader: `
    uniform float uTime;
    uniform float uProgress;
    attribute vec3 aScatterPos;
    attribute vec3 aTreePos;
    attribute float aRandom;
    
    varying float vAlpha;
    varying vec3 vColor;
    
    float easeOutCubic(float x) {
      return 1.0 - pow(1.0 - x, 3.0);
    }

    void main() {
      float t = easeOutCubic(uProgress);
      vec3 pos = mix(aScatterPos, aTreePos, t);
      
      float wind = sin(uTime * 2.0 + pos.y * 0.5 + pos.x) * 0.05 * t;
      pos.x += wind;
      pos.z += wind;
      
      if (t < 0.5) {
         pos.y += sin(uTime + aRandom * 10.0) * 0.1;
      }

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      gl_PointSize = (8.0 * aRandom + 3.0) * (10.0 / -mvPosition.z);
      vAlpha = 0.6 + 0.4 * sin(uTime * 3.0 + aRandom * 100.0);
    }
  `,
  fragmentShader: `
    uniform vec3 uColor;
    uniform vec3 uHighlight;
    uniform vec3 uGold;
    varying float vAlpha;
    varying vec3 vColor;

    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;
      
      float strength = 1.0 - (dist * 2.0);
      strength = pow(strength, 1.5);
      
      vec3 finalColor = mix(uColor, uGold, strength * 0.5);
      gl_FragColor = vec4(finalColor, vAlpha * strength);
    }
  `
};

// --- COMPONENTS ---

// Shared mutable object type
type SyncData = { 
  value: number;
  handX: number; // 0 to 1, 0.5 is center
  handY: number; // 0 to 1, 0.5 is center
  hasHand: boolean;
};

// Gesture Controller Component (Logic only)
const GestureController = ({ 
  setTreeState, 
  syncData 
}: { 
  setTreeState: (isTree: boolean) => void, 
  syncData: SyncData 
}) => {
  useEffect(() => {
    let recognizer: GestureRecognizer;
    let video: HTMLVideoElement;
    let lastVideoTime = -1;
    let frameId: number;

    const setup = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });

        video = document.createElement("video");
        video.style.display = "none";
        document.body.appendChild(video);

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await new Promise((resolve) => {
           video.onloadedmetadata = () => {
             video.play();
             resolve(true);
           }
        });

        const loop = () => {
          if (video && video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;
            const result = recognizer.recognizeForVideo(video, Date.now());

            if (result.gestures.length > 0 && result.landmarks.length > 0) {
              const gesture = result.gestures[0][0];
              const landmarks = result.landmarks[0];
              
              // Calculate Hand Center (Palm center approx)
              // Landmark 0 is wrist, 9 is middle finger mcp.
              const handX = 1.0 - landmarks[9].x; // Mirror X
              const handY = landmarks[9].y;

              syncData.hasHand = true;
              syncData.handX = THREE.MathUtils.lerp(syncData.handX, handX, 0.1);
              syncData.handY = THREE.MathUtils.lerp(syncData.handY, handY, 0.1);

              // Gesture Logic
              // "Open_Palm" -> Unleash (Scatter) -> isTree = false
              // "Closed_Fist" -> Tree -> isTree = true
              if (gesture.categoryName === "Open_Palm") {
                setTreeState(false);
              } else if (gesture.categoryName === "Closed_Fist") {
                setTreeState(true);
              }
            } else {
              syncData.hasHand = false;
            }
          }
          frameId = requestAnimationFrame(loop);
        };
        loop();

      } catch (e) {
        console.error("Camera/MediaPipe Error:", e);
      }
    };

    setup();

    return () => {
      cancelAnimationFrame(frameId);
      if (video && video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        video.remove();
      }
    };
  }, [setTreeState, syncData]);

  return null;
};

const Foliage = ({ syncData }: { syncData: SyncData }) => {
  const count = 12000;
  const meshRef = useRef<THREE.Points>(null);
  
  const { positions, scatterPositions, randoms } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const scatter = new Float32Array(count * 3);
    const rand = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      const treeP = pointInCone(TREE_HEIGHT, TREE_RADIUS);
      pos[i * 3] = treeP.x;
      pos[i * 3 + 1] = treeP.y;
      pos[i * 3 + 2] = treeP.z;
      
      const scatterP = randomInSphere(SCATTER_RADIUS);
      scatter[i * 3] = scatterP.x;
      scatter[i * 3 + 1] = scatterP.y;
      scatter[i * 3 + 2] = scatterP.z;
      
      rand[i] = Math.random();
    }
    return { positions: pos, scatterPositions: scatter, randoms: rand };
  }, []);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.uTime.value = state.clock.elapsedTime;
      // Read directly from mutable object, no react re-render needed
      material.uniforms.uProgress.value = syncData.value;
    }
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-aTreePos" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-aScatterPos" count={count} array={scatterPositions} itemSize={3} />
        <bufferAttribute attach="attributes-aRandom" count={count} array={randoms} itemSize={1} />
      </bufferGeometry>
      <shaderMaterial
        attach="material"
        args={[FoliageMaterial]}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

const MorphingInstances = ({ 
  count, 
  geometry, 
  material, 
  getTreePos, 
  syncData, 
  scale = 1
}: any) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  const data = useMemo(() => {
    return new Array(count).fill(0).map((_, i) => {
      const treePos = getTreePos(i);
      
      // Height-based Scaling
      const normalizedHeight = (treePos.y + TREE_HEIGHT / 2) / TREE_HEIGHT;
      const heightScaleFactor = 1.0 - normalizedHeight * 0.5;

      const scatterPos = randomInSphere(SCATTER_RADIUS * 0.8);
      const rot = new THREE.Euler(Math.random()*Math.PI, Math.random()*Math.PI, 0);
      
      return { 
        treePos, 
        scatterPos, 
        rot, 
        scale: scale * heightScaleFactor * (0.8 + Math.random() * 0.4) 
      };
    });
  }, [count, scale, getTreePos]);

  useFrame((state) => {
    if (!meshRef.current) return;
    
    const time = state.clock.elapsedTime;
    const progress = syncData.value; // Read mutable value
    const easeProgress = 1.0 - Math.pow(1.0 - progress, 3.0); // Cubic ease out

    data.forEach((item, i) => {
      const targetPos = progress > 0.5 ? item.treePos : item.scatterPos;
      
      const x = THREE.MathUtils.lerp(item.scatterPos.x, item.treePos.x, easeProgress);
      const y = THREE.MathUtils.lerp(item.scatterPos.y, item.treePos.y, easeProgress);
      const z = THREE.MathUtils.lerp(item.scatterPos.z, item.treePos.z, easeProgress);
      
      dummy.position.set(x, y, z);
      
      // Float effect
      dummy.position.y += Math.sin(time + i * 10) * 0.1;

      dummy.rotation.copy(item.rot);
      dummy.rotation.x += time * 0.2;
      dummy.rotation.y += time * 0.1;
      
      dummy.scale.setScalar(item.scale);
      
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[geometry, material, count]} castShadow receiveShadow>
      {/* Instances handled manually */}
    </instancedMesh>
  );
};

const Ornaments = ({ syncData }: { syncData: SyncData }) => {
  const sphereGeo = useMemo(() => new THREE.SphereGeometry(1, 16, 16), []);
  const boxGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);
  const starGeo = useMemo(() => {
    const pts = [];
    for (let i = 0; i < 10; i++) {
        const dist = i % 2 === 0 ? 1 : 0.5;
        const ang = (i / 10) * Math.PI * 2;
        pts.push(new THREE.Vector2(Math.cos(ang) * dist, Math.sin(ang) * dist));
    }
    const shape = new THREE.Shape(pts);
    const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.2, bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.05, bevelSegments: 1 });
    geo.center();
    return geo;
  }, []);

  const goldMaterial = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#FFD700", 
    metalness: 1, 
    roughness: 0.05, 
    emissive: "#C5A059",
    emissiveIntensity: 0.8
  }), []);
  
  const starMaterial = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#FFD700", 
    metalness: 1, 
    roughness: 0.1,
    emissive: "#FFD700",
    emissiveIntensity: 3.0
  }), []);
  
  const pearlMaterial = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#F5F5F0", 
    metalness: 0.1, 
    roughness: 0.1 
  }), []);

  const redGemMaterial = useMemo(() => new THREE.MeshPhysicalMaterial({ 
    color: "#8A0B28", 
    metalness: 0.1, 
    roughness: 0.0,
    transmission: 0.6,
    thickness: 1
  }), []);

  const giftGreenMat = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#0F3B25", 
    metalness: 0.2, 
    roughness: 0.5 
  }), []);
  
  const giftBrownMat = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#C4A484", 
    metalness: 0.1, 
    roughness: 0.6 
  }), []);
  
  const giftPearlMat = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: "#FFF5EE", 
    metalness: 0.3, 
    roughness: 0.2 
  }), []);

  const getGiftPos = () => {
    const r = 2.5 + Math.random() * 4.0;
    const a = Math.random() * Math.PI * 2;
    const yOffset = Math.random() * 1.5;
    return new THREE.Vector3(r * Math.cos(a), -TREE_HEIGHT/2 + 0.6 + yOffset, r * Math.sin(a));
  };
  
  const getSurfaceDistributedPos = (rScale = 1.0) => {
    const u = Math.random();
    const t = 1 - Math.sqrt(u); 
    return pointOnConeSurface(TREE_HEIGHT, TREE_RADIUS * rScale, t);
  };

  return (
    <group>
      <MorphingInstances 
        count={300}
        geometry={sphereGeo}
        material={pearlMaterial}
        scale={0.12}
        syncData={syncData}
        getTreePos={(i: number) => getSpiralPos(i, 300, TREE_HEIGHT, TREE_RADIUS)}
      />
      <MorphingInstances
        count={160}
        geometry={sphereGeo}
        material={goldMaterial}
        scale={0.35} 
        syncData={syncData}
        getTreePos={() => getSurfaceDistributedPos(0.9)}
      />
      <MorphingInstances
        count={50}
        geometry={sphereGeo}
        material={redGemMaterial}
        scale={0.18}
        syncData={syncData}
        getTreePos={() => getSurfaceDistributedPos(0.85)}
      />
      <MorphingInstances
        count={25}
        geometry={boxGeo}
        material={giftGreenMat}
        scale={1.0}
        syncData={syncData}
        getTreePos={getGiftPos}
      />
      <MorphingInstances
        count={25}
        geometry={boxGeo}
        material={giftBrownMat}
        scale={0.9}
        syncData={syncData}
        getTreePos={getGiftPos}
      />
      <MorphingInstances
        count={25}
        geometry={boxGeo}
        material={giftPearlMat}
        scale={1.1}
        syncData={syncData}
        getTreePos={getGiftPos}
      />
      <TopStar syncData={syncData} geometry={starGeo} material={starMaterial} />
    </group>
  );
};

const TopStar = ({ syncData, geometry, material }: any) => {
  const mesh = useRef<THREE.Mesh>(null);
  const startPos = useMemo(() => randomInSphere(SCATTER_RADIUS), []);
  const endPos = useMemo(() => new THREE.Vector3(0, TREE_HEIGHT/2 + 0.5, 0), []);

  useFrame((state) => {
    if(!mesh.current) return;
    const time = state.clock.elapsedTime;
    const progress = syncData.value;
    const easeProgress = 1.0 - Math.pow(1.0 - progress, 3.0);
    
    mesh.current.position.lerpVectors(startPos, endPos, easeProgress);
    mesh.current.rotation.y = time * 0.5;
    mesh.current.rotation.z = Math.sin(time) * 0.1;
    const s = 1.0 + Math.sin(time * 2) * 0.1;
    mesh.current.scale.setScalar(s);
  });

  return <mesh ref={mesh} geometry={geometry} material={material} castShadow />;
};

const Overlay = ({ state, toggleState }: { state: boolean, toggleState: () => void }) => {
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      zIndex: 10
    }}>
      <div style={{
        padding: '40px',
        width: '100%',
        display: 'flex',
        justifyContent: window.innerWidth < 768 ? 'flex-start' : 'center',
      }}>
        <h1 style={{
          margin: 0,
          fontSize: 'clamp(2rem, 5vw, 4rem)',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          textShadow: '0 0 20px rgba(255, 215, 0, 0.5)',
          background: 'linear-gradient(to bottom, #FFD700, #C5A059)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          Merry Christmas
        </h1>
      </div>

      {/* Button Container */}
      <div style={{
        padding: '40px',
        display: 'flex',
        justifyContent: 'center',
        pointerEvents: 'auto',
        marginBottom: '40px' // Make space for footer
      }}>
        <button 
          onClick={toggleState}
          style={{
            background: 'transparent',
            border: '1px solid #FFD700',
            color: '#FFD700',
            padding: '15px 40px',
            fontSize: '1rem',
            fontFamily: 'Cinzel, serif',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            backdropFilter: 'blur(5px)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#FFD700';
            e.currentTarget.style.color = '#020504';
            e.currentTarget.style.boxShadow = '0 0 30px rgba(255, 215, 0, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = '#FFD700';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          {state ? 'Scatter' : 'Reform Tree'}
        </button>
      </div>

      {/* Footer Text */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        width: '100%',
        textAlign: 'center',
        color: '#E8E8E8', // Silver White
        fontSize: '0.9rem',
        fontWeight: 400, // Regular, but visually thin in this font
        opacity: 0.8,
        letterSpacing: '1px',
        textShadow: '0 0 5px rgba(232, 232, 232, 0.3)'
      }}>
        @cider, 2025
      </div>
    </div>
  );
};

const Background = () => {
  return (
    <mesh scale={100}>
      <sphereGeometry />
      <meshBasicMaterial color="#020504" side={THREE.BackSide} />
    </mesh>
  );
};

const Scene = ({ isTree, syncData }: { isTree: boolean, syncData: SyncData }) => {
  const controlsRef = useRef<any>(null);

  useFrame((state, delta) => {
    // 1. Animation Progress Logic
    const target = isTree ? 1 : 0;
    syncData.value = THREE.MathUtils.damp(syncData.value, target, 1.5, delta);

    // 2. Hand Gesture Camera Control Logic
    if (controlsRef.current) {
      if (syncData.hasHand) {
        // Disable auto rotate if user is controlling
        controlsRef.current.autoRotate = false;

        // Map Hand X (0..1) to Azimuthal Angle
        // 0 -> -45deg, 1 -> +45deg offset from center? 
        // Better: Map to full rotation. Center (0.5) is front.
        // Range: -Math.PI/2 to Math.PI/2
        const targetAzimuth = (syncData.handX - 0.5) * Math.PI; // +/- 90 degrees
        
        // Map Hand Y (0..1) to Polar Angle
        // Top (0) -> Higher view (smaller polar), Bottom (1) -> Lower view (larger polar)
        // Range: PI/4 to PI/1.5
        const targetPolar = Math.PI/4 + syncData.handY * (Math.PI/1.5 - Math.PI/4);

        // Smoothly interpolate current angles to target
        // Note: We access internal spherical coordinates or use api
        // Drei controls don't expose simple 'set' for smooth lerp easily without ref access
        // We will just gently nudge the camera if possible, or update the controls target
        
        // Direct manipulation of OrbitControls properties
        const currentAzimuth = controlsRef.current.getAzimuthalAngle();
        const currentPolar = controlsRef.current.getPolarAngle();
        
        const nextAzimuth = THREE.MathUtils.lerp(currentAzimuth, targetAzimuth, 0.05);
        const nextPolar = THREE.MathUtils.lerp(currentPolar, targetPolar, 0.05);
        
        controlsRef.current.setAzimuthalAngle(nextAzimuth);
        controlsRef.current.setPolarAngle(nextPolar);
        
      } else {
        // Resume auto rotate if in Tree mode and no hand
        controlsRef.current.autoRotate = isTree;
      }
      controlsRef.current.update();
    }
  });

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 20]} fov={45} />
      <OrbitControls 
        ref={controlsRef}
        enablePan={false} 
        minPolarAngle={Math.PI / 4} 
        maxPolarAngle={Math.PI / 1.5}
        minDistance={10}
        maxDistance={40}
        autoRotate={isTree}
        autoRotateSpeed={0.5}
      />
      
      {/* Enhanced Lighting for Brightness and Gloss */}
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 10, 5]} intensity={4} color="#FFD700" castShadow />
      <spotLight position={[-10, 20, -5]} intensity={5} color="#4fb" angle={0.5} penumbra={1} />
      <spotLight position={[0, 10, -20]} intensity={8} color="#FFD700" distance={50} />

      <group position={[0, -2, 0]}>
         <Foliage syncData={syncData} />
         <Ornaments syncData={syncData} />
      </group>

      <Background />

      <EffectComposer disableNormalPass>
        <Bloom luminanceThreshold={0.5} mipmapBlur intensity={1.5} radius={0.6} />
        <Noise opacity={0.05} />
        <Vignette eskil={false} offset={0.1} darkness={1.1} />
      </EffectComposer>
    </>
  );
};

function App() {
  const [isTree, setIsTree] = useState(false);
  
  // Mutable sync object for performance (animation & hand tracking)
  const syncData = useMemo<SyncData>(() => ({ 
    value: 0, 
    handX: 0.5, 
    handY: 0.5, 
    hasHand: false 
  }), []);

  return (
    <>
      <GestureController setTreeState={setIsTree} syncData={syncData} />
      <Canvas 
        shadows 
        dpr={[1, 2]} 
        gl={{ antialias: false, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 2.5 }}
      >
        <Scene isTree={isTree} syncData={syncData} />
      </Canvas>
      <Overlay state={isTree} toggleState={() => setIsTree(!isTree)} />
    </>
  );
}

// MOUNTING LOGIC
const rootElement = document.getElementById('root');
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
