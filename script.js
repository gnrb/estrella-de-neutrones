// Cinematic scientific illustration. Surface, emission, nebula and screen lens are artistic approximations; not GRMHD/ray tracing.
/* ==========================================================================
   SIMULACIÓN ASTROFÍSICA REALISTA DE ESTRELLA DE NEUTRONES (PÚLSAR / MAGNETAR)
   - Nebulosa de Viento de Púlsar (PWN) volumétrica suave en GPU (sin puntos duros)
   - Líneas de campo dipolar continuas + Chorros relativistas con Doppler beaming
   - Corteza con geodésica de Beloborodov + Boosting Doppler (delta^4)
   - Corte Interno 3D nítido (sin saturación de Bloom) mostrando las 4 capas TOV
   ========================================================================== */

const BLOOM_MIN_HEIGHT = 720;

/* ==========================================================================
   1. SHADER DE LENTE GRAVITATORIA DE EINSTEIN & LENSE-THIRRING
   ========================================================================== */
const GravitationalLensingShader = {
    uniforms: {
        tDiffuse: { value: null },
        uStarScreenUV: { value: new THREE.Vector2(0.5, 0.5) },
        uAspect: { value: 1.777 },
        uEinsteinRad: { value: 0.040 },
        uFrameDrag: { value: 0.012 },
        uEnabled: { value: 1.0 }
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform vec2 uStarScreenUV;
        uniform float uAspect;
        uniform float uEinsteinRad;
        uniform float uFrameDrag;
        uniform float uEnabled;
        varying vec2 vUv;

        void main() {
            vec2 sampleUv = vUv;
            float ringCaustic = 0.0;

            if (uEnabled > 0.5) {
                vec2 delta = vUv - uStarScreenUV;
                delta.x *= uAspect;
                float r = length(delta);

                float thetaE2 = uEinsteinRad * uEinsteinRad;
                float deflection = thetaE2 / (r * r + 0.0022);

                vec2 ortho = vec2(-delta.y, delta.x) / (r + 0.001);
                vec2 frameDrag = ortho * (uFrameDrag * thetaE2 / (r * r + 0.005));

                vec2 lensedDelta = delta * (1.0 - deflection * 0.52) + frameDrag;
                lensedDelta.x /= uAspect;
                sampleUv = clamp(uStarScreenUV + lensedDelta, 0.001, 0.999);

                float ringDist = abs(r - uEinsteinRad * 1.38);
                ringCaustic = 0.0; // no synthetic Einstein ring
            }

            vec4 color = texture2D(tDiffuse, sampleUv);
            color.rgb += vec3(0.45, 0.78, 1.0) * ringCaustic;

            // Viñeta óptica suave
            vec2 c = vUv - 0.5;
            color.rgb *= clamp(1.0 - dot(c, c) * 0.35, 0.0, 1.0);

            gl_FragColor = color;
        }
    `
};

/* ==========================================================================
   2. NEBULOSA DE VIENTO DE PÚLSAR (PWN) VOLUMÉTRICA EN GPU
   Usa nubes gaussianas suaves para eliminar el aspecto de "puntos duros"
   ========================================================================== */
class VolumetricPulsarNebula {
    constructor(renderer, size = 144) {
        this.renderer = renderer;
        this.texSize = size; // 20,736 nubes suaves superpuestas
        this.count = this.texSize * this.texSize;
        this.useGPGPU = false;

        try {
            const gl = this.renderer.getContext();
            const hasFloat = this.renderer.capabilities.isWebGL2 || gl.getExtension('OES_texture_float');
            const hasHalf = gl.getExtension('OES_texture_half_float');

            if (typeof THREE.GPUComputationRenderer !== 'undefined' && (hasFloat || hasHalf)) {
                this.gpuCompute = new THREE.GPUComputationRenderer(this.texSize, this.texSize, this.renderer);
                if (!hasFloat) this.gpuCompute.setDataType(THREE.HalfFloatType);
                this._initGPGPU();
                this._initVolumetricPoints();
                this.useGPGPU = true;
            }
        } catch (e) {
            console.warn("GPGPU fallback:", e);
        }
    }

    _initGPGPU() {
        const dtPos = this.gpuCompute.createTexture();
        const dtVel = this.gpuCompute.createTexture();

        const posData = dtPos.image.data;
        const velData = dtVel.image.data;

        for (let i = 0; i < this.count; i++) {
            const i4 = i * 4;
            // Distribución continua desde cerca de la magnetósfera (r = 1.6) hasta el borde exterior (r = 11.5)
            // Sin hueco artificial negro en el centro
            const u = Math.random();
            const r = 1.55 + Math.pow(u, 1.35) * 9.8;
            const angle = Math.random() * Math.PI * 2 + r * 0.65;

            // Espesor vertical que crece suavemente con el radio (geometría de toro MHD)
            const thickness = 0.18 + 0.09 * r * Math.sin(u * Math.PI);
            const y = (Math.random() - 0.5) * thickness * 2.0;

            posData[i4] = Math.cos(angle) * r;
            posData[i4 + 1] = y;
            posData[i4 + 2] = Math.sin(angle) * r;
            posData[i4 + 3] = 0.5 + Math.random() * 0.5; // semilla individual

            velData[i4] = 0; velData[i4 + 1] = 0; velData[i4 + 2] = 0; velData[i4 + 3] = 1.0;
        }

        const velShader = `
            uniform float uTime;
            uniform float uAudioWave;
            uniform float uStep;
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec4 pos = texture2D(texturePosition, uv);
                vec4 vel = texture2D(textureVelocity, uv);

                vec3 dir = -pos.xyz;
                float r = length(dir.xz) + 0.001;

                // Equilibrio hidrodinámico suave en r_shock ~ 2.1
                vec3 gravity = normalize(dir) * (1.8 / (r * r + 0.3));
                float innerPush = (1.0 - smoothstep(1.3, 2.4, r)) * 1.9;
                vec3 wind = normalize(-dir) * innerPush;

                // Rotación kepleriana/MHD suave
                vec3 tang = cross(normalize(dir), vec3(0.0, 1.0, 0.0)) * (2.6 / (r * r + 0.7));
                tang *= smoothstep(1.0, 2.2, r);

                float outerContain = max(r - 10.8, 0.0);
                vec3 contain = normalize(dir) * outerContain * 1.5;
                float yRestore = -pos.y * 1.4;

                // Ondas magneto-acústicas (wisps del Cangrejo)
                float wisp = sin(r * 2.8 - uTime * 2.6) * (0.04 + uAudioWave * 0.05);

                vel.xyz += (gravity + wind + tang + contain + vec3(0.0, yRestore + wisp, 0.0)) * 0.01 * uStep;
                vel.xyz *= pow(0.972, uStep);

                gl_FragColor = vec4(vel.xyz, 1.0);
            }
        `;

        const posShader = `
            uniform float uStep;
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec4 pos = texture2D(texturePosition, uv);
                vec4 vel = texture2D(textureVelocity, uv);
                pos.xyz += vel.xyz * 0.016 * uStep;
                gl_FragColor = vec4(pos.xyz, pos.w);
            }
        `;

        this.velVar = this.gpuCompute.addVariable("textureVelocity", velShader, dtVel);
        this.posVar = this.gpuCompute.addVariable("texturePosition", posShader, dtPos);

        this.gpuCompute.setVariableDependencies(this.velVar, [this.posVar, this.velVar]);
        this.gpuCompute.setVariableDependencies(this.posVar, [this.posVar, this.velVar]);

        this.velUniforms = this.velVar.material.uniforms;
        this.velUniforms["uTime"] = { value: 0.0 };
        this.velUniforms["uAudioWave"] = { value: 0.0 };

        this.velUniforms.uStep = {value:0};
        this.posVar.material.uniforms.uStep = {value:0};
        const gpuError = this.gpuCompute.init();
        if (gpuError) throw new Error(gpuError);
    }

    _initVolumetricPoints() {
        const geo = new THREE.BufferGeometry();
        const dummy = new Float32Array(this.count * 3);
        const ref = new Float32Array(this.count * 2);

        for (let i = 0; i < this.count; i++) {
            ref[i * 2] = ((i % this.texSize) + 0.5) / this.texSize;
            ref[i * 2 + 1] = (Math.floor(i / this.texSize) + 0.5) / this.texSize;
        }

        geo.setAttribute('position', new THREE.BufferAttribute(dummy, 3));
        geo.setAttribute('reference', new THREE.BufferAttribute(ref, 2));

        this.material = new THREE.ShaderMaterial({
            uniforms: {
                texturePosition: { value: null },
                uTime: { value: 0 },
                uDopplerEnabled: { value: 1.0 },
                uBetaEq: { value: 0.05 },
                uColorInner: { value: new THREE.Color('#38bdf8') },
                uColorOuter: { value: new THREE.Color('#312e81') }
            },
            vertexShader: `
                uniform sampler2D texturePosition;
                uniform float uTime;
                uniform float uDopplerEnabled;
                uniform float uBetaEq;
                uniform vec3 uColorInner;
                uniform vec3 uColorOuter;
                attribute vec2 reference;
                varying vec3 vColor;
                varying float vAlpha;

                void main() {
                    vec4 posData = texture2D(texturePosition, reference);
                    vec3 pos = posData.xyz;
                    float seed = posData.w;

                    float r = length(pos.xz);
                    float t = clamp((r - 1.5) / 9.2, 0.0, 1.0);

                    // Ondas concéntricas de sincrotrón (wisps reales de rayos X)
                    float wispRing = 0.72 + 0.28 * sin(r * 3.4 - uTime * 2.8 + seed * 6.28);

                    vec3 shockWhite = vec3(0.78, 0.92, 1.0);
                    vec3 baseCol = t < 0.28
                        ? mix(shockWhite, uColorInner, t / 0.28)
                        : mix(uColorInner, uColorOuter, (t - 0.28) / 0.72);

                    // Asimetría Doppler sutil en el toro
                    vec3 tangDir = normalize(vec3(-pos.z, 0.0, pos.x));
                    vec3 viewDir = normalize(cameraPosition - pos);
                    float doppler = uDopplerEnabled * dot(tangDir, viewDir) * clamp(uBetaEq * 1.8, 0.05, 0.35);

                    vColor = baseCol * wispRing * (1.0 + doppler * 0.65);

                    // Opacidad baja por partícula + tamaño amplio = neblina volumétrica continua (sin puntos duros)
                    float innerFade = smoothstep(1.35, 2.3, r);
                    float outerFade = 1.0 - smoothstep(7.8, 11.2, r);
                    vAlpha = innerFade * outerFade * 0.018 * seed;

                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = (420.0 * (0.65 + 0.6 * t)) / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vAlpha;
                void main() {
                    vec2 uv = gl_PointCoord - 0.5;
                    float d2 = dot(uv, uv) * 4.0;
                    if (d2 > 1.0) discard;
                    // Caída gaussiana ultra-suave
                    float gauss = exp(-d2 * 3.8) * (1.0 - d2);
                    gl_FragColor = vec4(vColor, gauss * vAlpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.points = new THREE.Points(geo, this.material);
        this.points.frustumCulled = false;
    }

    setPalette(innerHex, outerHex) {
        if (this.useGPGPU) {
            this.material.uniforms.uColorInner.value.set(innerHex);
            this.material.uniforms.uColorOuter.value.set(outerHex);
        }
    }

    update(time, audioBass, dopplerEnabled, betaEq, dt) {
        if (!this.useGPGPU) return;
        this.velUniforms["uTime"].value = time;
        this.velUniforms["uAudioWave"].value = audioBass;
        this.velUniforms.uStep.value = dt * 60;
        this.posVar.material.uniforms.uStep.value = dt * 60;
        if (dt > 0) this.gpuCompute.compute();

        this.material.uniforms.texturePosition.value =
            this.gpuCompute.getCurrentRenderTarget(this.posVar).texture;
        this.material.uniforms.uTime.value = time;
        this.material.uniforms.uDopplerEnabled.value = dopplerEnabled ? 1.0 : 0.0;
        this.material.uniforms.uBetaEq.value = betaEq;
    }
}

/* ==========================================================================
   3. REPRODUCTOR DE MÚSICA INTEGRADO (Con tus 3 archivos MP3)
   ========================================================================== */
class MusicDockController {
    constructor(onThemeChange) {
        this.onThemeChange = onThemeChange || (() => { });
        this.audio = document.getElementById('bg-audio');
        this.dock = document.getElementById('music-dock');
        this.disc = document.getElementById('disc-art');
        this.titleEl = document.getElementById('track-title');
        this.artistEl = document.getElementById('track-artist');
        this.playBtn = document.getElementById('btn-play');
        this.prevBtn = document.getElementById('btn-prev');
        this.nextBtn = document.getElementById('btn-next');
        this.seekSlider = document.getElementById('seek-slider');
        this.curTimeEl = document.getElementById('time-cur');
        this.durTimeEl = document.getElementById('time-dur');
        this.volSlider = document.getElementById('vol-slider');
        this.volIcon = document.getElementById('vol-icon');
        this.pillsContainer = document.getElementById('track-pills');
        this.fileInput = document.getElementById('file-mp3');

        this.tracks = [
            {
                title: "in the pool",
                artist: "Kensuke Ushio",
                src: "in the pool - kensuke ushio.mp3",
                altSrc: "musica/in the pool - kensuke ushio.mp3",
                inner: "#38bdf8",
                outer: "#312e81"
            },
            {
                title: "Sparks",
                artist: "Coldplay",
                src: "Sparks - Coldplay.mp3",
                altSrc: "musica/Sparks - Coldplay.mp3",
                inner: "#f59e0b",
                outer: "#4c1d95"
            },
            {
                title: "Yellow",
                artist: "Coldplay",
                src: "Yellow - Coldplay.mp3",
                altSrc: "musica/Yellow - Coldplay.mp3",
                inner: "#eab308",
                outer: "#1e3a8a"
            }
        ];

        this.idx = 0;
        this.isPlaying = false;
        this.audio.volume = 0.75;

        this._bind();
        this.load(0, false);
    }

    _fmt(s) {
        if (!Number.isFinite(s) || s < 0) return "0:00";
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${sec < 10 ? '0' : ''}${sec}`;
    }

    load(index, playNow = false) {
        this.idx = (index + this.tracks.length) % this.tracks.length;
        const t = this.tracks[this.idx];
        this.titleEl.textContent = t.title;
        this.artistEl.textContent = t.artist;
        this.disc.style.background =
            `radial-gradient(circle at center, ${t.inner} 0 28%, #090d16 29% 35%, transparent 36%), ` +
            `repeating-radial-gradient(circle, #111827 0 2px, #1e293b 3px 4px)`;

        this.triedAlt = false;
        this.audio.src = t.src;
        this.audio.load();

        this.pillsContainer.querySelectorAll('.song-pill').forEach((b, i) => {
            b.classList.toggle('active', i === this.idx);
        });

        this.onThemeChange(t);
        if (playNow) this.play();
    }

    play() {
        this.audio.play().then(() => {
            this.isPlaying = true;
            this._updateUI();
        }).catch(() => { });
    }

    pause() {
        this.audio.pause();
        this.isPlaying = false;
        this._updateUI();
    }

    _updateUI() {
        this.dock.classList.toggle('playing', this.isPlaying);
        this.playBtn.innerHTML = this.isPlaying
            ? '<i class="fa-solid fa-pause"></i>'
            : '<i class="fa-solid fa-play"></i>';
    }

    _bind() {
        this.playBtn.addEventListener('click', () => this.isPlaying ? this.pause() : this.play());
        this.prevBtn.addEventListener('click', () => this.load(this.idx - 1, true));
        this.nextBtn.addEventListener('click', () => this.load(this.idx + 1, true));

        this.audio.addEventListener('error', () => {
            const t = this.tracks[this.idx];
            if (!this.triedAlt && t.altSrc) {
                this.triedAlt = true;
                this.audio.src = t.altSrc;
                this.audio.load();
                if (this.isPlaying) this.play();
            }
        });

        this.audio.addEventListener('loadedmetadata', () => {
            this.durTimeEl.textContent = this._fmt(this.audio.duration);
        });

        this.audio.addEventListener('timeupdate', () => {
            if (this.audio.duration) {
                this.seekSlider.value = (this.audio.currentTime / this.audio.duration) * 100;
                this.curTimeEl.textContent = this._fmt(this.audio.currentTime);
            }
        });

        this.audio.addEventListener('ended', () => this.load(this.idx + 1, true));

        this.seekSlider.addEventListener('input', () => {
            if (this.audio.duration) {
                this.audio.currentTime = (this.seekSlider.value / 100) * this.audio.duration;
            }
        });

        this.volSlider.addEventListener('input', () => {
            this.audio.volume = parseFloat(this.volSlider.value);
        });

        this.pillsContainer.querySelectorAll('.song-pill').forEach((btn) => {
            btn.addEventListener('click', () => this.load(Number(btn.dataset.idx), true));
        });

        this.fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            if (!files.length) return;
            const colors = [
                { inner: "#38bdf8", outer: "#312e81" },
                { inner: "#f59e0b", outer: "#4c1d95" },
                { inner: "#eab308", outer: "#1e3a8a" }
            ];
            this.tracks = files.map((f, i) => ({
                title: f.name.replace(/\.[^/.]+$/, ""),
                artist: "Audio Local",
                src: URL.createObjectURL(f),
                ...colors[i % colors.length]
            }));
            this.pillsContainer.innerHTML = '';
            this.tracks.slice(0, 3).forEach((t, i) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = `song-pill ${i === 0 ? 'active' : ''}`;
                b.textContent = t.title;
                b.addEventListener('click', () => this.load(i, true));
                this.pillsContainer.appendChild(b);
            });
            this.load(0, true);
        });
    }
}

/* ==========================================================================
   4. MOTOR PRINCIPAL DE LA ESTRELLA DE NEUTRONES
   ========================================================================== */
class NeutronStarSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.clock = new THREE.Clock();

        this.params = {
            spinHz: 30.0,
            compactness: 0.35,
            tiltDeg: 45.0,
            logB: 12.6,
            dopplerEnabled: true,
            lensingEnabled: false,
            cutaway: false
        };

        this.spinPhase = 0;
        this.visualTime = 0;
        this.playback = 1;
        this.observation = 'composite';
        this.temperatureMK = 1.5;
        this.environmentStrength = 1;
        this.viewZoom = 1;
        this.paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        this.viewDistance = 5.2;
        this.cameraTransition = false;
        this.lastTime = 0;
        this.keys = { w: false, a: false, s: false, d: false };
        this.fwd = new THREE.Vector3();
        this.right = new THREE.Vector3();
        this.move = new THREE.Vector3();

        this._initScene();
        this._initPostProcessing();
        this._buildStarfield();
        this._buildDeepSpace();
        this._buildNeutronStarAndInterior();
        this._buildContinuousMagneticLines();
        this._buildRelativisticJets();
        this._buildNebula();
        this._buildHalo();
        this._buildBeams();
        this.dipoleGroup.visible = false;

        this.music = new MusicDockController((track) => {
            if (this.nebula) this.nebula.setPalette(track.inner, track.outer);
        });

        this._bindUI();
        document.getElementById('btn-pause-scene').textContent = this.paused ? 'Reanudar' : 'Pausar';
        this._updateReadouts();
        this._setObservation('composite');
        this._animate();
    }

    _getSize() {
        const r = this.canvas.getBoundingClientRect();
        return {
            width: Math.max(1, r.width || window.innerWidth),
            height: Math.max(1, r.height || window.innerHeight)
        };
    }

    _initScene() {
        const { width, height } = this._getSize();
        this.isMobile = width < 768;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(38, width / height, 0.05, 180);
        this.camera.position.set(0, 1.4, 5.0);

        this.rotorGroup = new THREE.Group();
        this.magneticGroup = new THREE.Group();
        this.rotorGroup.add(this.magneticGroup);
        this.scene.add(this.rotorGroup);
        this.magneticGroup.rotation.z = THREE.MathUtils.degToRad(this.params.tiltDeg);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: true,
            antialias: true,
            premultipliedAlpha: false
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.isMobile ? 1.25 : 1.75));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.05;

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 2.0;
        this.controls.rotateSpeed = 0.28;
        this.controls.zoomSpeed = 0.45;
        this.controls.enablePan = false;
        this.controls.addEventListener('start', () => { this.cameraTransition = false; });
        this.controls.maxDistance = 26.0;
    }

    _initPostProcessing() {
        const { width, height } = this._getSize();
        const renderPass = new THREE.RenderPass(this.scene, this.camera);

        // Umbral alto (0.88) y fuerza moderada (0.38) para que el Bloom NO queme la estrella ni el corte interno
        this.bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(width, Math.max(height, BLOOM_MIN_HEIGHT)),
            0.38,
            0.45,
            0.88
        );

        this.lensingPass = new THREE.ShaderPass(GravitationalLensingShader);
        this.lensingPass.renderToScreen = true;
        this.lensingPass.uniforms.uAspect.value = width / Math.max(1, height);

        this.composer = new THREE.EffectComposer(this.renderer);
        this.composer.addPass(renderPass);
        this.composer.addPass(this.bloomPass);
        this.composer.addPass(this.lensingPass);
    }

    _buildStarfield() {
        const count = this.isMobile ? 2500 : 4500;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(count * 3);
        const col = new Float32Array(count * 3);
        const size = new Float32Array(count);

        const palette = [
            new THREE.Color('#ffffff'),
            new THREE.Color('#cbd5e1'),
            new THREE.Color('#bae6fd'),
            new THREE.Color('#fef08a')
        ];

        for (let i = 0; i < count; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 40 + Math.random() * 65;

            pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            pos[i * 3 + 2] = r * Math.cos(phi);

            const c = palette[Math.floor(Math.random() * palette.length)];
            col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
            size[i] = 0.30 + Math.pow(Math.random(),7.0) * 2.8;
        }

        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
        geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));

        const mat = new THREE.ShaderMaterial({
            vertexShader: `
                attribute float aSize;
                attribute vec3 aColor;
                varying vec3 vColor;
                void main() {
                    vColor = aColor;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = clamp(aSize * 2.0, 0.65, 5.6);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                void main() {
                    vec2 uv = gl_PointCoord - 0.5;
                    float d = length(uv) * 2.0;
                    if (d > 1.0) discard;
                    gl_FragColor = vec4(vColor, exp(-d*d*7.0) * 0.72);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.backgroundStars = new THREE.Points(geo, mat);
        this.scene.add(this.backgroundStars);
    }

    // -------------------------------------------------------------------------
    // ESTRELLA DE NEUTRONES + CORTE TRANSVERSAL 3D NÍTIDO (SIN DESTELLO CEGADOR)
    // -------------------------------------------------------------------------
    _buildNeutronStarAndInterior() {
        const R = 1.0;
        const sphereGeo = new THREE.SphereGeometry(R, 80, 80);

        this.crustMaterial = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uCompactness: { value: this.params.compactness },
                uBetaEq: { value: 0.05 },
                uDopplerEnabled: { value: 1.0 },
                uBFieldNorm: { value: 0.6 },
                uCutaway: { value: 0.0 },
                uThermal: { value: 0.0 },
                uTemperature: { value: 1.5 },
                uSpinAxisWorld: { value: new THREE.Vector3(0, 1, 0) },
                uMagAxisWorld: { value: new THREE.Vector3(0, 1, 0) }
            },
            vertexShader: `
                varying vec3 vWorldNormal;
                varying vec3 vWorldPos;
                varying vec3 vLocalPos;

                void main() {
                    vLocalPos = position;
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    vWorldPos = worldPos.xyz;
                    vWorldNormal = normalize(mat3(modelMatrix) * normal);
                    gl_Position = projectionMatrix * viewMatrix * worldPos;
                }
            `,
            fragmentShader: `
                uniform float uTime;
                uniform float uCompactness;
                uniform float uBetaEq;
                uniform float uDopplerEnabled;
                uniform float uBFieldNorm;
                uniform float uCutaway;
                uniform float uThermal;
                uniform float uTemperature;
                uniform vec3 uSpinAxisWorld;
                uniform vec3 uMagAxisWorld;

                varying vec3 vWorldNormal;
                varying vec3 vWorldPos;
                varying vec3 vLocalPos;
                float hash3(vec3 p) { return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453); }
                float noise3(vec3 p) {
                    vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
                    return mix(mix(mix(hash3(i),hash3(i+vec3(1,0,0)),f.x),
                                   mix(hash3(i+vec3(0,1,0)),hash3(i+vec3(1,1,0)),f.x),f.y),
                               mix(mix(hash3(i+vec3(0,0,1)),hash3(i+vec3(1,0,1)),f.x),
                                   mix(hash3(i+vec3(0,1,1)),hash3(i+vec3(1,1,1)),f.x),f.y),f.z);
                }
                float fbm(vec3 p) {
                    float v=0.0, a=0.5;
                    for(int i=0;i<5;i++){v+=a*noise3(p);p=p*2.03+vec3(7.1,3.7,1.9);a*=0.5;}
                    return v;
                }
                void main() {
                    // Corte de un octante frontal limpio (+X, +Z, Y > -0.15)
                    if (uCutaway > 0.5 && vLocalPos.x > 0.0 && vLocalPos.z > 0.0 && vLocalPos.y > -0.15) {
                        discard;
                    }

                    vec3 N = normalize(vWorldNormal);
                    vec3 V = normalize(cameraPosition - vWorldPos);
                    float cosPsi = clamp(dot(N, V), 0.0, 1.0);

                    // Geodésica de Beloborodov
                    float invOneMinusRs = 1.0 / max(0.25, 1.0 - uCompactness);
                    float cosAlpha = uCompactness + (1.0 - uCompactness) * cosPsi; // approximate emission angle only; no ray tracing
                    float sinAlpha = sqrt(max(0.0, 1.0 - cosAlpha * cosAlpha));

                    vec3 tangentDir = normalize(N - cosPsi * V + vec3(1e-5));
                    vec3 emittedNormal = normalize(cosAlpha * V + sinAlpha * tangentDir);

                    float magCos = abs(dot(emittedNormal, normalize(uMagAxisWorld)));
                    float polarCap = pow(magCos, 22.0);

                    // Granulación magnética de la corteza de hierro
                    vec3 p = vLocalPos * 9.0;
                    float broad = fbm(p + vec3(0.0,uTime*0.018,0.0));
                    float fine = fbm(p*7.0 + broad*3.0);
                    float pattern = smoothstep(0.22,0.78,broad*0.7+fine*0.3);

                    vec3 colCold = vec3(0.055, 0.12, 0.20);
                    vec3 colWarm = vec3(0.48, 0.70, 0.86);
                    // Solo el casquete polar supera ligeramente 1.0 para un bloom sutil y controlado
                    vec3 colPole = vec3(1.15, 1.35, 1.55);

                    vec3 surface = mix(colCold, colWarm, pattern * 0.75);
                    surface = mix(surface, colPole, polarCap*0.85);
                    surface *= (0.42+0.58*pow(cosPsi,0.35));
                    surface += vec3(0.28,0.51,0.65)*pow(fine,4.0)*0.8;

                    // Boosting Doppler relativista
                    vec3 spinTang = cross(normalize(uSpinAxisWorld), N);
                    float betaLos = dot(spinTang, V) * uBetaEq * uDopplerEnabled;
                    float gamma = inversesqrt(max(0.1, 1.0 - uBetaEq * uBetaEq * uDopplerEnabled));
                    float delta = 1.0 / max(0.4, gamma * (1.0 - betaLos));
                    float dopplerBoost = clamp(pow(delta, 4.0), 0.55, 1.65);

                    vec3 dopplerTint = mix(
                        vec3(1.25, 0.58, 0.42),
                        vec3(0.68, 0.90, 1.30),
                        clamp(0.5 + betaLos * 1.8, 0.0, 1.0)
                    );
                    surface *= mix(vec3(1.0), dopplerTint, abs(betaLos) * 2.2) * dopplerBoost;

                    // Limbo gravitatorio controlado
                    float limb = pow(1.0 - cosPsi, 2.4) * (0.35 + 0.35 * uCompactness);
                    surface += vec3(0.28, 0.65, 0.95) * limb;

                    // Illustrative temperature map, not visible-light blackbody color.
                    float heat = clamp((uTemperature*(0.75+0.25*pattern+polarCap*0.5)-0.3)/3.7,0.0,1.0);
                    vec3 heatColor = mix(vec3(0.12,0.025,0.25),vec3(1.3,0.23,0.045),smoothstep(0.0,0.55,heat));
                    heatColor = mix(heatColor,vec3(1.65,1.45,0.8),smoothstep(0.4,1.0,heat));
                    surface *= 0.65+0.35*pow(uTemperature/1.5,0.35);
                    surface = mix(surface,heatColor*(0.55+0.45*cosPsi),uThermal);
                    gl_FragColor = vec4(surface, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });

        this.starMesh = new THREE.Mesh(sphereGeo, this.crustMaterial);
        this.magneticGroup.add(this.starMesh);

        // Construir la anatomía interna 3D (con colores < 0.95 para que el Bloom NUNCA los queme)
        this._buildCrispInternalCutaway(R);
    }

    _buildCrispInternalCutaway(R) {
        this.interiorGroup = new THREE.Group();
        this.interiorGroup.visible = false;

        this.interiorMat = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uRadius: { value: R }
            },
            vertexShader: `
                varying vec3 vPos;
                varying vec3 vNormal;
                void main() {
                    vPos = position;
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float uTime;
                uniform float uRadius;
                varying vec3 vPos;
                varying vec3 vNormal;

                void main() {
                    // Recortar fuera de la esfera y mantener solo el octante abierto (+X, +Z, Y >= -0.15)
                    float r = length(vPos) / uRadius;
                    if (r > 1.0 || vPos.x < -0.002 || vPos.z < -0.002 || vPos.y < -0.152) discard;

                    vec3 col = vec3(0.0);

                    // 1. Ultranúcleo de Quarks u, d, s (0.00 - 0.35 R)
                    if (r < 0.35) {
                        float q = 1.0 - (r / 0.35);
                        col = mix(vec3(0.42, 0.14, 0.68), vec3(0.82, 0.55, 0.96), pow(q, 1.2));
                    }
                    // 2. Núcleo Externo: Superfluido de Neutrones con Vórtices Cuánticos (0.35 - 0.80 R)
                    else if (r < 0.80) {
                        float cylR = length(vPos.xz);
                        float vortexLines = smoothstep(0.65, 0.95, 0.5 + 0.5 * sin(cylR * 55.0));
                        col = mix(vec3(0.04, 0.28, 0.48), vec3(0.15, 0.72, 0.86), 0.35 + 0.65 * vortexLines);
                    }
                    // 3. Corteza Interna: Pasta Nuclear (Gnocchi, Spaghetti, Lasaña) (0.80 - 0.92 R)
                    else if (r < 0.92) {
                        float lasagne = 0.5 + 0.5 * sin(r * 170.0 + sin(vPos.y * 40.0) * 2.0);
                        col = mix(vec3(0.52, 0.22, 0.04), vec3(0.92, 0.62, 0.12), lasagne);
                    }
                    // 4. Corteza Externa: Red Cristalina de Hierro-56 (0.92 - 1.00 R)
                    else {
                        col = vec3(0.22, 0.65, 0.92);
                    }

                    // Anillos nítidos de separación entre fases
                    float line1 = smoothstep(0.008, 0.0, abs(r - 0.35));
                    float line2 = smoothstep(0.008, 0.0, abs(r - 0.80));
                    float line3 = smoothstep(0.007, 0.0, abs(r - 0.92));
                    col = mix(col, vec3(0.92, 0.96, 1.0), (line1 + line2 + line3) * 0.85);

                    // Sombreado tridimensional entre los tres planos de corte para dar relieve real
                    float shade = 0.78 + 0.22 * abs(vNormal.y);
                    gl_FragColor = vec4(col * shade, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });

        const planeSize = R * 2.05;
        const pGeo = new THREE.PlaneGeometry(planeSize, planeSize, 1, 1);

        // Plano Z = 0
        const pXY = new THREE.Mesh(pGeo, this.interiorMat);
        // Plano X = 0
        const pZY = new THREE.Mesh(pGeo, this.interiorMat);
        pZY.rotation.y = Math.PI * 0.5;
        // Plano Y = -0.15
        const pXZ = new THREE.Mesh(pGeo, this.interiorMat);
        pXZ.rotation.x = Math.PI * 0.5;
        pXZ.position.y = -0.15;

        // Esfera sólida en el centro para que el ultranúcleo de quarks se vea en 3D volumétrico
        const quarkCoreGeo = new THREE.SphereGeometry(R * 0.348, 48, 48);
        const quarkCoreMat = new THREE.MeshBasicMaterial({ color: 0x9333ea });
        const quarkCoreMesh = new THREE.Mesh(quarkCoreGeo, quarkCoreMat);

        this.interiorGroup.add(pXY, pZY, pXZ, quarkCoreMesh);
        this.magneticGroup.add(this.interiorGroup);
    }

    // -------------------------------------------------------------------------
    // LÍNEAS DE CAMPO DIPOLAR CONTINUAS Y SEDOSAS (r = L sin^2(theta))
    // -------------------------------------------------------------------------
    _buildContinuousMagneticLines() {
        this.dipoleGroup = new THREE.Group();
        const shells = [2.0, 3.1, 4.4];
        const azimuths = 16;
        const segments = 90;

        this.fieldLineMat = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uBBoost: { value: 1.0 }
            },
            vertexShader: `
                attribute float aProgress;
                attribute float aPhase;
                uniform float uTime;
                varying float vFlow;
                varying float vEndFade;

                void main() {
                    // Pulso de plasma viajando a lo largo de la línea dipolar cerrada
                    float wave = fract(aProgress * 2.0 - uTime * 0.45 + aPhase);
                    vFlow = smoothstep(0.0, 0.3, wave) * (1.0 - smoothstep(0.3, 1.0, wave));
                    vEndFade = sin(aProgress * 3.14159265);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float uBBoost;
                varying float vFlow;
                varying float vEndFade;
                void main() {
                    vec3 baseCol = mix(vec3(0.18, 0.55, 0.95), vec3(0.65, 0.90, 1.0), vFlow);
                    float alpha = (0.06 + 0.24 * vFlow) * vEndFade * uBBoost;
                    gl_FragColor = vec4(baseCol, alpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        shells.forEach((L) => {
            for (let a = 0; a < azimuths; a++) {
                const phi = (a / azimuths) * Math.PI * 2;
                const phaseVal = Math.random();
                const pts = new Float32Array((segments + 1) * 3);
                const prog = new Float32Array(segments + 1);
                const phases = new Float32Array(segments + 1);

                for (let i = 0; i <= segments; i++) {
                    const t = i / segments;
                    const theta = 0.14 + t * (Math.PI - 0.28);
                    const sinT = Math.sin(theta);
                    const r = L * sinT * sinT;

                    pts[i * 3] = r * sinT * Math.cos(phi);
                    pts[i * 3 + 1] = r * Math.cos(theta);
                    pts[i * 3 + 2] = r * sinT * Math.sin(phi);

                    prog[i] = t;
                    phases[i] = phaseVal;
                }

                const lineGeo = new THREE.BufferGeometry();
                lineGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
                lineGeo.setAttribute('aProgress', new THREE.BufferAttribute(prog, 1));
                lineGeo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

                this.dipoleGroup.add(new THREE.Line(lineGeo, this.fieldLineMat));
            }
        });

        this.magneticGroup.add(this.dipoleGroup);
    }

    // -------------------------------------------------------------------------
    // CHORROS RELATIVISTAS COLIMADOS (JETS DE SINCROTRÓN)
    // -------------------------------------------------------------------------
    _buildRelativisticJets() {
        const count = this.isMobile ? 6000 : 11000;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(count * 3);
        const seed = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            const ang = Math.random() * Math.PI * 2;
            pos[i * 3] = Math.cos(ang);
            pos[i * 3 + 1] = i % 2 === 0 ? 1.0 : -1.0;
            pos[i * 3 + 2] = Math.sin(ang);

            seed[i * 3] = Math.random();
            seed[i * 3 + 1] = Math.pow(Math.random(), 1.8);
            seed[i * 3 + 2] = i % 2 === 0 ? 1.0 : -1.0;
        }

        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 3));

        this.jetMaterial = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uMagAxisWorld: { value: new THREE.Vector3(0, 1, 0) },
                uDopplerEnabled: { value: 1.0 }
            },
            vertexShader: `
                attribute vec3 aSeed;
                uniform float uTime;
                uniform vec3 uMagAxisWorld;
                uniform float uDopplerEnabled;
                varying vec3 vColor;
                varying float vAlpha;

                void main() {
                    float t = fract(aSeed.x + uTime * 1.25);
                    float rNorm = aSeed.y;
                    float pole = aSeed.z;

                    float yDist = 1.01 + t * 22.0;
                    float coneR = (0.06 + 0.65 * pow(t, 1.2)) * rNorm;
                    float twist = t * 8.0 - uTime * 5.0;

                    float rx = position.x * cos(twist) - position.z * sin(twist);
                    float rz = position.x * sin(twist) + position.z * cos(twist);
                    vec3 localP = vec3(rx * coneR, pole * yDist, rz * coneR);

                    vec3 worldP = (modelMatrix * vec4(localP, 1.0)).xyz;
                    vec3 toCam = normalize(cameraPosition - worldP);
                    float cosTheta = dot(normalize(uMagAxisWorld * pole), toCam);
                    float beam = mix(1.0, clamp(pow(1.0 / (1.0 - 0.5 * cosTheta), 2.0), 0.4, 2.2), uDopplerEnabled);

                    vColor = mix(vec3(0.85, 0.96, 1.15), vec3(0.25, 0.52, 0.98), t) * beam;
                    float fadeLen = smoothstep(0.0, 0.08, t) * (1.0 - smoothstep(0.55, 1.0, t));
                    vAlpha = fadeLen * (1.0 - 0.65 * rNorm) * 0.095;

                    vec4 mvPosition = modelViewMatrix * vec4(localP, 1.0);
                    gl_PointSize = (65.0 * (0.7 + 0.5 * t)) / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vAlpha;
                void main() {
                    vec2 uv = gl_PointCoord - 0.5;
                    float d2 = dot(uv, uv) * 4.0;
                    if (d2 > 1.0) discard;
                    float g = exp(-d2 * 3.5) * (1.0 - d2);
                    gl_FragColor = vec4(vColor, g * vAlpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        const jets = new THREE.Points(geo, this.jetMaterial);
        jets.frustumCulled = false; // vertices displaced in shader
        this.jets = jets;
        this.rotorGroup.add(jets); // extended PWN jets follow spin axis, not the rotating magnetic beam
    }

    _buildBeams() {
        this.beamGroup=new THREE.Group();
        const material=new THREE.ShaderMaterial({
            vertexShader:`varying vec2 v;void main(){v=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
            fragmentShader:`varying vec2 v;void main(){float a=sin(v.y*3.14159)*(0.025+0.03*pow(0.5+0.5*cos(v.x*6.283),4.0));gl_FragColor=vec4(0.35,0.65,1.0,a);}`,
            transparent:true,depthWrite:false,side:THREE.DoubleSide,blending:THREE.AdditiveBlending
        });
        for(const pole of [-1,1]){
            const beam=new THREE.Mesh(new THREE.CylinderGeometry(0.85,0.03,7.5,48,1,true),material);
            beam.position.y=pole*4.75;beam.scale.y=pole;
            this.beamGroup.add(beam);
        }
        this.beamGroup.visible=false;
        this.magneticGroup.add(this.beamGroup);
    }

    _buildHalo() {
        this.halo = new THREE.Mesh(new THREE.SphereGeometry(1.045,64,48),new THREE.ShaderMaterial({
            vertexShader:`varying vec3 n; varying vec3 p;
            void main(){vec4 w=modelMatrix*vec4(position,1.0);p=w.xyz;n=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
            fragmentShader:`varying vec3 n; varying vec3 p;
            void main(){float rim=pow(1.0-abs(dot(normalize(n),normalize(cameraPosition-p))),4.0);
            gl_FragColor=vec4(vec3(0.27,0.60,0.95),rim*0.24);}`,
            transparent:true,depthWrite:false,blending:THREE.AdditiveBlending
        }));
        this.rotorGroup.add(this.halo);
    }

    _buildDeepSpace() {
        // Distant emissive backdrop, not gas filling the local vacuum.
        this.skyMaterial = new THREE.ShaderMaterial({
            uniforms:{uStrength:{value:0.55}},
            vertexShader:`varying vec3 d;void main(){d=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
            fragmentShader:`varying vec3 d;uniform float uStrength;float sq(float x){return x*x;}
            float h(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
            float n(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
            return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z);}
            void main(){vec3 p=normalize(d);float cloud=n(p*6.0)*0.6+n(p*17.0)*0.28+n(p*49.0)*0.12;
            float band=exp(-sq(dot(p,normalize(vec3(0.4,1.0,0.2)))+0.15*(cloud-0.5))*26.0);
            float dust=smoothstep(0.37,0.66,n(p*22.0));
            vec3 col=mix(vec3(0.035,0.045,0.075),vec3(0.095,0.065,0.12),cloud);
            col*=band*cloud*(1.0-dust*0.78)*uStrength;
            gl_FragColor=vec4(col+vec3(0.0007,0.001,0.002),1.0);}`,
            side:THREE.BackSide,depthWrite:false
        });
        this.sky = new THREE.Mesh(new THREE.SphereGeometry(140,32,20),this.skyMaterial);
        this.sky.renderOrder=-10;
        this.scene.add(this.sky);
    }

    _buildNebula() {
        // Ray-integrated illustrative PWN: shock, outward wisps and diffuse envelope.
        // Star and PWN scales are compressed independently for inspection.
        this.windMaterial = new THREE.ShaderMaterial({
            uniforms:{uTime:{value:0},uStrength:{value:1},uXray:{value:0},uSteps:{value:this.isMobile?40:64}},
            vertexShader:`varying vec3 wp;void main(){wp=(modelMatrix*vec4(position,1.0)).xyz;gl_Position=projectionMatrix*viewMatrix*vec4(wp,1.0);}`,
            fragmentShader:`varying vec3 wp;uniform float uTime,uStrength,uXray,uSteps;float sq(float x){return x*x;}
            vec2 hitSphere(vec3 ro,vec3 rd,float r){float b=dot(ro,rd),c=dot(ro,ro)-r*r;float h=b*b-c;if(h<0.0)return vec2(-1.0);h=sqrt(h);return vec2(-b-h,-b+h);}
            void main(){vec3 ro=cameraPosition,rd=normalize(wp-ro);vec2 hit=hitSphere(ro,rd,14.0);
            float a=max(0.0,hit.x),b=hit.y;vec2 star=hitSphere(ro,rd,1.0);if(star.x>0.0)b=min(b,star.x);
            if(b<=a)discard;float stepLen=(b-a)/uSteps;vec4 acc=vec4(0.0);
            float jitter=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);
            for(int i=0;i<80;i++){if(float(i)>=uSteps)break;vec3 p=ro+rd*(a+(float(i)+jitter)*stepLen);
                float r=length(p.xz),ang=atan(p.z,p.x);float warp=sin(ang*5.0+r*1.4-uTime*0.4)*0.12;
                float height=p.y+warp;float envelope=exp(-sq(height/(0.24+r*0.065)));
                float edge=smoothstep(2.1,3.0,r)*(1.0-smoothstep(9.0,13.0,r));
                float wisps=pow(0.5+0.5*sin(r*4.6-uTime*1.3+sin(ang*3.0)*0.6),9.0);
                float filament=0.5+0.5*sin(ang*19.0+r*7.0+sin(ang*7.0-r*2.0));
                float shock=exp(-sq((r-3.2)/0.24))*exp(-height*height/0.10);
                float diffuse=exp(-dot(p,p)/70.0)*0.012*(0.6+0.4*sin(r*2.0+ang*4.0));
                float density=(envelope*edge*(0.025+wisps*(0.10+filament*0.07))+shock*0.14+diffuse)*uStrength;
                vec3 cool=mix(vec3(0.24,0.53,0.85),vec3(0.17,0.09,0.33),smoothstep(3.0,11.0,r));
                vec3 xr=mix(vec3(0.30,0.85,1.1),vec3(0.48,0.18,0.68),smoothstep(3.0,10.0,r));
                vec3 color=mix(cool,xr,uXray)+vec3(0.5,0.65,0.72)*shock;
                float alpha=1.0-exp(-density*stepLen);
                acc.rgb+=(1.0-acc.a)*alpha*color;acc.a+=(1.0-acc.a)*alpha;
                if(acc.a>0.96)break;
            }
            if(acc.a<0.001)discard;gl_FragColor=vec4(acc.rgb/max(acc.a,0.001),acc.a);}`,
            transparent:true,side:THREE.BackSide,depthWrite:false,depthTest:false
        });
        this.wind = new THREE.Mesh(new THREE.SphereGeometry(14,32,24),this.windMaterial);
        this.wind.frustumCulled=false;
        this.wind.renderOrder=5;
        this.scene.add(this.wind);
    }

    _setObservation(mode) {
        this.observation=mode;
        const thermal=mode==='thermal', field=mode==='field';
        this.crustMaterial.uniforms.uThermal.value=thermal?1:0;
        this.windMaterial.uniforms.uXray.value=mode==='xray'?1:0;
        this.dipoleGroup.visible=field || document.getElementById('chk-field').checked;
        this.jets.visible=!thermal && !field;
        document.querySelectorAll('[data-observation]').forEach(b=>b.classList.toggle('active',b.dataset.observation===mode));
        const descriptions={composite:'Composición artística · superficie y entorno en escalas distintas',thermal:'Mapa térmico · falso color · temperatura superficial adoptada',xray:'Rayos X ilustrativos · viento, choque y emisión sincrotrón',field:'Geometría magnética · líneas de campo como guía, no luz visible'};
        document.getElementById('observation-caption').textContent=descriptions[mode];
        document.getElementById('thermal-readouts').hidden=!thermal;
        this._updateThermal();
    }

    _updateThermal() {
        const T=this.temperatureMK*1e6;
        const g=Math.sqrt(1-this.params.compactness);
        // Uniform blackbody reference, Schwarzschild redshift, fixed R=12 km.
        const L=4*Math.PI*12000**2*5.670374419e-8*T**4*g*g;
        document.getElementById('thermal-readouts').textContent=
          `T superficial: ${this.temperatureMK.toFixed(2)} MK · T∞: ${(this.temperatureMK*g).toFixed(2)} MK · λmáx,∞: ${(2.897771955e-3/(T*g)*1e9).toFixed(2)} nm · L∞: ${L.toExponential(2)} W (cuerpo negro uniforme)`;
    }

    _bindUI() {
        window.addEventListener('resize', () => this._onResize());

        // Tecla H o botón superior para Modo Contemplación (Ocultar UI)
        const toggleZen = () => {
            document.body.classList.toggle('zen-active');
        };
        document.getElementById('btn-zen-mode')?.addEventListener('click', toggleZen);
        window.addEventListener('keydown', (e) => {
            const k = e.key.toLowerCase();
            if (k === 'h') toggleZen();
            if (k in this.keys) this.keys[k] = true;
        });
        window.addEventListener('keyup', (e) => {
            const k = e.key.toLowerCase();
            if (k in this.keys) this.keys[k] = false;
        });

        // Minimizar inspector
        document.getElementById('btn-collapse-inspector')?.addEventListener('click', () => {
            document.getElementById('inspector-panel')?.classList.toggle('collapsed');
        });

        // Selector Exterior vs Corte Interno
        const guideEl = document.getElementById('cutaway-guide');
        document.querySelectorAll('.mode-tab').forEach((btn) => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const isCutaway = btn.dataset.view === 'cutaway';
                this.params.cutaway = isCutaway;
                this.crustMaterial.uniforms.uCutaway.value = isCutaway ? 1.0 : 0.0;
                this.interiorGroup.visible = isCutaway;
                if (guideEl) guideEl.hidden = !isCutaway;

                // En modo Corte Interno reducimos el bloom y acercamos suavemente la cámara para ver las capas nítidas
                if (this.bloomPass) {
                    this.bloomPass.strength = isCutaway ? 0.16 : 0.38;
                }
            });
        });

        window.addEventListener('blur', () => { Object.keys(this.keys).forEach(k=>this.keys[k]=false); });
        document.getElementById('ctrl-pace').addEventListener('input', e => {
            this.playback = Number(e.target.value);
            document.getElementById('lbl-pace').textContent = this.playback.toFixed(2)+'×';
        });
        document.getElementById('btn-pause-scene').addEventListener('click', e => {
            this.paused = !this.paused;
            e.currentTarget.textContent = this.paused ? 'Reanudar' : 'Pausar';
        });
        document.getElementById('chk-field').addEventListener('change', e=>this.dipoleGroup.visible=e.target.checked || this.observation==='field');
        document.querySelectorAll('[data-camera]').forEach(btn=>btn.addEventListener('click',()=>{
            this.viewDistance = Number(btn.dataset.camera);
            this.cameraTransition = true;
            document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b===btn));
        }));
        document.querySelectorAll('[data-observation]').forEach(b=>b.addEventListener('click',()=>this._setObservation(b.dataset.observation)));
        document.getElementById('ctrl-size').addEventListener('input',e=>{
            this.viewZoom=Number(e.target.value);
            this.camera.zoom=this.viewZoom;this.camera.updateProjectionMatrix();
            document.getElementById('lbl-size').textContent=this.viewZoom.toFixed(2)+'×';
        });
        document.getElementById('ctrl-temperature').addEventListener('input',e=>{
            this.temperatureMK=Number(e.target.value);
            this.crustMaterial.uniforms.uTemperature.value=this.temperatureMK;
            document.getElementById('lbl-temperature').textContent=this.temperatureMK.toFixed(2)+' MK';
            this._updateThermal();
        });
        document.getElementById('ctrl-environment').addEventListener('input',e=>{this.environmentStrength=Number(e.target.value);});
        document.getElementById('ctrl-space').addEventListener('input',e=>{this.skyMaterial.uniforms.uStrength.value=Number(e.target.value);});
        document.getElementById('ctrl-quality').addEventListener('change',e=>{
            const high=e.target.value==='high';
            const auto=e.target.value==='auto';
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,auto?(this.isMobile?1.25:1.75):(high?2:1)));
            this.composer.setPixelRatio(this.renderer.getPixelRatio());
            this.windMaterial.uniforms.uSteps.value=auto?(this.isMobile?40:64):(high?80:40);
            this._onResize();
        });
        // Sliders
        const sSpin = document.getElementById('ctrl-spin');
        const sComp = document.getElementById('ctrl-compact');
        const sTilt = document.getElementById('ctrl-tilt');
        const sB = document.getElementById('ctrl-bfield');

        sSpin?.addEventListener('input', (e) => {
            this.params.spinHz = parseFloat(e.target.value);
            this._updateReadouts();
        });
        sComp?.addEventListener('input', (e) => {
            this.params.compactness = parseFloat(e.target.value);
            this._updateReadouts();
        });
        sTilt?.addEventListener('input', (e) => {
            this.params.tiltDeg = parseFloat(e.target.value);
            this.magneticGroup.rotation.z = THREE.MathUtils.degToRad(this.params.tiltDeg);
            this._updateReadouts();
        });
        sB?.addEventListener('input', (e) => {
            this.params.logB = parseFloat(e.target.value);
            this._updateReadouts();
        });

        document.getElementById('chk-doppler')?.addEventListener('change', (e) => {
            this.params.dopplerEnabled = e.target.checked;
            this._updateReadouts();
        });
        document.getElementById('chk-lensing')?.addEventListener('change', (e) => {
            this.params.lensingEnabled = e.target.checked;
            this._updateReadouts();
        });

        // Preajustes
        const presets = {
            crab: { spinHz: 30, compactness: 0.35, tiltDeg: 45, logB: 12.6, title: "PSR B0531+21 (Cangrejo)" },
            magnetar: { spinHz: 2, compactness: 0.42, tiltDeg: 18, logB: 15.2, title: "SGR 1806-20 (Magnetar)" },
            millisecond: { spinHz: 716, compactness: 0.38, tiltDeg: 65, logB: 8.4, title: "PSR J1748-2446ad (716 Hz)" }
        };

        document.querySelectorAll('.preset-pill').forEach((btn) => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const p = presets[btn.dataset.preset];
                if (!p) return;

                this.params.spinHz = p.spinHz;
                this.params.compactness = p.compactness;
                this.params.tiltDeg = p.tiltDeg;
                this.params.logB = p.logB;

                sSpin.value = p.spinHz;
                sComp.value = p.compactness;
                sTilt.value = p.tiltDeg;
                sB.value = p.logB;

                this.magneticGroup.rotation.z = THREE.MathUtils.degToRad(p.tiltDeg);
                const titleEl = document.querySelector('.sim-title');
                if (titleEl) titleEl.textContent = p.title;

                this._updateReadouts();
            });
        });
    }

    _updateReadouts() {
        const nu = this.params.spinHz;
        const comp = this.params.compactness;
        const rLC = Math.round(299792.458 / (2 * Math.PI * nu));
        const betaEq = (2 * Math.PI * 12 * nu) / 299792.458;
        const zg = (1.0 / Math.sqrt(1.0 - comp)) - 1.0;
        const surfVis = Math.min(95.0, (1.0 / (2.0 * (1.0 - comp))) * 100.0);

        const oblateness = 1.0; // spherical reference; oblateness requires a rotating-star model
        this.starMesh.scale.set(1.0, oblateness, 1.0);
        this.interiorGroup.scale.set(1.0, oblateness, 1.0);

        this.visualBeta = betaEq;
        const bNorm = (this.params.logB - 8.0) / 7.5;

        this.crustMaterial.uniforms.uCompactness.value = comp;
        this.crustMaterial.uniforms.uBetaEq.value = this.visualBeta;
        this.crustMaterial.uniforms.uDopplerEnabled.value = this.params.dopplerEnabled ? 1.0 : 0.0;
        this.crustMaterial.uniforms.uBFieldNorm.value = bNorm;

        this.jetMaterial.uniforms.uDopplerEnabled.value = this.params.dopplerEnabled ? 1.0 : 0.0;
        this.fieldLineMat.uniforms.uBBoost.value = 0.65 + bNorm * 0.85;
        this.lensingPass.uniforms.uEnabled.value = this.params.lensingEnabled ? 1.0 : 0.0;

        document.getElementById('lbl-spin').textContent = `${nu.toFixed(0)} Hz`;
        document.getElementById('lbl-compact').textContent = comp.toFixed(2);
        document.getElementById('lbl-tilt').textContent = `${this.params.tiltDeg.toFixed(0)}°`;
        document.getElementById('lbl-bfield').innerHTML = `10<sup>${this.params.logB.toFixed(1)}</sup> G`;

        document.getElementById('out-zg').textContent = zg.toFixed(3);
        document.getElementById('out-surf').textContent = `${surfVis.toFixed(1)}%`;
        document.getElementById('out-rlc').textContent = `${rLC.toLocaleString()} km`;
        document.getElementById('out-beta').textContent = `${betaEq.toFixed(3)} c`;
        this._updateThermal();
    }

    _onResize() {
        const { width, height } = this._getSize();
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);
        this.bloomPass.setSize(width, Math.max(height, BLOOM_MIN_HEIGHT));
        this.lensingPass.uniforms.uAspect.value = width / Math.max(1, height);
    }

    _animate() {
        const elapsed = this.clock.getElapsedTime();
        const dt = Math.min(0.05, elapsed - this.lastTime);
        this.lastTime = elapsed;

        // Display a rotation over 90 s at 1x; actual frequency remains in physics readouts.
        const visualDt = this.paused ? 0 : dt * this.playback;
        this.visualTime += visualDt * 0.035;
        const slowTime = this.visualTime;
        this.spinPhase += (2*Math.PI/90) * visualDt;
        document.getElementById('time-scale').textContent = this.paused ? 'EN PAUSA' :
            `CÁMARA LENTA · 1:${Math.round(90*this.params.spinHz/this.playback).toLocaleString('es-PE')}`;
        this.rotorGroup.rotation.y = this.spinPhase;

        const magAxisWorld = new THREE.Vector3(0, 1, 0)
            .applyEuler(this.magneticGroup.rotation)
            .applyEuler(this.rotorGroup.rotation)
            .normalize();

        this.crustMaterial.uniforms.uTime.value = slowTime;
        this.crustMaterial.uniforms.uMagAxisWorld.value.copy(magAxisWorld);
        this.interiorMat.uniforms.uTime.value = slowTime;
        this.fieldLineMat.uniforms.uTime.value = slowTime;
        this.jetMaterial.uniforms.uTime.value = slowTime;
        this.jetMaterial.uniforms.uMagAxisWorld.value.set(0,1,0);

        const audioWave = 0; // music does not drive astrophysical dynamics
        if (this.nebula) {
            this.nebula.update(slowTime, audioWave, this.params.dopplerEnabled, this.visualBeta, visualDt*0.035);
        }

        // Actualizar posición en pantalla para la Lente de Einstein
        const starProj = new THREE.Vector3(0, 0, 0).project(this.camera);
        this.lensingPass.uniforms.uStarScreenUV.value.set(
            starProj.x * 0.5 + 0.5,
            starProj.y * 0.5 + 0.5
        );
        const camDist = Math.max(1.8, this.camera.position.length());
        this.lensingPass.uniforms.uEinsteinRad.value = (0.28 * this.params.compactness) / camDist;
        this.lensingPass.uniforms.uFrameDrag.value = (this.visualBeta || 0.05) * 0.10;

        // Vuelo libre WASD
        this.move.set(0, 0, 0);
        this.camera.getWorldDirection(this.fwd).normalize();
        this.right.copy(this.fwd).cross(this.camera.up).normalize();
        if (this.keys.w) this.move.add(this.fwd);
        if (this.keys.s) this.move.sub(this.fwd);
        if (this.keys.d) this.move.add(this.right);
        if (this.keys.a) this.move.sub(this.right);
        if (this.move.lengthSq() > 0) {
            this.move.normalize().multiplyScalar(0.65 * dt);
            this.camera.position.add(this.move);
            this.controls.target.add(this.move);
        }

        if (this.cameraTransition) {
            const direction = this.camera.position.clone().sub(this.controls.target).normalize();
            const distance = this.camera.position.distanceTo(this.controls.target);
            const next = THREE.MathUtils.lerp(distance,this.viewDistance,1-Math.exp(-dt*1.8));
            this.camera.position.copy(this.controls.target).addScaledVector(direction,next);
            if(Math.abs(next-this.viewDistance)<0.005) this.cameraTransition=false;
        }
        this.beamGroup.visible = !this.params.cutaway && (this.observation==='field' || document.getElementById('chk-beams').checked);
        this.jets.visible = !this.params.cutaway && this.observation!=='thermal' && this.observation!=='field';
        this.halo.visible = !this.params.cutaway && this.observation!=='field';
        this.wind.visible = !this.params.cutaway && this.observation!=='thermal' && this.observation!=='field';
        this.windMaterial.uniforms.uTime.value=slowTime;
        this.windMaterial.uniforms.uStrength.value=this.environmentStrength*(this.observation==='xray'?1.8:1.0);
        this.sky.position.copy(this.camera.position);
        this.backgroundStars.position.copy(this.camera.position);
        this.controls.update();
        this.composer.render();
        requestAnimationFrame(() => this._animate());
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('neutron-canvas');
    if (canvas) {
        try {
            window.neutronSim = new NeutronStarSimulation(canvas);
        } catch(error) {
            console.error(error);
            const notice = document.createElement('div');
            notice.className = 'startup-error';
            notice.textContent = 'No se pudo iniciar la escena 3D. Comprueba la conexión a Internet y activa WebGL o la aceleración gráfica en tu navegador.';
            document.body.appendChild(notice);
        }
    }
});