import { useEffect, useRef, useCallback } from "react";
import "./particle.css";

/** Spring stiffness range — under-damped, so a pushed particle overshoots its
 *  home position once or twice before settling (§8.18.1). */
const K_MIN = 0.010;
const K_MAX = 0.008;
const FRICTION = 0.92;
const MAX_SPEED = 24;
/** Frame budget at 60 Hz; `dtScale` normalises any other refresh rate to it. */
const FRAME_MS = 16.667;

const MOUSE_RADIUS = 150;
const PUSH_SCALE = 1.4;
const SWIRL_SCALE = 0.8;
const BRIGHTNESS_BOOST = 20;
const LIGHTNESS_CEILING = 92;
const SPEED_CAP = 40;

const WAVE_LIFETIME = 1.2;
const WAVE_RADIUS_PER_SECOND = 650;
const WAVE_BAND = 36;
const WAVE_IMPULSE = 5;
const MAX_WAVES = 3;

const TENSION_MIN = 8;
const TENSION_MAX = 140;

/** A pointer-triggered shockwave: origin, spawn time, and age-derived radius. */
interface Shockwave {
  x: number;
  y: number;
  t0: number;
}

interface Particle {
  tx: number;
  ty: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  k: number;
  phase: number;
  floatSpeed: number;
  floatAmp: number;
  size: number;
  hue: number;
  lightness: number;
  /** Per-frame render values, written by the integrate pass and read by the
   *  draw pass (the two passes are separate so tension lines can be stroked
   *  beneath every particle in one path). */
  alpha?: number;
  renderLightness?: number;
}

/**
 * Particle rendering of the word MINDFLOW behind the login panel.
 *
 * Port of the reference `ParticleText.jsx` (sampling gap 4px, blue-cyan
 * particles), with three deliberate additions:
 *   - a seeded PRNG so visual tests get a deterministic layout;
 *   - a `reduce motion` path that draws the static sample once;
 *   - §8.18 physics: every particle is a spring, so the word billows slowly
 *     instead of gliding, and the pointer pushes, swirls and ripples it.
 */
export default function ParticleText() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef({ x: -9999, y: -9999, speed: 0 });
  const wavesRef = useRef<Shockwave[]>([]);
  const animIdRef = useRef<number | null>(null);

  // Deterministic PRNG (mulberry32) — identical particle layout per load so
  // screenshots of this canvas are comparable run to run.
  const makeRandom = useCallback((seed: number) => {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }, []);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = (canvas.width = window.innerWidth);
    const h = (canvas.height = window.innerHeight);

    // Draw text to sample pixel positions
    const fontSize = Math.min(w * 0.12, 200);
    ctx.font = `900 ${fontSize}px "Segoe UI", "Arial Black", sans-serif`;
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("MINDFLOW", w / 2, h / 2);

    const imageData = ctx.getImageData(0, 0, w, h);
    const pixels = imageData.data;

    const targets: Array<{ x: number; y: number }> = [];
    const gap = 4; // sampling gap - smaller = more particles
    for (let y = 0; y < h; y += gap) {
      for (let x = 0; x < w; x += gap) {
        const i = (y * w + x) * 4;
        if (pixels[i + 3] > 128) targets.push({ x, y });
      }
    }

    const random = makeRandom(0x4d494e44);
    particlesRef.current = targets.map((t) => ({
      tx: t.x,
      ty: t.y,
      x: random() * w,
      y: random() * h,
      vx: 0,
      vy: 0,
      k: K_MIN + random() * K_MAX,
      phase: random() * Math.PI * 2,
      floatSpeed: 0.15 + random() * 0.35,
      floatAmp: 1.5 + random() * 2.5,
      size: 1.2 + random() * 1.8,
      hue: 200 + random() * 40, // blue-cyan range
      lightness: 55 + random() * 25,
    }));
  }, [makeRandom]);

  useEffect(() => {
    init();

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resetPointer = () => {
      mouseRef.current = { x: -9999, y: -9999, speed: 0 };
    };

    const onPointerMove = (e: PointerEvent) => {
      mouseRef.current.x = e.clientX;
      mouseRef.current.y = e.clientY;
    };
    const onPointerOut = (e: PointerEvent) => {
      // Only a pointer leaving the window reports a null relatedTarget; a
      // pointer moving between elements must keep its position.
      if (e.relatedTarget === null) resetPointer();
    };
    const onPointerRelease = (e: PointerEvent) => {
      if (e.pointerType === "touch") resetPointer();
    };
    const onPointerDown = (e: PointerEvent) => {
      // The login card is an interactive surface: clicking a field or button
      // must not ripple the background behind it.
      if ((e.target as Element | null)?.closest?.(".login-card")) return;
      const waves = wavesRef.current;
      waves.push({ x: e.clientX, y: e.clientY, t0: performance.now() });
      while (waves.length > MAX_WAVES) waves.shift();
    };

    let time = 0;
    let last = 0;
    // Pointer position at the previous frame, for the smoothed speed term.
    let prevPointerX = -9999;
    let prevPointerY = -9999;

    const drawFrame = (now: number, advance: boolean) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      const dtScale = advance && last > 0 ? Math.min(Math.max((now - last) / FRAME_MS, 0), 2) : 1;
      last = now;
      if (advance) time += 0.008 * dtScale;

      const mouse = mouseRef.current;
      const pointerOnScreen = mouse.x > -9999;

      // Smoothed pointer speed, refreshed once per frame (§8.18.2).
      if (pointerOnScreen) {
        const moved = Math.hypot(mouse.x - prevPointerX, mouse.y - prevPointerY);
        mouse.speed = mouse.speed * 0.85 + (prevPointerX > -9999 ? moved : 0) * 0.15;
        prevPointerX = mouse.x;
        prevPointerY = mouse.y;
      } else {
        mouse.speed = 0;
        prevPointerX = -9999;
        prevPointerY = -9999;
      }

      // Shockwaves expire by age; the array keeps insertion order. A wave can
      // be spawned between two rAF timestamps, so its age is clamped at 0 —
      // a negative age makes the ring radius negative and ctx.arc throws.
      const waveAge = (wave: Shockwave) => Math.max(0, (now - wave.t0) / 1000);
      const waves = wavesRef.current;
      while (waves.length > 0 && waveAge(waves[0]) > WAVE_LIFETIME) waves.shift();

      if (!reduceMotion && pointerOnScreen) {
        const glow = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, MOUSE_RADIUS);
        glow.addColorStop(0, "rgba(110, 200, 255, 0.10)");
        glow.addColorStop(1, "rgba(110, 200, 255, 0)");
        ctx.fillStyle = glow;
        ctx.fillRect(mouse.x - MOUSE_RADIUS, mouse.y - MOUSE_RADIUS, MOUSE_RADIUS * 2, MOUSE_RADIUS * 2);
      }

      const particles = particlesRef.current;
      const pushStrength = PUSH_SCALE * (1 + Math.min(mouse.speed, SPEED_CAP) / 20);

      // ── Integrate, then record the stretch each particle is under ───────
      const stretches: Array<{ x: number; y: number; targetX: number; targetY: number }> = [];

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        const targetX = p.tx + Math.sin(time * p.floatSpeed + p.phase) * p.floatAmp;
        const targetY = p.ty + Math.cos(time * p.floatSpeed * 0.7 + p.phase + 1) * p.floatAmp;

        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        let fx = 0;
        let fy = 0;
        let proximity = 0;
        if (!reduceMotion && dist > 0 && dist < MOUSE_RADIUS) {
          proximity = (1 - dist / MOUSE_RADIUS) ** 2;
          const nx = dx / dist;
          const ny = dy / dist;
          const push = pushStrength * proximity;
          const swirl = SWIRL_SCALE * proximity;
          // Radial push away from the pointer plus a tangential swirl, so the
          // crowd parts and turns instead of just scattering.
          fx += nx * push - ny * swirl;
          fy += ny * push + nx * swirl;
        }

        // A passing wavefront drags the particles it crosses along its radius.
        for (let j = 0; j < waves.length; j++) {
          const wave = waves[j];
          const age = waveAge(wave);
          const radius = age * WAVE_RADIUS_PER_SECOND;
          const band = Math.abs(dist - radius);
          if (band < WAVE_BAND && dist > 0) {
            const impulse = WAVE_IMPULSE * (1 - band / WAVE_BAND) * (1 - age / WAVE_LIFETIME);
            fx += (dx / dist) * impulse;
            fy += (dy / dist) * impulse;
          }
        }

        const ax = (targetX - p.x) * p.k + fx;
        const ay = (targetY - p.y) * p.k + fy;

        p.vx = (p.vx + ax * dtScale) * FRICTION ** dtScale;
        p.vy = (p.vy + ay * dtScale) * FRICTION ** dtScale;

        const speed = Math.hypot(p.vx, p.vy);
        if (speed > MAX_SPEED) {
          const scale = MAX_SPEED / speed;
          p.vx *= scale;
          p.vy *= scale;
        }

        p.x += p.vx * dtScale;
        p.y += p.vy * dtScale;

        const disp = Math.hypot(p.x - targetX, p.y - targetY);
        if (disp > TENSION_MIN && disp < TENSION_MAX) {
          stretches.push({ x: p.x, y: p.y, targetX, targetY });
        }

        // Brightness rises as the pointer approaches, and the alpha formula is
        // the reference one, so the resting look is unchanged.
        const lightness = proximity > 0
          ? Math.min(LIGHTNESS_CEILING, p.lightness + BRIGHTNESS_BOOST * proximity)
          : p.lightness;
        const alpha = proximity > 0
          ? 0.6 + proximity * 0.4
          : 0.5 + Math.sin(time * 1.5 + p.phase) * 0.2;

        p.alpha = alpha;
        p.renderLightness = lightness;
      }

      // ── Shockwave rings ────────────────────────────────────────────────
      for (let j = 0; j < waves.length; j++) {
        const wave = waves[j];
        const age = waveAge(wave);
        ctx.beginPath();
        ctx.arc(wave.x, wave.y, age * WAVE_RADIUS_PER_SECOND, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(120, 210, 255, ${0.25 * (1 - age / WAVE_LIFETIME)})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // ── Tension lines: the stretched rubber bands, one path per frame ───
      if (stretches.length > 0) {
        ctx.beginPath();
        for (let i = 0; i < stretches.length; i++) {
          const s = stretches[i];
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(s.targetX, s.targetY);
        }
        ctx.strokeStyle = "rgba(120, 200, 255, 0.16)";
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }

      // ── Particles ──────────────────────────────────────────────────────
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const lightness = p.renderLightness ?? p.lightness;
        const alpha = p.alpha ?? 0.5;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, ${lightness}%, ${alpha * 0.15})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 85%, ${lightness}%, ${alpha})`;
        ctx.fill();
      }
    };

    if (reduceMotion) {
      // Static sample: converge under the same springs once, then stop. No
      // pointer state exists on this path, so nothing can disturb it.
      for (let i = 0; i < 240; i += 1) drawFrame(performance.now(), false);
    } else {
      const animate = (now: number) => {
        drawFrame(now, true);
        animIdRef.current = requestAnimationFrame(animate);
      };
      animIdRef.current = requestAnimationFrame(animate);

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerout", onPointerOut);
      window.addEventListener("pointerup", onPointerRelease);
      window.addEventListener("pointercancel", onPointerRelease);
      window.addEventListener("pointerdown", onPointerDown);
    }

    const onResize = () => {
      particlesRef.current = [];
      wavesRef.current = [];
      init();
    };
    window.addEventListener("resize", onResize);

    return () => {
      if (animIdRef.current != null) cancelAnimationFrame(animIdRef.current);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerout", onPointerOut);
      window.removeEventListener("pointerup", onPointerRelease);
      window.removeEventListener("pointercancel", onPointerRelease);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("resize", onResize);
    };
  }, [init]);

  return <canvas ref={canvasRef} className="particle-canvas" aria-hidden="true" />;
}
