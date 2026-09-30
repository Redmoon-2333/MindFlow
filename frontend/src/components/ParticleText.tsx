import { useEffect, useRef, useCallback } from "react";
import "./particle.css";

/**
 * Particle rendering of the word MINDFLOW behind the login panel.
 *
 * Port of the reference `ParticleText.jsx` (sampling gap 4px, blue-cyan
 * particles, 100px mouse repulsion radius) with two deliberate additions:
 *   - a seeded PRNG so visual tests get a deterministic layout, and
 *   - a `reduce motion` path that draws the static sample once.
 * The animation loop itself is unchanged from the reference.
 */
export default function ParticleText() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<
    Array<{
      tx: number;
      ty: number;
      x: number;
      y: number;
      ox: number;
      oy: number;
      phase: number;
      floatSpeed: number;
      floatAmp: number;
      size: number;
      hue: number;
      lightness: number;
      ease: number;
      idx: number;
    }>
  >([]);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const animIdRef = useRef<number | null>(null);
  const initializedRef = useRef(false);

  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

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
    particlesRef.current = targets.map((t, idx) => ({
      tx: t.x,
      ty: t.y,
      x: random() * w,
      y: random() * h,
      ox: 0,
      oy: 0,
      phase: random() * Math.PI * 2,
      floatSpeed: 0.3 + random() * 0.7,
      floatAmp: 1.5 + random() * 3,
      size: 1.2 + random() * 1.8,
      hue: 200 + random() * 40, // blue-cyan range
      lightness: 55 + random() * 25,
      ease: 0.03 + random() * 0.04,
      idx,
    }));
    initializedRef.current = true;
  }, [makeRandom]);

  useEffect(() => {
    init();

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 };
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseleave", onMouseLeave);

    let time = 0;
    const drawFrame = (advance: boolean) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      const mouse = mouseRef.current;
      const mouseRadius = 100;
      const mouseForce = 110;

      const particles = particlesRef.current;
      if (advance) time += 0.016;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        p.ox = Math.sin(time * p.floatSpeed + p.phase) * p.floatAmp;
        p.oy = Math.cos(time * p.floatSpeed * 0.7 + p.phase + 1) * p.floatAmp;

        const targetX = p.tx + p.ox;
        const targetY = p.ty + p.oy;

        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        let fx = 0;
        let fy = 0;
        if (dist < mouseRadius && dist > 0) {
          const force = (1 - dist / mouseRadius) * mouseForce;
          fx = (dx / dist) * force;
          fy = (dy / dist) * force;
        }

        p.x = lerp(p.x, targetX + fx, p.ease * 3);
        p.y = lerp(p.y, targetY + fy, p.ease * 3);

        const alpha =
          dist < mouseRadius
            ? 0.6 + (1 - dist / mouseRadius) * 0.4
            : 0.5 + Math.sin(time * 1.5 + p.phase) * 0.2;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, ${p.lightness}%, ${alpha * 0.15})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 85%, ${p.lightness}%, ${alpha})`;
        ctx.fill();
      }
    };

    if (reduceMotion) {
      // Static sample: converge the particles once, then stop.
      for (let i = 0; i < 120; i += 1) drawFrame(false);
    } else {
      const animate = () => {
        drawFrame(true);
        animIdRef.current = requestAnimationFrame(animate);
      };
      animIdRef.current = requestAnimationFrame(animate);
    }

    const onResize = () => {
      initializedRef.current = false;
      particlesRef.current = [];
      init();
    };
    window.addEventListener("resize", onResize);

    return () => {
      if (animIdRef.current != null) cancelAnimationFrame(animIdRef.current);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseleave", onMouseLeave);
      window.removeEventListener("resize", onResize);
    };
  }, [init, makeRandom]);

  return <canvas ref={canvasRef} className="particle-canvas" aria-hidden="true" />;
}
