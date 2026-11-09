import { useEffect, useRef, useState } from "react";

/**
 * Measured width of an element, for SVG charts.
 *
 * Why measure instead of stretching a fixed viewBox: with a 720-wide viewBox
 * scaled into a 320px card, every label shrinks to ~44% and becomes
 * unreadable. Rendering at the measured width keeps type at its real size on
 * phones and lets the layout reflow (labels thin out, bars narrow) instead.
 *
 * SSR/初始渲染: `fallback` is returned until the first measurement.
 */
export function useElementWidth<T extends HTMLElement = HTMLDivElement>(fallback = 640): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const read = () => {
      // clientWidth excludes scrollbars; 0 means "not laid out yet".
      const next = node.clientWidth;
      if (next > 0) setWidth(next);
    };
    read();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", read);
      return () => window.removeEventListener("resize", read);
    }
    const observer = new ResizeObserver(read);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
