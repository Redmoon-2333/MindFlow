import { useCallback, useEffect, useState } from "react";

/**
 * Measured width of an element, for SVG charts.
 *
 * Why measure instead of stretching a fixed viewBox: with a 720-wide viewBox
 * scaled into a 320px card, every label shrinks to ~44% and becomes
 * unreadable. Rendering at the measured width keeps type at its real size on
 * phones and lets the layout reflow (labels thin out, bars narrow) instead.
 *
 * The ref is a callback, not an object ref: a chart that first renders its
 * empty state ("暂无数据") has no element to attach, and an object ref with
 * `[]` deps would never re-measure once data arrives — the series would then
 * be laid out at the fallback width and stretched by CSS (§8.19).
 *
 * SSR/初始渲染: `fallback` is returned until the first measurement.
 */
export function useElementWidth<T extends HTMLElement = HTMLDivElement>(
  fallback = 640,
): [(el: T | null) => void, number] {
  const [node, setNode] = useState<T | null>(null);
  const [width, setWidth] = useState(fallback);

  const ref = useCallback((el: T | null) => setNode(el), []);

  useEffect(() => {
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
  }, [node]);

  return [ref, width];
}
