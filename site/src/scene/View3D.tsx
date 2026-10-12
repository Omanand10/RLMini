import { useEffect, useRef, useState, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { PerformanceMonitor } from "@react-three/drei";

/**
 * Canvas wrapper used by every 3D view:
 * - stops rendering entirely while scrolled off screen (frameloop "never")
 * - adapts the pixel ratio to the measured frame rate (PerformanceMonitor)
 */
export function View3D({ children, overlay, className, fov = 38 }: {
  children: ReactNode; overlay?: ReactNode; className?: string; fov?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [dpr, setDpr] = useState(1.25);
  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { rootMargin: "120px" });
    io.observe(ref.current!);
    const t = setTimeout(() => window.dispatchEvent(new Event("resize")), 300);   // re-measure once the grid has settled
    return () => { io.disconnect(); clearTimeout(t); };
  }, []);
  return (
    <div ref={ref} className={className}>
      <Canvas frameloop={visible ? "always" : "never"} dpr={dpr} shadows resize={{ scroll: false, debounce: { scroll: 50, resize: 0 } }}
        camera={{ fov, near: 0.1, far: 200 }} gl={{ antialias: true, powerPreference: "high-performance" }}>
        <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(1.5)} flipflops={3} onFallback={() => setDpr(1)} />
        {children}
      </Canvas>
      {overlay}
    </div>
  );
}
