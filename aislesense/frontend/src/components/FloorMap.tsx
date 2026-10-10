import { useEffect, useMemo, useRef, useState } from "react";
import type { Deadzones, Flag, Floor, Heatmap, LiveTrack, Zone } from "../api/types";
import { fmtNum, ramp, scale } from "../lib/grid";

export type MapVariant = "full" | "decile" | "aware";

interface Props {
  floor: Floor;
  heat: Heatmap | null;
  dz?: Deadzones | null;
  zones?: Zone[];
  tracks?: LiveTrack[];
  flags?: Flag[];
  variant?: MapVariant;
  layers?: { heat: boolean; zones: boolean; tracks: boolean; coverage: boolean };
  label?: string;
  maxHeight?: number;
  onPointer?: (p: { x: number; y: number } | null) => void;
}

const ALL = { heat: true, zones: true, tracks: true, coverage: true };

/** Floor plan: heatmap cells on a canvas (fast for ATC's 4.8k cells), every mark a judge reads in SVG on top.
 *  - confidence shading: cells veiled by (1 - coverage)
 *  - UNSEEN: grey hatch; DEAD: solid outline; WATCH: dashed outline   (aware rule)
 *  - decile variant: cells the bottom-decile rule calls dead, filled red   (default entry's rule) */
export function FloorMap({ floor, heat, dz, zones = [], tracks = [], flags = [], variant = "full", layers = ALL, label, maxHeight = 640, onPointer }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(800);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const k = Math.min(w / floor.width, maxHeight / floor.height);
  const W = Math.round(floor.width * k), H = Math.round(floor.height * k);
  const sx = (x: number) => (x - floor.x0) * k;
  const sy = (y: number) => H - (y - floor.y0) * k;

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    c.width = W * dpr; c.height = H * dpr;
    let ctx: CanvasRenderingContext2D | null = null;
    try { ctx = c.getContext("2d"); } catch { ctx = null; }
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = ramp(0);
    ctx.fillRect(0, 0, W, H);
    if (heat && layers.heat) {
      for (let iy = 0; iy < heat.ny; iy++) for (let ix = 0; ix < heat.nx; ix++) {
        const v = heat.cells[iy * heat.nx + ix];
        if (!v) continue;
        ctx.fillStyle = ramp(scale(v, heat.max));
        const x = sx(heat.x0 + ix * heat.cell), y = sy(heat.y0 + (iy + 1) * heat.cell);
        ctx.fillRect(x, y, heat.cell * k + 0.5, heat.cell * k + 0.5);
      }
    }
    if (dz && layers.coverage && variant !== "decile") {
      for (let i = 0; i < dz.coverage.length; i++) {
        const cov = dz.coverage[i];
        if (cov >= 0.999) continue;
        const ix = i % dz.nx, iy = Math.floor(i / dz.nx);
        ctx.fillStyle = `rgba(246,243,236,${(0.75 * (1 - cov)).toFixed(3)})`;
        ctx.fillRect(sx(dz.x0 + ix * dz.cell), sy(dz.y0 + (iy + 1) * dz.cell), dz.cell * k + 0.5, dz.cell * k + 0.5);
      }
    }
  }, [heat, dz, W, H, k, layers.heat, layers.coverage, variant]); // eslint-disable-line react-hooks/exhaustive-deps

  const marks = useMemo(() => {
    if (!dz) return [];
    const out: { i: number; kind: "unseen" | "dead" | "watch" | "decile" }[] = [];
    for (let i = 0; i < dz.state.length; i++) {
      if (variant === "decile") { if (dz.decile[i]) out.push({ i, kind: "decile" }); continue; }
      const s = dz.state[i];
      if (s === "unseen" || s === "dead" || s === "watch") out.push({ i, kind: s });
    }
    return out;
  }, [dz, variant]);

  const grid = dz ?? heat;
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!grid) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = floor.x0 + (e.clientX - r.left) / k, y = floor.y0 + (H - (e.clientY - r.top)) / k;
    onPointer?.({ x, y });
    const ix = Math.floor((x - grid.x0) / grid.cell), iy = Math.floor((y - grid.y0) / grid.cell);
    setHover(ix >= 0 && iy >= 0 && ix < grid.nx && iy < grid.ny ? iy * grid.nx + ix : null);
  };

  const cellRect = (i: number) => {
    const g = dz!;
    const ix = i % g.nx, iy = Math.floor(i / g.nx);
    return { x: sx(g.x0 + ix * g.cell), y: sy(g.y0 + (iy + 1) * g.cell), s: g.cell * k };
  };

  const readout = hover !== null && grid ? (() => {
    const ix = hover % grid.nx, iy = Math.floor(hover / grid.nx);
    const parts = [`cell ${ix},${iy}`];
    if (heat && heat.nx === grid.nx) parts.push(`${fmtNum(heat.cells[hover], 1)} person-s`);
    if (dz) {
      parts.push(`coverage ${Math.round((dz.coverage[hover] ?? 1) * 100)}%`);
      const st = variant === "decile" ? (dz.decile[hover] ? "decile: dead" : "decile: ok") : dz.state[hover];
      parts.push(st.toUpperCase());
    }
    return parts.join(" · ");
  })() : null;

  return (
    <div className="map" ref={wrap} data-variant={variant}>
      <div className="map-stage" style={{ width: W, height: H }}>
        <canvas ref={canvas} style={{ width: W, height: H }} aria-hidden="true" />
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={label ?? "Floor map"} onMouseMove={onMove} onMouseLeave={() => { setHover(null); onPointer?.(null); }}>
          <defs>
            <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="var(--unseen-bg)" />
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--unseen)" strokeWidth="2.2" />
            </pattern>
          </defs>
          {layers.zones && zones.map((z, zi) => {
            const pts = z.polygon.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ");
            const cx = z.polygon.reduce((a, p) => a + p[0], 0) / z.polygon.length;
            const top = Math.max(...z.polygon.map((p) => p[1]));
            return (
              <g key={zi} className={`zone zone-${z.kind}`}>
                <polygon points={pts} />
                {z.kind === "section" && variant === "full" && k * 2 > 14 &&
                  <text x={sx(cx)} y={sy(top) + 14} textAnchor="middle">{z.name}</text>}
              </g>
            );
          })}
          {marks.map(({ i, kind }) => {
            const r = cellRect(i);
            return <rect key={`${kind}${i}`} className={`cell-${kind}`} data-state={kind}
              x={r.x} y={r.y} width={r.s} height={r.s} fill={kind === "unseen" ? "url(#hatch)" : undefined} />;
          })}
          {layers.tracks && variant === "full" && tracks.map((t) => (
            <circle key={t.id} className="track" cx={sx(t.x)} cy={sy(t.y)} r={Math.max(3, Math.min(6, k * 0.22))} />
          ))}
          {variant === "full" && flags.map((f) => (
            <g key={f.id} className={`flag flag-${f.type.toLowerCase()}`} data-flag={f.type}>
              <circle cx={sx(f.x)} cy={sy(f.y)} r={Math.max(9, k * 0.6)} />
              <text x={sx(f.x)} y={sy(f.y) - Math.max(12, k * 0.6) - 4} textAnchor="middle">{f.type}</text>
            </g>
          ))}
        </svg>
      </div>
      <div className="map-readout" aria-live="off">{readout ?? " "}</div>
    </div>
  );
}
