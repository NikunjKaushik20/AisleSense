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

/** The part of the floor worth drawing: walkway and shelf-front zones plus a margin, so a long shelf
 *  block doesn't eat half the projector. Sites without those zones (ATC, MERL) keep the whole floor. */
function walkableView(floor: Floor, zones: Zone[]): Floor {
  const walk = zones.filter((z) => z.kind !== "section");
  if (!walk.length) return floor;
  const xs = walk.flatMap((z) => z.polygon.map((p) => p[0])), ys = walk.flatMap((z) => z.polygon.map((p) => p[1]));
  const pad = 1.5;
  const x0 = Math.max(floor.x0, Math.min(...xs) - pad), x1 = Math.min(floor.x0 + floor.width, Math.max(...xs) + pad);
  const y0 = Math.max(floor.y0, Math.min(...ys) - pad), y1 = Math.min(floor.y0 + floor.height, Math.max(...ys) + pad);
  return { ...floor, x0, y0, width: x1 - x0, height: y1 - y0 };
}

interface FlagGroup { key: string; type: Flag["type"]; zone: string | null; n: number; x: number; y: number }

/** One marker per zone and flag type ("BROWSE x19 at Dairy front"), placed at the mean flag position.
 *  Flags outside any zone group by 2 m square. */
function groupFlags(flags: Flag[]): FlagGroup[] {
  const m = new Map<string, FlagGroup>();
  for (const f of flags) {
    const key = `${f.type}|${f.zone ?? `${Math.round(f.x / 2)},${Math.round(f.y / 2)}`}`;
    const g = m.get(key) ?? { key, type: f.type, zone: f.zone, n: 0, x: 0, y: 0 };
    g.n++; g.x += f.x; g.y += f.y;
    m.set(key, g);
  }
  return [...m.values()].map((g) => ({ ...g, x: g.x / g.n, y: g.y / g.n }));
}

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

  const view = useMemo(() => (variant === "full" ? walkableView(floor, zones) : floor), [floor, zones, variant]);
  const k = Math.min(w / view.width, maxHeight / view.height);
  const W = Math.round(view.width * k), H = Math.round(view.height * k);
  const sx = (x: number) => (x - view.x0) * k;
  const sy = (y: number) => H - (y - view.y0) * k;
  const groups = useMemo(() => groupFlags(flags), [flags]);

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
  }, [heat, dz, W, H, k, view, layers.heat, layers.coverage, variant]); // eslint-disable-line react-hooks/exhaustive-deps

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
    const x = view.x0 + (e.clientX - r.left) / k, y = view.y0 + (H - (e.clientY - r.top)) / k;
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
          {layers.zones && zones.map((z, zi) => (
            <g key={zi} className={`zone zone-${z.kind}`}>
              <polygon points={z.polygon.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
            </g>
          ))}
          {marks.map(({ i, kind }) => {
            const r = cellRect(i);
            return <rect key={`${kind}${i}`} className={`cell-${kind}`} data-state={kind}
              x={r.x} y={r.y} width={r.s} height={r.s} fill={kind === "unseen" ? "url(#hatch)" : undefined} />;
          })}
          {/* section names above the cell marks so a hatch or heat cell never hides them */}
          {layers.zones && variant === "full" && k * 2 > 14 && zones.filter((z) => z.kind === "section").map((z, zi) => {
            const cx = z.polygon.reduce((a, p) => a + p[0], 0) / z.polygon.length;
            const top = Math.min(Math.max(...z.polygon.map((p) => p[1])), view.y0 + view.height);
            if (top <= view.y0) return null;
            const half = z.name.length * 3.6 + 4; // keep the whole name on the map at phone width
            return <text key={zi} className="zone-label" x={Math.min(Math.max(sx(cx), half), W - half)} y={sy(top) + 15} textAnchor="middle">{z.name}</text>;
          })}
          {layers.tracks && variant === "full" && tracks.map((t) => (
            <circle key={t.id} className="track" cx={sx(t.x)} cy={sy(t.y)} r={Math.max(3, Math.min(6, k * 0.22))} />
          ))}
          {variant === "full" && groups.map((g) => {
            const px = sx(g.x), py = sy(g.y);
            const text = g.n > 1 ? `${g.type} ×${g.n}` : g.type;
            const pw = text.length * 7 + 14, ph = 20, r = Math.max(5, Math.min(8, k * 0.2));
            // BROWSE sits at a shelf, so its tag goes up into the shelf; BLOCK sits in the walkway, so its tag goes down
            const up = g.type === "BROWSE";
            const tx = Math.min(Math.max(px - pw / 2, 2), W - pw - 2);
            const ty = Math.min(Math.max(up ? py - r - 6 - ph : py + r + 6, 2), H - ph - 2);
            return (
              <g key={g.key} className={`flag flag-${g.type.toLowerCase()}`} data-flag={g.type} data-count={g.n}>
                <title>{`${g.n} ${g.type.toLowerCase()} flag${g.n === 1 ? "" : "s"}${g.zone ? ` at ${g.zone}` : ""}`}</title>
                <line x1={px} y1={py} x2={px} y2={up ? ty + ph : ty} />
                {g.type === "BLOCK"
                  ? <rect x={px - r} y={py - r} width={2 * r} height={2 * r} transform={`rotate(45 ${px} ${py})`} />
                  : <circle cx={px} cy={py} r={r} />}
                <rect className="flag-tag" x={tx} y={ty} width={pw} height={ph} rx={ph / 2} />
                <text x={tx + pw / 2} y={ty + ph / 2 + 4} textAnchor="middle">{text}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="map-readout" aria-live="off">{readout ?? " "}</div>
    </div>
  );
}
