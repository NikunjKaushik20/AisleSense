// Floor-grid helpers shared by the map, the mocks and the editor.
// Floor frame: metres, y up. Screen: px, y down. Cells are row-major with row = y index from y0.

export interface GridLike { x0: number; y0: number; cell: number; nx: number; ny: number }

export function pointInPolygon(x: number, y: number, poly: [number, number][] | number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}

/** Indices of cells whose centre lies inside the polygon. */
export function cellsInPolygon(g: GridLike, poly: [number, number][]): number[] {
  const out: number[] = [];
  for (let iy = 0; iy < g.ny; iy++)
    for (let ix = 0; ix < g.nx; ix++)
      if (pointInPolygon(g.x0 + (ix + 0.5) * g.cell, g.y0 + (iy + 0.5) * g.cell, poly)) out.push(iy * g.nx + ix);
  return out;
}

/** Sequential ramp for person-seconds: paper -> ink blue. Kept clear of the flag hues (green, red). */
const STOPS: [number, number, number][] = [
  [246, 243, 236], [214, 223, 238], [150, 174, 216], [84, 112, 178], [38, 52, 120], [22, 27, 74],
];

export function ramp(t: number): string {
  const v = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(v));
  const f = v - i;
  const c = STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** sqrt scaling so a few hot cells do not wash out the rest of the floor. */
export function scale(v: number, max: number): number {
  return max > 0 ? Math.sqrt(Math.max(0, v) / max) : 0;
}

export function fmtNum(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}
