// PLACEHOLDERS for endpoints that still answer 501 (M3/M4 not landed).
// Rules: never invent a headline number. Where a placeholder can be computed honestly from real M1
// data (heatmap person-seconds, live track zones) it is; everything else is empty or clearly fake
// geometry, and every panel using it shows a "Preview" ribbon naming the milestone that replaces it.
import type { CellState, Deadzones, Diff, Dwell, Floor, Heatmap, Journeys, Site, Tips } from "./types";
import { cellsInPolygon } from "../lib/grid";

export function mockDwell(site: Site | null, heat: Heatmap | null, session: string | null): Dwell {
  const sections = (site?.zones ?? []).filter((z) => z.kind === "section").map((z) => {
    // person-seconds is real: summed from the M1 heatmap cells whose centre is inside the section
    const person_s = heat ? cellsInPolygon(heat, z.polygon).reduce((a, i) => a + (heat.cells[i] ?? 0), 0) : 0;
    return { section: z.name, person_s: Math.round(person_s), entries: 0, dwell_s: null, prior_s: null, n_long: 0 };
  });
  return {
    session: session ?? "", unit: "person-seconds per entry",
    prior: { name: "Prior (constant per section)", source: "arrives with M3" }, sections, long_stays: [],
  };
}

/** Fake geometry only: a blind patch whose coverage follows the slider, so the shading and the two
 *  maps can be exercised. `pooled` stays null: the real counts only ever come from the endpoint. */
export function mockDeadzones(floor: Floor | null, heat: Heatmap | null, keep: number, session: string | null): Deadzones {
  const cell = heat?.cell ?? floor?.cell ?? 1;
  const nx = heat?.nx ?? Math.max(1, Math.ceil((floor?.width ?? 20) / cell));
  const ny = heat?.ny ?? Math.max(1, Math.ceil((floor?.height ?? 10) / cell));
  const n = nx * ny;
  const bx0 = Math.floor(nx * 0.55), bx1 = Math.floor(nx * 0.8), by0 = Math.floor(ny * 0.3), by1 = Math.floor(ny * 0.7);
  const inBlind = (ix: number, iy: number) => ix >= bx0 && ix < bx1 && iy >= by0 && iy < by1;
  const base = Array.from({ length: n }, (_, i) => {
    const ix = i % nx, iy = Math.floor(i / nx);
    const h = heat?.cells[i];
    return h !== undefined ? h : Math.exp(-(((ix - nx / 2) / (nx / 3)) ** 2 + ((iy - ny / 2) / (ny / 3)) ** 2));
  });
  const coverage = base.map((_, i) => (inBlind(i % nx, Math.floor(i / nx)) ? keep : 1));
  const seen = base.map((v, i) => v * coverage[i]);
  const sorted = [...seen].filter((v) => v > 0).sort((a, b) => a - b);
  const cut = sorted.length ? sorted[Math.floor(sorted.length * 0.1)] : 0;
  const decile = seen.map((v) => v > 0 && v <= cut);
  const state: CellState[] = seen.map((v, i) => {
    if (coverage[i] < 0.25) return "unseen";
    if (base[i] === 0) return "unknown";
    return v <= cut * 0.2 ? "dead" : "ok";
  });
  return { session: session ?? "", keep, x0: heat?.x0 ?? floor?.x0 ?? 0, y0: heat?.y0 ?? floor?.y0 ?? 0, cell, nx, ny,
    state, decile, coverage, pooled: null, source: null };
}

/** Real but partial: transitions between sections tallied in this browser from /ws/live since page load. */
export function mockJourneys(counts: Map<string, number>, session: string | null): Journeys {
  const names: string[] = [];
  const idx = (s: string) => { let i = names.indexOf(s); if (i < 0) { names.push(s); i = names.length - 1; } return i; };
  const links = [...counts.entries()].map(([k, value]) => {
    const [a, b] = k.split("→");
    return { source: idx(a), target: idx(b), value };
  });
  return { session: session ?? "", nodes: names.map((name) => ({ name })), links };
}

export function mockTips(session: string | null): Tips {
  return {
    session: session ?? "",
    tips: [
      { id: "p1", kind: "UNSEEN", severity: 2, zone: null,
        text: "Hatched cells are floor the cameras barely see. AisleSense will not call them dead; it will ask for a camera there." },
      { id: "p2", kind: "BLOCK", severity: 1, zone: null,
        text: "Red marks will show people stopped in a walkway while others squeeze past. Shelf-front stops stay green." },
      { id: "p3", kind: "DEAD", severity: 1, zone: null,
        text: "Outlined cells will be floor that is well covered yet visited far less than its neighbours." },
    ],
  };
}

export function mockDiff(a: string, b: string): Diff { return { a, b, cleared: [], added: [], kept: [] }; }
