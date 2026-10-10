import type {
  Deadzones, Diff, Dwell, Flags, Heatmap, Journeys, Result, SessionInfo, SessionStarted, Site,
  SiteSummary, StoredSession, Tips,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function raw(path: string, init?: RequestInit): Promise<Response> {
  return fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await raw(path, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const b = await r.json(); msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b); } catch { /* keep status */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

/** Endpoints that are "later" in docs/api.md answer 501. Then the UI shows `fallback()` and marks it preview. */
async function contract<T>(path: string, fallback: () => T, milestone: string): Promise<Result<T>> {
  const r = await raw(path);
  if (r.status === 501 || r.status === 404) return { data: fallback(), preview: true, milestone };
  if (!r.ok) throw new ApiError(r.status, `${path} -> ${r.status}`);
  return { data: (await r.json()) as T, preview: false };
}

const q = (params: Record<string, string | number | null | undefined>) => {
  const s = Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join("&");
  return s ? `?${s}` : "";
};

export const api = {
  health: () => json<{ status: string; version: string; session: string | null; running: boolean }>("/api/health"),
  sites: () => json<SiteSummary[]>("/api/sites"),
  site: (id: string) => json<Site>(`/api/site/${encodeURIComponent(id)}`),
  session: () => json<SessionInfo>("/api/session"),
  sessions: () => json<StoredSession[]>("/api/sessions"),
  start: (site: string, mode: "live" | "replay", speed?: number) =>
    json<SessionStarted>("/api/session", { method: "POST", body: JSON.stringify({ site, mode, speed }) }),
  heatmap: (session?: string | null, t0?: number, t1?: number) => json<Heatmap>(`/api/heatmap${q({ session, t0, t1 })}`),

  dwell: (session: string | null, fb: () => Dwell) => contract(`/api/dwell${q({ session })}`, fb, "M3"),
  deadzones: (session: string | null, keep: number, fb: () => Deadzones) =>
    contract(`/api/deadzones${q({ session, keep })}`, fb, "M4"),
  flags: (session: string | null, fb: () => Flags) => contract(`/api/flags${q({ session })}`, fb, "M4"),
  journeys: (session: string | null, fb: () => Journeys) => contract(`/api/journeys${q({ session })}`, fb, "M4"),
  tips: (session: string | null, fb: () => Tips) => contract(`/api/tips${q({ session })}`, fb, "M4"),
  diff: (a: string, b: string, fb: () => Diff) => contract(`/api/diff${q({ a, b })}`, fb, "M4"),

  saveZones: (site: string, fc: unknown) =>
    json<{ saved: boolean; n_zones: number; path: string }>(`/api/site/${site}/zones`, { method: "PUT", body: JSON.stringify(fc) }),
  saveCamera: (site: string, cam: string, image_points: number[][], floor_points: number[][]) =>
    json<{ saved: boolean; path: string }>(`/api/site/${site}/camera/${cam}`, { method: "PUT", body: JSON.stringify({ image_points, floor_points }) }),
  frameUrl: (site: string, cam: string) => `/api/site/${site}/camera/${cam}/frame`,
};

export function wsUrl(path = "/ws/live"): string {
  const { protocol, host } = window.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}${path}`;
}
