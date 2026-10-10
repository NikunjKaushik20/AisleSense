// Contract types. M1 shapes come from docs/api.md as written.
// Shapes marked "M5 proposal" are what this UI expects from M3/M4 endpoints that still answer 501;
// they are also written in docs/api.md ("M5 additions") so the backend can match them.

export type SiteKind = "video" | "atc" | "synth";

export interface SiteSummary { id: string; name: string; kind: SiteKind; label: string | null }

export interface Zone { name: string; kind: "section" | "walkway" | "shelf_front"; polygon: [number, number][] }

export interface Floor { x0: number; y0: number; width: number; height: number; cell: number }

export interface Site {
  id: string; name: string; kind: SiteKind; label: string | null;
  floor: Floor; cameras: { id: string; source: string }[]; zones: Zone[];
}

export interface SessionStarted { session: string; site: string; mode: "live" | "replay"; speed: number; label: string; started: boolean }

export interface SessionInfo {
  session: string | null; running: boolean; site?: string; mode?: "live" | "replay"; speed?: number; label?: string;
  error?: string | null; n_events?: number; n_tracks?: number; media_seconds?: number; wall_seconds?: number;
  fps_effective?: number; realtime_factor?: number;
  /** M6: replay-fallback banner text while a live stream is down, else null. */
  fallback?: string | null;
}

export interface StoredSession { id: string; site: string; mode: string; started: number }

export interface Heatmap {
  session: string; site: string; t0: number | null; t1: number | null; unit: "person-seconds";
  x0: number; y0: number; cell: number; nx: number; ny: number;
  cells: number[]; max: number; births: number[]; deaths: number[];
}

export interface LiveTrack { id: string; x: number; y: number; zone: string | null }

export interface Flag {
  id: string; type: "BLOCK" | "BROWSE"; zone: string | null; x: number; y: number;
  t0: number; t1: number; evidence?: string;
}

export interface StreamHealth { cam: string; state: "connecting" | "live" | "replay" | "ended"; fps: number; since_frame_s: number | null }
export interface LiveMsg {
  session: string | null; t: number | null; fps: number; tracks: LiveTrack[]; flags: Flag[];
  /** M6 multi-stream video sessions only. */
  streams?: StreamHealth[]; fallback?: string | null;
}

// ---- M5 proposal: M3 dwell ----
export interface DwellRow {
  section: string; person_s: number; entries: number; dwell_s: number | null;
  prior_s: number | null; n_long: number;
}
export interface Dwell {
  session: string; unit: string; prior: { name: string; source: string };
  sections: DwellRow[]; long_stays: { track: string; section: string; seconds: number; t_in: number }[];
}

// ---- M5 proposal: M4 dead zones ----
export type CellState = "ok" | "dead" | "watch" | "unseen" | "unknown";
export interface DeadzoneArmCounts { n: number; aware_dead: number; aware_unknown: number; decile_dead: number }
export interface Deadzones {
  session: string; keep: number; x0: number; y0: number; cell: number; nx: number; ny: number;
  state: CellState[];      // aware rule (ours), row-major like /api/heatmap
  decile: boolean[];       // bottom-decile rule (the default entry)
  coverage: number[];      // 0..1 per cell, drives confidence shading
  pooled: { blind: DeadzoneArmCounts; ignored: DeadzoneArmCounts } | null; // from results/atc_deadzone.json
  source: string | null;   // e.g. "results/atc_deadzone.json"
}

// ---- M5 proposal: M4 flags / journeys / tips / diff ----
export interface Flags { session: string; flags: Flag[] }
export interface Journeys { session: string; nodes: { name: string }[]; links: { source: number; target: number; value: number }[] }
export type TipKind = "DEAD" | "UNSEEN" | "WATCH" | "BLOCK" | "BROWSE" | "DWELL";
export interface Tip { id: string; kind: TipKind; text: string; zone?: string | null; severity: 1 | 2 | 3 }
export interface Tips { session: string; tips: Tip[] }
export interface DiffItem { kind: TipKind; zone: string | null; text: string }
export interface Diff { a: string; b: string; cleared: DiffItem[]; added: DiffItem[]; kept: DiffItem[] }

/** Response of a contract call. `preview` = endpoint answered 501/404 and the payload is a labelled mock. */
export interface Result<T> { data: T; preview: boolean; milestone?: string }
