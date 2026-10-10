import type { SessionInfo, SiteSummary } from "../api/types";
import type { LinkState } from "../hooks/useLive";

const SPEEDS = [1, 10, 60, 600];

interface Props {
  sites: SiteSummary[];
  siteId: string | null;
  session: SessionInfo | null;
  fps: number | null;
  link: LinkState;
  staleSec: number;
  busy: boolean;
  error: string | null;
  onStart: (site: string, mode: "live" | "replay", speed: number) => void;
}

/** Site picker, Live / Replay toggle with speed, and the stream-health badge. */
export function TopBar({ sites, siteId, session, fps, link, staleSec, busy, error, onStart }: Props) {
  const site = sites.find((s) => s.id === siteId);
  const isVideo = site?.kind === "video";
  const mode = session?.mode ?? (isVideo ? "live" : "replay");
  const speed = session?.speed ?? 1;

  let health: { cls: string; text: string };
  if (link !== "open") health = { cls: "bad", text: link === "connecting" ? "Connecting" : "Server unreachable" };
  else if (session?.error) health = { cls: "bad", text: "Stream error" };
  else if (!session?.running) health = { cls: "idle", text: "No stream" };
  else if (staleSec > 3) health = { cls: "warn", text: `Stalled ${Math.round(staleSec)} s` };
  else health = { cls: "ok", text: "Streaming" };

  return (
    <header className="topbar">
      <div className="brand">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <rect x="2.5" y="2.5" width="19" height="19" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <path d="M8 3v18M16 3v18" stroke="currentColor" strokeWidth="1.6" />
          <circle cx="12" cy="9" r="2.2" fill="var(--block)" />
        </svg>
        <span>AisleSense</span>
      </div>

      <label className="field">
        <span>Site</span>
        <select value={siteId ?? ""} disabled={busy}
          onChange={(e) => { const s = sites.find((x) => x.id === e.target.value); if (s) onStart(s.id, s.kind === "video" ? "live" : "replay", s.kind === "video" ? 1 : speed); }}>
          {!siteId && <option value="">Choose a site</option>}
          {sites.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
      </label>

      <div className="seg" role="group" aria-label="Mode">
        {(["live", "replay"] as const).map((m) => {
          const disabled = busy || !siteId || (m === "replay" && isVideo) || (m === "live" && site?.kind === "atc");
          return (
            <button key={m} type="button" aria-pressed={mode === m} disabled={disabled}
              title={m === "replay" && isVideo ? "Video sites run live: the file loops as a camera" : undefined}
              onClick={() => siteId && onStart(siteId, m, speed)}>{m === "live" ? "Live" : "Replay"}</button>
          );
        })}
      </div>

      {mode === "replay" && (
        <div className="seg" role="group" aria-label="Replay speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s} disabled={busy || !siteId}
              onClick={() => siteId && onStart(siteId, "replay", s)}>{s}&times;</button>
          ))}
        </div>
      )}

      <div className="spacer" />
      {error && <span className="topbar-error" role="alert">{error}</span>}
      <div className={`health health-${health.cls}`} data-testid="health">
        <i aria-hidden="true" />
        <span>{health.text}</span>
        <span className="num">{fps !== null ? `${fps.toFixed(1)} fps` : "— fps"}</span>
        {session?.realtime_factor ? <span className="num">{session.realtime_factor.toFixed(1)}&times; real time</span> : null}
      </div>
    </header>
  );
}
