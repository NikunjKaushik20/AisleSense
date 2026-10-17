import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api/client";
import { mockDeadzones, mockDwell, mockJourneys, mockTips } from "./api/mocks";
import type { Flag, Result, SessionInfo, Site, SiteSummary, StoredSession } from "./api/types";
import { BeforeAfter } from "./components/BeforeAfter";
import { CameraTile } from "./components/CameraTile";
import { AtcBanner, FallbackBanner, Legend, Panel, Skeleton, SynthBanner } from "./components/bits";
import { CoverageSlider } from "./components/CoverageSlider";
import { DwellTable } from "./components/DwellTable";
import { FloorMap } from "./components/FloorMap";
import { Sankey } from "./components/Sankey";
import { Tips } from "./components/Tips";
import { TopBar } from "./components/TopBar";
import { ZoneEditor } from "./components/ZoneEditor";
import { fmtNum } from "./lib/grid";
import { useLive } from "./hooks/useLive";
import { usePoll } from "./hooks/usePoll";

type Tab = "dwell" | "journeys" | "deadzones" | "beforeafter" | "setup";
const TABS: { id: Tab; label: string }[] = [
  { id: "dwell", label: "Dwell" }, { id: "journeys", label: "Journeys" }, { id: "deadzones", label: "Dead zones" },
  { id: "beforeafter", label: "Before / after" }, { id: "setup", label: "Setup" },
];

function stored<T extends string>(key: string, fallback: T): T {
  try { return (localStorage.getItem(key) as T) || fallback; } catch { return fallback; }
}

export default function App() {
  const [sites, setSites] = useState<SiteSummary[] | null>(null);
  const [sitesErr, setSitesErr] = useState<string | null>(null);
  const [site, setSite] = useState<Site | null>(null);
  const [siteRev, setSiteRev] = useState(0);
  const [busy, setBusy] = useState(false);
  const [startErr, setStartErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>(() => {
    const h = window.location.hash.slice(1) as Tab; // demo deep link, e.g. /#deadzones
    return TABS.some((t) => t.id === h) ? h : stored<Tab>("aisle.tab", "dwell");
  });
  const [keep, setKeep] = useState(1);
  const [layers, setLayers] = useState({ heat: true, zones: true, tracks: true, coverage: true });
  const [now, setNow] = useState(Date.now());
  const live = useLive();
  const tabsRef = useRef<HTMLElement>(null);
  const [vh, setVh] = useState(() => window.innerHeight);

  useEffect(() => { const f = () => setVh(window.innerHeight); window.addEventListener("resize", f); return () => window.removeEventListener("resize", f); }, []);

  useEffect(() => { try { localStorage.setItem("aisle.tab", tab); } catch { /* private mode */ } }, [tab]);
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => {
    api.sites().then(setSites).catch((e) => setSitesErr(String(e.message ?? e)));
  }, []);

  const sess = usePoll<SessionInfo>(() => api.session(), 2000, []);
  const sessionId = sess.data?.session ?? null;
  const siteId = sess.data?.site ?? null;
  const siteKind = sites?.find((s) => s.id === siteId)?.kind;

  useEffect(() => {
    if (!siteId) { setSite(null); return; }
    api.site(siteId).then(setSite).catch(() => setSite(null));
  }, [siteId, siteRev]);

  const heat = usePoll(() => api.heatmap(sessionId), 1000, [sessionId], !!sessionId);
  const H = heat.data && heat.data.session === sessionId ? heat.data : null;
  const dzKeep = keep;
  const dz = usePoll(() => api.deadzones(sessionId, dzKeep, () => mockDeadzones(site?.floor ?? null, H, dzKeep, sessionId)),
    5000, [sessionId, dzKeep, !!site, !!H], !!sessionId);
  const dwell = usePoll(() => api.dwell(sessionId, () => mockDwell(site, H, sessionId)), 3000, [sessionId, !!site], !!sessionId);
  const tips = usePoll(() => api.tips(sessionId, () => mockTips(sessionId)), 5000, [sessionId], !!sessionId);
  const flagsQ = usePoll(() => api.flags(sessionId, () => ({ session: sessionId ?? "", flags: [] })), 3000, [sessionId], !!sessionId);
  const journeys = usePoll(() => api.journeys(sessionId, () => mockJourneys(live.transitions, sessionId)), 3000, [sessionId], !!sessionId);
  const storedSessions = usePoll<StoredSession[]>(() => api.sessions(), 0, [tab === "beforeafter", sessionId], tab === "beforeafter");

  const flags = useMemo(() => {
    const m = new Map<string, Flag>();
    for (const f of flagsQ.data?.data.flags ?? []) m.set(f.id, f);
    for (const f of live.msg?.flags ?? []) m.set(f.id, f);
    return [...m.values()];
  }, [flagsQ.data, live.msg]);

  const start = useCallback(async (id: string, mode: "live" | "replay", speed: number) => {
    setBusy(true); setStartErr(null);
    try { await api.start(id, mode, mode === "replay" ? speed : undefined); }
    catch (e) { setStartErr((e as Error).message); }
    finally { setBusy(false); }
  }, []);

  // On a 720p projector the tab panel starts low on the screen; bring it up when a tab is picked.
  const pickTab = (t: Tab) => {
    setTab(t);
    const el = tabsRef.current;
    if (el && el.getBoundingClientRect().top > window.innerHeight * 0.45) el.scrollIntoView({ behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  };

  if (sitesErr) return (
    <main className="fatal">
      <h1>AisleSense can't reach its server</h1>
      <p>The dashboard talks to the API on this same address. Start it from the <code>aislesense</code> folder with</p>
      <pre>python run.py --site synth_aisle</pre>
      <p className="muted">Error: {sitesErr}</p>
      <button type="button" className="primary" onClick={() => window.location.reload()}>Try again</button>
    </main>
  );

  const fps = live.msg?.session === sessionId ? live.msg?.fps ?? null : null;
  const stale = live.lastAt ? (now - live.lastAt) / 1000 : 0;
  const tracks = live.msg?.session === sessionId ? live.msg?.tracks ?? [] : [];
  const preview = (r: Result<unknown> | null) => !!r?.preview;
  const anyPreview = [dz.data, dwell.data, tips.data, journeys.data].some((r) => r?.preview);

  return (
    <div className="app">
      <TopBar sites={sites ?? []} siteId={siteId} session={sess.data} fps={fps} link={live.link} staleSec={stale}
        busy={busy} error={startErr} onStart={start} />
      <AtcBanner kind={siteKind} label={site?.label} />
      <SynthBanner kind={siteKind} label={site?.label} />
      <FallbackBanner text={(live.msg?.session === sessionId ? live.msg?.fallback : null) ?? sess.data?.fallback} />

      {!sessionId ? (
        <main className="empty-start">
          <h1>Pick a floor to watch</h1>
          <p>Choose a site above. Video sites run live from their camera; ATC and the synthetic aisle replay recorded positions at the speed you pick.</p>
          {!sites ? <Skeleton h={60} /> : (
            <div className="site-choices">
              {sites.map((s) => (
                <button key={s.id} type="button" disabled={busy} onClick={() => start(s.id, s.kind === "video" ? "live" : "replay", s.kind === "atc" ? 60 : 1)}>
                  <strong>{s.name}</strong><span>{s.label}</span>
                </button>
              ))}
            </div>
          )}
        </main>
      ) : (
        <main className="layout">
          <Panel id="map" title={site?.name ?? "Floor"} aside={
            <div className="layers" role="group" aria-label="Map layers">
              {(Object.keys(layers) as (keyof typeof layers)[]).map((l) => (
                <label key={l} className="check"><input type="checkbox" checked={layers[l]} onChange={(e) => setLayers({ ...layers, [l]: e.target.checked })} />{l === "coverage" ? "Confidence" : l[0].toUpperCase() + l.slice(1)}</label>
              ))}
            </div>
          }>
            {site && H ? (
              <FloorMap floor={site.floor} heat={H} dz={dz.data?.data ?? null} zones={site.zones} tracks={tracks} flags={flags}
                layers={layers} label={`Live heatmap of ${site.name}`} maxHeight={Math.max(260, Math.min(640, vh - 330))} />
            ) : heat.error ? <p className="error" role="alert">Heatmap unavailable: {heat.error}</p> : <Skeleton h={360} />}
            <div className="map-foot">
              <Legend />
              {dz.data?.preview && <span className="tag" title="Dead-zone and coverage marks are placeholder geometry until M4">Marks: preview</span>}
            </div>
          </Panel>

          <aside className="rail">
            {siteKind === "video" && site && site.cameras.length > 0 && (
              <Panel id="camera" title="What the camera sees">
                <CameraTile cams={site.cameras} />
              </Panel>
            )}
            <Panel id="tips" title="What to do next">
              <Tips tips={tips.data?.data ?? null} preview={preview(tips.data)} milestone={tips.data?.milestone} />
            </Panel>
            <Panel id="now" title="Right now">
              <dl className="facts">
                <div><dt>People on floor</dt><dd className="num">{fmtNum(tracks.length)}</dd></div>
                <div><dt>Tracks this session</dt><dd className="num">{fmtNum(sess.data?.n_tracks)}</dd></div>
                <div><dt>Media time</dt><dd className="num">{sess.data?.media_seconds !== undefined ? `${fmtNum((sess.data.media_seconds ?? 0) / 60, 1)} min` : "\u2014"}</dd></div>
                <div><dt>Hottest cell</dt><dd className="num">{H ? `${fmtNum(H.max, 0)} person-s` : "\u2014"}</dd></div>
                <div><dt>Flags</dt><dd className="num">{flags.length ? `${flags.filter((f) => f.type === "BLOCK").length} block \u00b7 ${flags.filter((f) => f.type === "BROWSE").length} browse` : "none"}</dd></div>
              </dl>
              {sess.data?.error && <p className="error" role="alert">{sess.data.error}</p>}
              {flagsQ.data?.preview && <p className="hint">Browse and block flags start with M4.</p>}
            </Panel>
          </aside>

          <section className="panel tabs-panel" ref={tabsRef}>
            <div className="tabs" role="tablist" aria-label="Analysis">
              {TABS.map((t) => (
                <button key={t.id} type="button" role="tab" id={`tab-${t.id}`} aria-selected={tab === t.id} aria-controls={`pane-${t.id}`} onClick={() => pickTab(t.id)}>{t.label}</button>
              ))}
            </div>
            <div className="tab-pane" role="tabpanel" id={`pane-${tab}`} aria-labelledby={`tab-${tab}`}>
              {tab === "dwell" && <DwellTable dwell={dwell.data?.data ?? null} preview={preview(dwell.data)} milestone={dwell.data?.milestone} />}
              {tab === "journeys" && <Sankey journeys={journeys.data?.data ?? null} preview={preview(journeys.data)} milestone={journeys.data?.milestone} />}
              {tab === "deadzones" && site && (
                <>
                  {siteKind !== "atc" && sites?.some((s) => s.id === "atc") && (
                    <div className="switch-note" role="note">
                      <p><strong>The dead-zone test runs on a real mall.</strong> The maps below are this site&rsquo;s live floor; the scored
                        comparison uses a full day of ATC mall trajectories.</p>
                      <button type="button" className="primary" disabled={busy} onClick={() => start("atc", "replay", 600)}>Switch to the ATC mall</button>
                    </div>
                  )}
                  <CoverageSlider floor={site.floor} heat={H} keep={keep} onKeep={setKeep} dz={dz.data?.data ?? null}
                    preview={preview(dz.data)} milestone={dz.data?.milestone} />
                </>
              )}
              {tab === "beforeafter" && site && <BeforeAfter siteId={site.id} floor={site.floor} sessions={storedSessions.data ?? []} />}
              {tab === "setup" && site && <ZoneEditor site={site} onSaved={() => setSiteRev((r) => r + 1)} />}
            </div>
          </section>
        </main>
      )}
      <footer className="foot">
        <span>No video frames are stored; only floor positions.{siteKind === "video" ? " The camera view is drawn live with heads pixelated." : ""}</span>
        {anyPreview && <span>Panels marked &ldquo;Coming&rdquo; show placeholders until their milestone lands.</span>}
      </footer>
    </div>
  );
}
