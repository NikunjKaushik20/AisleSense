import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Site, Zone } from "../api/types";

type Kind = Zone["kind"];
const KINDS: { v: Kind; label: string }[] = [
  { v: "section", label: "Section" }, { v: "walkway", label: "Walkway" }, { v: "shelf_front", label: "Shelf front" },
];

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, w };
}

/** Floor-plan editor: click to place vertices, close the shape, name it, save to sites/<id>/zones.geojson. */
function Zones({ site, onSaved }: { site: Site; onSaved: () => void }) {
  const { ref, w } = useWidth();
  const [zones, setZones] = useState<Zone[]>(site.zones);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [kind, setKind] = useState<Kind>("section");
  const [name, setName] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setZones(site.zones); setDraft([]); }, [site.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const f = site.floor;
  const k = Math.min(w / f.width, 460 / f.height);
  const W = f.width * k, H = f.height * k;
  const sx = (x: number) => (x - f.x0) * k, sy = (y: number) => H - (y - f.y0) * k;
  const snap = (v: number) => Math.round(v * 4) / 4; // 25 cm

  const click = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setDraft((d) => [...d, [snap(f.x0 + (e.clientX - r.left) / k), snap(f.y0 + (H - (e.clientY - r.top)) / k)]]);
  };
  const close = () => {
    if (draft.length < 3 || !name.trim()) return;
    setZones((z) => [...z, { name: name.trim(), kind, polygon: [...draft, draft[0]] }]);
    setDraft([]); setName(""); setMsg(null);
  };
  const save = async () => {
    setSaving(true); setMsg(null);
    try {
      const fc = { type: "FeatureCollection", features: zones.map((z) => ({ type: "Feature", properties: { name: z.name, kind: z.kind }, geometry: { type: "Polygon", coordinates: [z.polygon] } })) };
      const r = await api.saveZones(site.id, fc);
      setMsg({ ok: true, text: `Saved ${r.n_zones} zones to ${r.path}. New sessions use them.` });
      onSaved();
    } catch (e) { setMsg({ ok: false, text: `Not saved: ${(e as Error).message}` }); }
    finally { setSaving(false); }
  };

  return (
    <div className="editor">
      <div className="editor-tools">
        <div className="seg" role="group" aria-label="Zone kind">
          {KINDS.map((x) => <button key={x.v} type="button" aria-pressed={kind === x.v} onClick={() => setKind(x.v)}>{x.label}</button>)}
        </div>
        <label className="field"><span>Name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Snacks" /></label>
        <button type="button" onClick={close} disabled={draft.length < 3 || !name.trim()}
          title={draft.length < 3 ? "Click at least 3 points on the plan" : !name.trim() ? "Name the zone first" : undefined}>Close shape</button>
        <button type="button" className="ghost" onClick={() => setDraft((d) => d.slice(0, -1))} disabled={!draft.length}>Undo point</button>
        <div className="spacer" />
        <button type="button" className="primary" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save zones"}</button>
      </div>
      <p className="hint">Click the plan to place corners (snaps to 25 cm). Walkways are where people pass; shelf fronts are where they stop to browse. A stop on a walkway can become a block; a stop at a shelf never does.</p>
      <div className="editor-body">
      <div ref={ref}>
        <svg className="editor-plan" width={W} height={H} onClick={click} role="img" aria-label="Floor plan editor">
          {Array.from({ length: Math.floor(f.width / f.cell) + 1 }, (_, i) => <line key={`v${i}`} className="gridline" x1={i * f.cell * k} x2={i * f.cell * k} y1={0} y2={H} />)}
          {Array.from({ length: Math.floor(f.height / f.cell) + 1 }, (_, i) => <line key={`h${i}`} className="gridline" y1={H - i * f.cell * k} y2={H - i * f.cell * k} x1={0} x2={W} />)}
          {zones.map((z, i) => <g key={i} className={`zone zone-${z.kind}`}><polygon points={z.polygon.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} /></g>)}
          {draft.length > 0 && <polyline className="draft" points={draft.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />}
          {draft.map(([x, y], i) => <circle key={i} className="vertex" cx={sx(x)} cy={sy(y)} r={4} />)}
        </svg>
      </div>
      <div>
      <h3 className="sub">Zones <span className="num muted">{zones.length}</span></h3>
      <ul className="zone-list">
        {zones.length === 0 && <li className="empty">No zones yet.</li>}
        {zones.map((z, i) => (
          <li key={i}><i className={`sw sw-zone-${z.kind}`} aria-hidden="true" /><span>{z.name}</span><span className="muted">{KINDS.find((x) => x.v === z.kind)?.label}</span>
            <button type="button" className="ghost small" onClick={() => setZones((zs) => zs.filter((_, j) => j !== i))} aria-label={`Delete ${z.name}`}>Delete</button></li>
        ))}
      </ul>
      </div>
      </div>
      {msg && <p className={msg.ok ? "ok-msg" : "error"} role="status">{msg.text}</p>}
    </div>
  );
}

/** 4 clicks on the camera still, then the same 4 points on the floor plan -> homography saved in site.yaml. */
function Homography({ site, onSaved }: { site: Site; onSaved: () => void }) {
  const [cam, setCam] = useState(site.cameras[0]?.id ?? "");
  const [img, setImg] = useState<[number, number][]>([]);
  const [flr, setFlr] = useState<[number, number][]>([]);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [imgErr, setImgErr] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const { ref, w } = useWidth();
  useEffect(() => { setImg([]); setFlr([]); setMsg(null); setImgErr(false); }, [cam]);
  if (!site.cameras.length) return <p className="empty">This site has no cameras (positions come from {site.kind === "atc" ? "the dataset" : "the generator"}), so there is nothing to calibrate.</p>;

  const f = site.floor;
  const k = Math.min((w / 2 - 12) / f.width, 300 / f.height);
  const W = f.width * k, H = f.height * k;
  const step = img.length < 4 ? "image" : flr.length < 4 ? "floor" : "done";
  const save = async () => {
    try {
      const r = await api.saveCamera(site.id, cam, img, flr);
      setMsg({ ok: true, text: `Saved to ${r.path}. Restart the session to use it.` }); onSaved();
    } catch (e) { setMsg({ ok: false, text: `Not saved: ${(e as Error).message}` }); }
  };

  return (
    <div className="editor">
      <div className="editor-tools">
        <label className="field"><span>Camera</span>
          <select value={cam} onChange={(e) => setCam(e.target.value)}>{site.cameras.map((c) => <option key={c.id}>{c.id}</option>)}</select>
        </label>
        <span className="step num">{step === "image" ? `Click point ${img.length + 1} of 4 on the camera image` : step === "floor" ? `Click where point ${flr.length + 1} lies on the floor plan` : "4 pairs placed"}</span>
        <div className="spacer" />
        <button type="button" className="ghost" onClick={() => { setImg([]); setFlr([]); setMsg(null); }}>Start over</button>
        <button type="button" className="primary" disabled={step !== "done"} onClick={save}>Save homography</button>
      </div>
      <div className="homog" ref={ref}>
        <div className="homog-img">
          {imgErr ? <p className="empty">No still from this camera. Check its source in site.yaml.</p> : (
            <div className="img-wrap" onClick={(e) => {
              if (step !== "image" || !nat) return;
              const r = e.currentTarget.getBoundingClientRect();
              setImg((p) => [...p, [Math.round(((e.clientX - r.left) / r.width) * nat.w), Math.round(((e.clientY - r.top) / r.height) * nat.h)]]);
            }}>
              <img src={api.frameUrl(site.id, cam)} alt={`Still from ${cam}`} onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} onError={() => setImgErr(true)} />
              {nat && img.map(([u, v], i) => <span key={i} className="pin" style={{ left: `${(100 * u) / nat.w}%`, top: `${(100 * v) / nat.h}%` }}>{i + 1}</span>)}
            </div>
          )}
        </div>
        <svg className="editor-plan" width={W} height={H} role="img" aria-label="Floor points"
          onClick={(e) => {
            if (step !== "floor") return;
            const r = e.currentTarget.getBoundingClientRect();
            setFlr((p) => [...p, [Math.round((f.x0 + (e.clientX - r.left) / k) * 100) / 100, Math.round((f.y0 + (H - (e.clientY - r.top)) / k) * 100) / 100]]);
          }}>
          {site.zones.map((z, i) => <g key={i} className={`zone zone-${z.kind}`}><polygon points={z.polygon.map(([x, y]) => `${(x - f.x0) * k},${H - (y - f.y0) * k}`).join(" ")} /></g>)}
          {flr.map(([x, y], i) => <g key={i} className="pin-svg"><circle cx={(x - f.x0) * k} cy={H - (y - f.y0) * k} r={8} /><text x={(x - f.x0) * k} y={H - (y - f.y0) * k} dy="0.35em" textAnchor="middle">{i + 1}</text></g>)}
        </svg>
      </div>
      <p className="hint">Pick four floor marks visible in the camera, not all on one line (tile corners, shelf feet). The still is used for setup only and is not stored.</p>
      {msg && <p className={msg.ok ? "ok-msg" : "error"} role="status">{msg.text}</p>}
    </div>
  );
}

export function ZoneEditor({ site, onSaved }: { site: Site; onSaved: () => void }) {
  const [tab, setTab] = useState<"zones" | "camera">("zones");
  return (
    <div>
      <div className="seg" role="tablist" aria-label="Setup">
        <button type="button" role="tab" aria-selected={tab === "zones"} aria-pressed={tab === "zones"} onClick={() => setTab("zones")}>Zones</button>
        <button type="button" role="tab" aria-selected={tab === "camera"} aria-pressed={tab === "camera"} onClick={() => setTab("camera")}>Camera homography</button>
      </div>
      {tab === "zones" ? <Zones site={site} onSaved={onSaved} /> : <Homography site={site} onSaved={onSaved} />}
    </div>
  );
}
