import { useEffect, useState } from "react";
import { api } from "../api/client";
import { mockDiff } from "../api/mocks";
import type { Diff, DiffItem, Floor, Heatmap, StoredSession } from "../api/types";
import { FloorMap } from "./FloorMap";
import { Preview, Skeleton } from "./bits";

/** Two stored sessions of the same site, side by side, plus the flags that cleared between them (diff.py, M4). */
export function BeforeAfter({ siteId, floor, sessions }: { siteId: string; floor: Floor; sessions: StoredSession[] }) {
  const mine = sessions.filter((s) => s.site === siteId);
  const [a, setA] = useState<string>("");
  const [b, setB] = useState<string>("");
  const [maps, setMaps] = useState<{ a: Heatmap | null; b: Heatmap | null }>({ a: null, b: null });
  const [diff, setDiff] = useState<{ data: Diff; preview: boolean; milestone?: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (mine.length >= 2 && (!a || !b)) { setA(mine[mine.length - 2].id); setB(mine[mine.length - 1].id); }
  }, [mine.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!a || !b) return;
    let alive = true;
    setErr(null);
    // maps first: the cleared / new / still-there lists can take a moment longer on an old session
    Promise.all([api.heatmap(a, 0, 1e12), api.heatmap(b, 0, 1e12)])
      .then(([ha, hb]) => { if (alive) setMaps({ a: ha, b: hb }); })
      .catch((e) => alive && setErr(String(e.message ?? e)));
    api.diff(a, b, () => mockDiff(a, b))
      .then((d) => { if (alive) setDiff(d); })
      .catch((e) => alive && setErr(String(e.message ?? e)));
    return () => { alive = false; };
  }, [a, b]);

  if (mine.length < 2) return (
    <p className="empty">Before / after needs two recorded sessions of this site. Run one, change the floor (move the cart, add the camera), run another.</p>
  );
  const fmt = (s: StoredSession) =>
    `${new Date(s.started * 1000).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })} \u00b7 ${s.mode}`;
  const col = (title: string, val: string, set: (v: string) => void, h: Heatmap | null) => (
    <figure>
      <figcaption>
        <label className="field"><span>{title}</span>
          <select value={val} onChange={(e) => set(e.target.value)}>{mine.map((s) => <option key={s.id} value={s.id}>{fmt(s)}</option>)}</select>
        </label>
      </figcaption>
      {h ? <FloorMap floor={floor} heat={h} variant="aware" maxHeight={280} label={`${title} heatmap`} /> : <Skeleton h={200} />}
    </figure>
  );
  const items = (label: string, xs: DiffItem[], cls: string) => (
    <div className={`diff-col ${cls}`}><h3 className="sub">{label} <span className="num muted">{xs.length}</span></h3>
      {xs.length ? <ul>{xs.map((x, i) => (
        <li key={i} title={x.text}><span className={`kind kind-${x.kind.toLowerCase()}`}>{x.kind}</span>{x.zone ?? "open floor"}</li>
      ))}</ul> : <p className="empty">None</p>}
    </div>
  );
  return (
    <div className="beforeafter">
      {err && <p className="error" role="alert">Could not load sessions: {err}</p>}
      {diff?.preview && <Preview milestone={diff.milestone}>Heatmaps are real (stored events). The list of cleared flags arrives with {diff.milestone}.</Preview>}
      <div className="twin">{col("Before", a, setA, maps.a)}{col("After", b, setB, maps.b)}</div>
      {diff && <p className="hint">Hover a line for the full reason.</p>}
      {diff && <div className="diff">{items("Cleared", diff.data.cleared, "cleared")}{items("New", diff.data.added, "added")}{items("Still there", diff.data.kept, "kept")}</div>}
    </div>
  );
}
