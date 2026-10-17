import { useEffect, useRef, useState } from "react";

const EVERY_MS = 300;

/** What the detector sees right now: newest frame of one camera, heads pixelated on the server, boxes and track ids
 *  drawn. Fetched as a blob and swapped only when the next frame has loaded, so it never flickers. Never saved. */
export function CameraTile({ cams }: { cams: { id: string }[] }) {
  const [cam, setCam] = useState(cams[0]?.id ?? "");
  const [src, setSrc] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(true);
  const last = useRef<string | null>(null);

  useEffect(() => { if (!cams.some((c) => c.id === cam)) setCam(cams[0]?.id ?? ""); }, [cams, cam]);

  useEffect(() => {
    let alive = true, timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetch(`/api/camera?cam=${encodeURIComponent(cam)}`, { cache: "no-store" });
        if (!alive) return;
        if (r.status === 200) {
          const url = URL.createObjectURL(await r.blob());
          if (!alive) { URL.revokeObjectURL(url); return; }
          if (last.current) URL.revokeObjectURL(last.current);
          last.current = url;
          setSrc(url);
          setWaiting(false);
        } else setWaiting(true);
      } catch { if (alive) setWaiting(true); }
      if (alive) timer = setTimeout(tick, EVERY_MS);
    };
    tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [cam]);

  useEffect(() => () => { if (last.current) URL.revokeObjectURL(last.current); }, []);

  return (
    <div className="camera">
      {cams.length > 1 && (
        <div className="seg" role="group" aria-label="Camera">
          {cams.map((c) => <button key={c.id} type="button" aria-pressed={cam === c.id} onClick={() => setCam(c.id)}>{c.id}</button>)}
        </div>
      )}
      <div className="camera-frame">
        {src && <img src={src} alt={`Live view of camera ${cam}: people boxed with their track numbers, heads pixelated`} />}
        {(waiting || !src) && <p className="camera-wait">{src ? "Stream paused" : "Waiting for the first frame…"}</p>}
      </div>
      <p className="hint">YOLO11n boxes and ByteTrack ids, live. Heads are pixelated and frames are never saved.</p>
    </div>
  );
}
