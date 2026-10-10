import { useEffect, useState } from "react";
import type { Tips as TipsT } from "../api/types";
import { Preview, Skeleton } from "./bits";

const KIND: Record<string, string> = {
  DEAD: "Dead floor", UNSEEN: "Camera gap", WATCH: "Watch", BLOCK: "Blocked walkway", BROWSE: "Browsing", DWELL: "Dwell",
};

/** One tip at a time: a manager reads one action, not a wall of alerts. */
export function Tips({ tips, preview, milestone }: { tips: TipsT | null; preview: boolean; milestone?: string }) {
  const [i, setI] = useState(0);
  const list = tips?.tips ?? [];
  useEffect(() => { if (i >= list.length) setI(0); }, [list.length, i]);
  if (!tips) return <Skeleton h={110} />;
  const t = list[i];
  return (
    <div className="tips">
      {preview && <Preview milestone={milestone}>These explain what the tips will cover; they are not findings.</Preview>}
      {!t ? <p className="empty">Nothing to act on right now. Tips appear when a flag holds for a session.</p> : (
        <article className={`tip tip-${t.kind.toLowerCase()}`} aria-live="polite">
          <p className="tip-kind"><i className={`sw sw-${t.kind.toLowerCase()}`} aria-hidden="true" />{KIND[t.kind] ?? t.kind}{t.zone ? ` · ${t.zone}` : ""}</p>
          <p className="tip-text">{t.text}</p>
        </article>
      )}
      {list.length > 1 && (
        <div className="tip-nav">
          <button type="button" onClick={() => setI((i - 1 + list.length) % list.length)} aria-label="Previous tip">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
          </button>
          <span className="num">{i + 1} of {list.length}</span>
          <button type="button" onClick={() => setI((i + 1) % list.length)} aria-label="Next tip">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="m6 3 5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
          </button>
        </div>
      )}
    </div>
  );
}
