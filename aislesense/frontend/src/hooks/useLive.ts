import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api/client";
import type { LiveMsg } from "../api/types";

export type LinkState = "connecting" | "open" | "down";

/** /ws/live with backoff reconnect. Also tallies section-to-section transitions per track
 *  (used as the honest browser-side Sankey preview until /api/journeys lands). */
export function useLive() {
  const [msg, setMsg] = useState<LiveMsg | null>(null);
  const [link, setLink] = useState<LinkState>("connecting");
  const [lastAt, setLastAt] = useState<number>(0);
  const transitions = useRef(new Map<string, number>());
  const lastZone = useRef(new Map<string, string>());
  const sessionRef = useRef<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let stop = false;
    let retry = 500;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      if (stop || typeof WebSocket === "undefined") return;
      setLink("connecting");
      ws = new WebSocket(wsUrl());
      ws.onopen = () => { setLink("open"); retry = 500; };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data) as LiveMsg;
        if (m.session !== sessionRef.current) {
          sessionRef.current = m.session;
          transitions.current.clear();
          lastZone.current.clear();
        }
        for (const tr of m.tracks) {
          if (!tr.zone) continue;
          const prev = lastZone.current.get(tr.id);
          if (prev && prev !== tr.zone) {
            const k = `${prev}→${tr.zone}`;
            transitions.current.set(k, (transitions.current.get(k) ?? 0) + 1);
          }
          lastZone.current.set(tr.id, tr.zone);
        }
        setMsg(m);
        setLastAt(Date.now());
        setTick((x) => x + 1);
      };
      ws.onclose = () => {
        setLink("down");
        if (!stop) { timer = setTimeout(connect, retry); retry = Math.min(retry * 2, 8000); }
      };
    };
    connect();
    return () => { stop = true; clearTimeout(timer); ws?.close(); };
  }, []);

  return { msg, link, lastAt, transitions: transitions.current, tick };
}
