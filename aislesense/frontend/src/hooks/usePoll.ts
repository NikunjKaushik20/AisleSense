import { useEffect, useRef, useState } from "react";

/** Calls `fn` now and every `ms` while `enabled`; keeps the last good value through transient errors. */
export function usePoll<T>(fn: () => Promise<T>, ms: number, deps: unknown[], enabled = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) { setLoading(false); return; }
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setLoading(true);
    const run = async () => {
      try {
        const v = await fnRef.current();
        if (alive) { setData(v); setError(null); }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) { setLoading(false); if (ms > 0) timer = setTimeout(run, ms); }
      }
    };
    run();
    return () => { alive = false; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, ms]);

  return { data, error, loading };
}
