import type { Dwell } from "../api/types";
import { fmtDuration, fmtNum } from "../lib/grid";

/** Media time as m:ss (h:mm:ss past an hour), for "when did this long stay start". */
function clock(t: number): string {
  const s = Math.max(0, Math.round(t)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
}
import { Preview, Skeleton } from "./bits";

/** Person-seconds as a readable total ("1 h 12 min", "3 min 20 s"). The raw value is in the cell's title. */
function fmtTotal(ps: number): string {
  if (ps < 60) return `${Math.round(ps)} s`;
  if (ps < 3600) return fmtDuration(ps).replace(/ 0 s$/, "");
  const h = Math.floor(ps / 3600);
  return `${h} h ${Math.round((ps - h * 3600) / 60)} min`;
}

/** Section dwell. The prior column is always rendered, even when empty: every number sits beside its baseline. */
export function DwellTable({ dwell, preview, milestone }: { dwell: Dwell | null; preview: boolean; milestone?: string }) {
  if (!dwell) return <Skeleton h={180} />;
  const rows = [...dwell.sections].sort((a, b) => b.person_s - a.person_s);
  const maxPs = Math.max(1, ...rows.map((r) => r.person_s));
  return (
    <div className="dwell">
      {preview && <Preview milestone={milestone}>Person-seconds are real (summed from the live heatmap). Entries, dwell per entry and the prior arrive with {milestone}.</Preview>}
      <div className="table-scroll"><table className="data" data-testid="dwell-table">
        <thead>
          <tr>
            <th scope="col">Section</th>
            <th scope="col" className="r" title="Total time spent in the section by everyone, in person-seconds">Total time</th>
            <th scope="col" className="r" title="How many times someone walked into the section">Visits</th>
            <th scope="col" className="r" title="Average time per visit, compared with the hand-timed average">Avg stay</th>
            <th scope="col" className="r" data-testid="prior-col" title={dwell.prior.source ?? undefined}>{dwell.prior.name}</th>
            <th scope="col" className="r" title="Visits well past the section's usual time">Long stays</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={6} className="empty">This site has no sections yet. Draw them in Setup.</td></tr>}
          {rows.map((r) => {
            const delta = r.dwell_s !== null && r.prior_s !== null && r.prior_s > 0 ? r.dwell_s / r.prior_s - 1 : null;
            return (
              <tr key={r.section}>
                <th scope="row">{r.section}</th>
                <td className="num r bar-cell"><span className="bar" style={{ width: `${(100 * r.person_s) / maxPs}%` }} />{fmtTotal(r.person_s)}</td>
                <td className="num r">{r.entries ? fmtNum(r.entries) : "—"}</td>
                <td className="num r stay">
                  <span>{fmtDuration(r.dwell_s)}</span>
                  {delta !== null && <span className={`delta ${delta >= 0 ? "up" : "down"}`}>{delta >= 0 ? "+" : "−"}{Math.abs(Math.round(delta * 100))}% vs hand-timed</span>}
                </td>
                <td className="num r prior">{fmtDuration(r.prior_s)}</td>
                <td className="num r">{r.n_long ? fmtNum(r.n_long) : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table></div>
      <h3 className="sub">Long stays</h3>
      {dwell.long_stays.length === 0
        ? <p className="empty">No long stays yet. People who stay well past their section's usual time are listed here.</p>
        : <ol className="longs">{dwell.long_stays.slice(0, 8).map((l) => (
            <li key={`${l.track}${l.t_in}`}><span>{l.section}</span><span className="num">{fmtDuration(l.seconds)}</span><span className="num muted">from {clock(l.t_in)}</span></li>
          ))}</ol>}
    </div>
  );
}
