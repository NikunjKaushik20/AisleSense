import type { Dwell } from "../api/types";
import { fmtDuration, fmtNum } from "../lib/grid";
import { Preview, Skeleton } from "./bits";

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
            <th scope="col" className="r">Person-seconds</th>
            <th scope="col" className="r">Entries</th>
            <th scope="col" className="r">Dwell / entry</th>
            <th scope="col" className="r" data-testid="prior-col" title={dwell.prior.source}>{dwell.prior.name}</th>
            <th scope="col" className="r">Long stays</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={6} className="empty">This site has no sections yet. Draw them in Setup.</td></tr>}
          {rows.map((r) => {
            const delta = r.dwell_s !== null && r.prior_s !== null && r.prior_s > 0 ? r.dwell_s / r.prior_s - 1 : null;
            return (
              <tr key={r.section}>
                <th scope="row">{r.section}</th>
                <td className="num r bar-cell"><span className="bar" style={{ width: `${(100 * r.person_s) / maxPs}%` }} />{fmtNum(r.person_s)}</td>
                <td className="num r">{r.entries ? fmtNum(r.entries) : "—"}</td>
                <td className="num r">{fmtDuration(r.dwell_s)}{delta !== null && <span className={`delta ${delta >= 0 ? "up" : "down"}`}> {delta >= 0 ? "+" : ""}{Math.round(delta * 100)}%</span>}</td>
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
            <li key={`${l.track}${l.t_in}`}><span>{l.section}</span><span className="num">{fmtDuration(l.seconds)}</span><span className="num muted">track {l.track}</span></li>
          ))}</ol>}
    </div>
  );
}
