import { useMemo } from "react";
import type { Deadzones, Floor, Heatmap } from "../api/types";
import { fmtNum } from "../lib/grid";
import { FloorMap } from "./FloorMap";
import { Legend, Preview, Skeleton } from "./bits";

/** Keep probabilities from the X11 injection (camera keeps p of the observations in the blind patch). */
export const KEEPS = [1, 0.15, 0.05, 0.01] as const;

interface Props {
  floor: Floor;
  heat: Heatmap | null;
  keep: number;
  onKeep: (k: number) => void;
  dz: Deadzones | null;
  preview: boolean;
  milestone?: string;
}

/** The 30-second D-1 proof: degrade the camera over one patch; the decile rule turns it "dead",
 *  the coverage-aware rule shades it as unseen. Counts come from the endpoint (results/atc_deadzone.json). */
export function CoverageSlider({ floor, heat, keep, onKeep, dz, preview, milestone }: Props) {
  const idx = Math.max(0, KEEPS.indexOf(keep as (typeof KEEPS)[number]));
  const counts = useMemo(() => {
    if (!dz) return null;
    let decile = 0, dead = 0, unseen = 0;
    for (let i = 0; i < dz.state.length; i++) {
      if (dz.decile[i]) decile++;
      if (dz.state[i] === "dead") dead++;
      if (dz.state[i] === "unseen") unseen++;
    }
    return { decile, dead, unseen };
  }, [dz]);

  return (
    <div className="coverage">
      {preview && <Preview milestone={milestone}>The blind patch below is fake geometry so the slider can be tried. Pooled counts appear only when <code>/api/deadzones</code> serves <code>results/atc_deadzone.json</code>.</Preview>}
      <div className="slider-row">
        <label htmlFor="keep" className="slider-label">Camera quality over the test patch</label>
        <input id="keep" type="range" min={0} max={KEEPS.length - 1} step={1} value={idx}
          aria-valuetext={`${Math.round(KEEPS[idx] * 100)}% of people seen`}
          onChange={(e) => onKeep(KEEPS[Number(e.target.value)])} />
        <output htmlFor="keep" className="num slider-out">{Math.round(KEEPS[idx] * 100)}% seen</output>
        <div className="ticks" aria-hidden="true">{KEEPS.map((k) => <span key={k}>{Math.round(k * 100)}%</span>)}</div>
      </div>

      <div className="twin">
        <figure>
          <figcaption><strong>Bottom-decile rule</strong><span>what most dashboards do</span></figcaption>
          {dz ? <FloorMap floor={floor} heat={heat} dz={dz} variant="decile" maxHeight={300} label="Decile rule map" /> : <Skeleton h={220} />}
          <Legend variant="decile" />
          <p className="num stat">{counts ? `${fmtNum(counts.decile)} cells called dead` : " "}</p>
        </figure>
        <figure>
          <figcaption><strong>AisleSense, coverage-aware</strong><span>unseen floor is shaded, never called dead</span></figcaption>
          {dz ? <FloorMap floor={floor} heat={heat} dz={dz} variant="aware" maxHeight={300} label="Coverage-aware map" /> : <Skeleton h={220} />}
          <ul className="legend"><li><i className="sw sw-unseen" />Unseen</li><li><i className="sw sw-dead" />Dead</li><li><i className="sw sw-watch" />Watch</li></ul>
          <p className="num stat">{counts ? `${fmtNum(counts.dead)} dead · ${fmtNum(counts.unseen)} unseen` : " "}</p>
        </figure>
      </div>

      <div className="table-scroll"><table className="pooled">
        <caption>Pooled over held-out ATC days and seeds{dz?.source ? <> &middot; <code>{dz.source}</code></> : null}</caption>
        <thead><tr><th scope="col">Patch</th><th scope="col" className="r">Cells</th><th scope="col" className="r">Decile calls dead</th><th scope="col" className="r">Ours calls dead</th><th scope="col" className="r">Ours abstains</th></tr></thead>
        <tbody>
          {(["blind", "ignored"] as const).map((arm) => {
            const c = dz?.pooled?.[arm];
            const pct = (v?: number) => (c && v !== undefined && c.n > 0 ? `${fmtNum(v)} (${((100 * v) / c.n).toFixed(1)}%)` : "—");
            return (
              <tr key={arm}>
                <th scope="row">{arm === "blind" ? "Blind (camera degraded)" : "Ignored (people avoid it)"}</th>
                <td className="num r">{c ? fmtNum(c.n) : "—"}</td>
                <td className="num r">{pct(c?.decile_dead)}</td>
                <td className="num r">{pct(c?.aware_dead)}</td>
                <td className="num r">{pct(c?.aware_unknown)}</td>
              </tr>
            );
          })}
        </tbody>
      </table></div>
    </div>
  );
}
