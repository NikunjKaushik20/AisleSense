import { useMemo } from "react";
import { sankey, sankeyLinkHorizontal, type SankeyGraph } from "d3-sankey";
import type { Journeys } from "../api/types";
import { Preview, Skeleton } from "./bits";

type N = { name: string };
type L = { source: number; target: number; value: number };

/** d3-sankey needs a DAG and journeys loop (A->B->A). Lay them out bipartite instead:
 *  every section appears once as "from" (left) and once as "to" (right), so direction is kept exactly. */
function bipartite(j: Journeys): { nodes: N[]; links: L[] } {
  const n = j.nodes.length;
  const links = j.links.filter((l) => l.source !== l.target && l.value > 0)
    .map((l) => ({ source: l.source, target: n + l.target, value: l.value }));
  return { nodes: [...j.nodes.map((x) => ({ ...x })), ...j.nodes.map((x) => ({ ...x }))], links };
}

export function Sankey({ journeys, preview, milestone, width = 640 }: { journeys: Journeys | null; preview: boolean; milestone?: string; width?: number }) {
  const layout = useMemo(() => {
    if (!journeys || journeys.links.length === 0) return null;
    const g = bipartite(journeys);
    if (!g.links.length) return null;
    const gen = sankey<N, L>().nodeWidth(10).nodePadding(14).extent([[1, 8], [width - 1, 272]]);
    try {
      return gen(g as SankeyGraph<N, L>);
    } catch { return null; }
  }, [journeys, width]);

  return (
    <div className="sankey">
      {preview && <Preview milestone={milestone}>Until then this is tallied in your browser from the live feed since the page opened (section changes per track).</Preview>}
      {!journeys ? <Skeleton h={240} /> : !layout ? (
        <p className="empty">No section-to-section moves yet. Journeys appear once people walk between sections.</p>
      ) : (
        <>
          <svg viewBox={`0 0 ${width} 280`} width="100%" role="img" aria-label="Journeys between sections">
            {layout.links.map((l, i) => (
              <path key={i} d={sankeyLinkHorizontal()(l) ?? undefined} className="sk-link" strokeWidth={Math.max(1, l.width ?? 1)}>
                <title>{`${(l.source as N).name} → ${(l.target as N).name}: ${l.value}`}</title>
              </path>
            ))}
            {layout.nodes.filter((n) => (n.value ?? 0) > 0).map((n, i) => (
              <g key={i}>
                <rect className="sk-node" x={n.x0} y={n.y0} width={(n.x1 ?? 0) - (n.x0 ?? 0)} height={Math.max(1, (n.y1 ?? 0) - (n.y0 ?? 0))} />
                <text x={(n.x0 ?? 0) < width / 2 ? (n.x1 ?? 0) + 6 : (n.x0 ?? 0) - 6} y={((n.y0 ?? 0) + (n.y1 ?? 0)) / 2} dy="0.35em"
                  textAnchor={(n.x0 ?? 0) < width / 2 ? "start" : "end"}>{n.name}</text>
              </g>
            ))}
          </svg>
          <table className="data compact">
            <caption className="sr-only">Journey counts</caption>
            <thead><tr><th scope="col">From</th><th scope="col">To</th><th scope="col" className="r">Moves</th></tr></thead>
            <tbody>
              {[...journeys.links].sort((a, b) => b.value - a.value).slice(0, 6).map((l, i) => (
                <tr key={i}><td>{journeys.nodes[l.source]?.name}</td><td>{journeys.nodes[l.target]?.name}</td><td className="num r">{l.value}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
