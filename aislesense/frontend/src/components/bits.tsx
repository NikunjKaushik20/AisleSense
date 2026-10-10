import type { ReactNode } from "react";
import type { SiteKind } from "../api/types";

/** Shown on any panel whose endpoint still answers 501: the data under it is a labelled placeholder. */
export function Preview({ milestone, children }: { milestone?: string; children?: ReactNode }) {
  return (
    <p className="preview" role="note">
      <strong>Coming in {milestone ?? "a later milestone"}.</strong>{" "}
      {children ?? "The endpoint answers 501 today; this panel shows placeholder data so the layout can be checked."}
    </p>
  );
}

/** ATC is a public dataset of sensor trajectories; it must never pass for our camera pipeline [D10]. */
export function AtcBanner({ kind, label }: { kind: SiteKind | undefined; label?: string | null }) {
  if (kind !== "atc") return null;
  return (
    <div className="banner" role="status" data-testid="atc-banner">
      <strong>Trajectory replay, not our video.</strong>{" "}
      {label ?? "Positions come from the ATC mall dataset's own sensors; no camera or detector is involved."}
    </div>
  );
}

/** M6 (F13): a live phone stream dropped for > 3 s; the map shows a replay of its recorded rows until it is back. */
export function FallbackBanner({ text }: { text?: string | null }) {
  if (!text) return null;
  return (
    <div className="banner banner-warn" role="alert" data-testid="fallback-banner">
      <strong>Replay fallback.</strong> {text}
    </div>
  );
}

export function SynthBanner({ kind, label }: { kind: SiteKind | undefined; label?: string | null }) {
  if (kind !== "synth") return null;
  return <div className="banner banner-quiet" role="status"><strong>Synthetic scenario.</strong> {label}</div>;
}

export function Panel({ title, aside, children, id }: { title: string; aside?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="panel" aria-labelledby={id ? `${id}-h` : undefined}>
      <header className="panel-head">
        <h2 id={id ? `${id}-h` : undefined}>{title}</h2>
        {aside}
      </header>
      {children}
    </section>
  );
}

export function Skeleton({ h = 120 }: { h?: number }) {
  return <div className="skeleton" style={{ height: h }} aria-busy="true" aria-label="Loading" />;
}

export function Legend({ variant = "full" }: { variant?: "full" | "decile" }) {
  if (variant === "decile") return (
    <ul className="legend"><li><i className="sw sw-decile" />Called dead by bottom-decile rule</li></ul>
  );
  return (
    <ul className="legend" aria-label="Map legend">
      <li><i className="sw sw-ramp" />Person-seconds</li>
      <li><i className="sw sw-browse" />Browse</li>
      <li><i className="sw sw-block" />Block</li>
      <li><i className="sw sw-unseen" />Unseen (camera can't tell)</li>
      <li><i className="sw sw-dead" />Dead</li>
      <li><i className="sw sw-watch" />Watch</li>
    </ul>
  );
}
