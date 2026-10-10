import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../src/api/client";
import { mockDeadzones, mockDwell } from "../src/api/mocks";
import type { Deadzones, Floor, Heatmap } from "../src/api/types";
import { AtcBanner, FallbackBanner } from "../src/components/bits";
import { CoverageSlider } from "../src/components/CoverageSlider";
import { DwellTable } from "../src/components/DwellTable";
import { FloorMap } from "../src/components/FloorMap";
import { useState } from "react";

const floor: Floor = { x0: 0, y0: 0, width: 4, height: 2, cell: 1 };
const heat: Heatmap = {
  session: "s", site: "x", t0: null, t1: null, unit: "person-seconds", x0: 0, y0: 0, cell: 1, nx: 4, ny: 2,
  cells: [1, 2, 3, 4, 5, 6, 7, 8], max: 8, births: Array(8).fill(0), deaths: Array(8).fill(0),
};
const dz: Deadzones = {
  session: "s", keep: 0.01, x0: 0, y0: 0, cell: 1, nx: 4, ny: 2,
  state: ["ok", "unseen", "unseen", "dead", "watch", "ok", "ok", "unknown"],
  decile: [false, true, true, false, false, false, false, false],
  coverage: [1, 0.01, 0.01, 1, 1, 1, 1, 1], pooled: null, source: null,
};

afterEach(() => vi.restoreAllMocks());

describe("FloorMap", () => {
  it("shades UNSEEN cells with the hatch and outlines DEAD/WATCH", () => {
    const { container } = render(<FloorMap floor={floor} heat={heat} dz={dz} />);
    const unseen = container.querySelectorAll('[data-state="unseen"]');
    expect(unseen).toHaveLength(2);
    unseen.forEach((el) => expect(el.getAttribute("fill")).toBe("url(#hatch)"));
    expect(container.querySelectorAll('[data-state="dead"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-state="watch"]')).toHaveLength(1);
  });

  it("decile variant marks what the decile rule calls dead, and never hatches", () => {
    const { container } = render(<FloorMap floor={floor} heat={heat} dz={dz} variant="decile" />);
    expect(container.querySelectorAll('[data-state="decile"]')).toHaveLength(2);
    expect(container.querySelectorAll('[data-state="unseen"]')).toHaveLength(0);
  });

  it("an UNSEEN cell is never drawn as DEAD", () => {
    const m = mockDeadzones(floor, heat, 0.01, "s");
    m.state.forEach((s, i) => { if (m.coverage[i] < 0.25) expect(s).toBe("unseen"); });
  });
});

describe("Camera-quality slider", () => {
  it("calls /api/deadzones?keep= with the slider value", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "not_implemented" }), { status: 501 }));
    function Harness() {
      const [keep, setKeep] = useState(1);
      const [d, setD] = useState<Deadzones | null>(null);
      return (
        <>
          <button onClick={() => api.deadzones("s1", keep, () => mockDeadzones(floor, heat, keep, "s1")).then((r) => setD(r.data))}>load</button>
          <CoverageSlider floor={floor} heat={heat} keep={keep} onKeep={(k) => { setKeep(k); api.deadzones("s1", k, () => mockDeadzones(floor, heat, k, "s1")).then((r) => setD(r.data)); }} dz={d} preview />
        </>
      );
    }
    render(<Harness />);
    fireEvent.change(screen.getByLabelText(/camera quality/i), { target: { value: "3" } });
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const url = String(fetchMock.mock.calls.at(-1)![0]);
    expect(url).toContain("/api/deadzones?");
    expect(url).toContain("keep=0.01");
    expect(url).toContain("session=s1");
    expect(await screen.findByText(/1% seen/)).toBeInTheDocument();
  });

  it("shows pooled counts only from the endpoint payload", () => {
    const withPooled: Deadzones = { ...dz, source: "results/atc_deadzone.json",
      pooled: { blind: { n: 656, aware_dead: 0, aware_unknown: 616, decile_dead: 416 }, ignored: { n: 10, aware_dead: 6, aware_unknown: 1, decile_dead: 7 } } };
    render(<CoverageSlider floor={floor} heat={heat} keep={0.01} onKeep={() => {}} dz={withPooled} preview={false} />);
    expect(screen.getByText("656")).toBeInTheDocument();
    expect(screen.getByText("0 (0.0%)")).toBeInTheDocument();
    expect(screen.getByText("results/atc_deadzone.json")).toBeInTheDocument();
  });
});

describe("Dwell table", () => {
  it("always renders the prior column, even with placeholder data", () => {
    const site = { id: "x", name: "x", kind: "synth" as const, label: null, floor, cameras: [],
      zones: [{ name: "Snacks", kind: "section" as const, polygon: [[0, 0], [2, 0], [2, 2], [0, 2]] as [number, number][] }] };
    render(<DwellTable dwell={mockDwell(site, heat, "s")} preview milestone="M3" />);
    expect(screen.getByTestId("prior-col")).toBeInTheDocument();
    expect(screen.getByText("Snacks")).toBeInTheDocument();
    // person-seconds from the real heatmap: cells (0,0) (1,0) (0,1) (1,1) = 1+2+5+6
    expect(screen.getByText("14")).toBeInTheDocument();
  });

  it("renders the prior column with an empty section list", () => {
    render(<DwellTable dwell={{ session: "s", unit: "u", prior: { name: "Prior", source: "x" }, sections: [], long_stays: [] }} preview={false} />);
    expect(screen.getByTestId("prior-col")).toHaveTextContent("Prior");
  });
});

describe("ATC banner", () => {
  it("is present on ATC and says it is a trajectory replay", () => {
    render(<AtcBanner kind="atc" label={null} />);
    expect(screen.getByTestId("atc-banner")).toHaveTextContent(/trajectory replay, not our video/i);
  });
  it("is absent on other sites", () => {
    render(<AtcBanner kind="synth" label={null} />);
    expect(screen.queryByTestId("atc-banner")).toBeNull();
  });
});

describe("contract client", () => {
  it("turns 501 into a labelled preview and passes 200 through", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "not_implemented" }), { status: 501 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ session: "s", tips: [] }), { status: 200 }));
    const a = await api.tips("s", () => ({ session: "s", tips: [] }));
    expect(a.preview).toBe(true);
    expect(a.milestone).toBe("M4");
    const b = await api.tips("s", () => ({ session: "mock", tips: [] }));
    expect(b.preview).toBe(false);
  });
});

describe("FallbackBanner (M6, F13)", () => {
  it("shows the replay banner only while a stream is down", () => {
    const { rerender } = render(<FallbackBanner text="Live stream cam1 dropped: showing a replay" />);
    expect(screen.getByTestId("fallback-banner")).toHaveTextContent(/replay fallback/i);
    rerender(<FallbackBanner text={null} />);
    expect(screen.queryByTestId("fallback-banner")).toBeNull();
  });
});
