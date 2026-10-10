import "@testing-library/jest-dom/vitest";

// jsdom has no canvas: the map draws heat on canvas and every judged mark in SVG, so a null context is fine.
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"];
