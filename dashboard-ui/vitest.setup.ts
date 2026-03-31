import "@testing-library/jest-dom/vitest";

// Polyfill ResizeObserver for jsdom (used by Lightweight Charts)
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
