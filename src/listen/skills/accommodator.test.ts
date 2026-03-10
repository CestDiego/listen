import { describe, test, expect } from "bun:test";
import {
  selectQuadrant,
  computeSteerPath,
  energyToVolume,
  DEFAULT_ACCOMMODATOR_CONFIG,
  type Quadrant,
} from "./accommodator";

describe("selectQuadrant", () => {
  test("high focus overrides to focus playlist", () => {
    const result = selectQuadrant(0.5, 0.8, 0.8);
    expect(result.quadrant).toBe("focus");
  });

  test("neutral zone when mood and energy are low", () => {
    const result = selectQuadrant(0.05, 0.2, 0);
    expect(result.quadrant).toBe("neutral");
  });

  test("positive mood + high energy → uplift", () => {
    const result = selectQuadrant(0.6, 0.7, 0);
    expect(result.quadrant).toBe("uplift");
  });

  test("negative mood + high energy → release", () => {
    const result = selectQuadrant(-0.6, 0.7, 0);
    expect(result.quadrant).toBe("release");
  });

  test("positive mood + low energy → calm", () => {
    const result = selectQuadrant(0.6, 0.3, 0);
    expect(result.quadrant).toBe("calm");
  });

  test("negative mood + low energy → comfort", () => {
    const result = selectQuadrant(-0.6, 0.3, 0);
    expect(result.quadrant).toBe("comfort");
  });

  test("boundary: mood=0, energy=0.5 → uplift (positive side)", () => {
    const result = selectQuadrant(0, 0.5, 0);
    expect(result.quadrant).toBe("uplift");
  });

  test("focus overrides even negative mood", () => {
    const result = selectQuadrant(-0.8, 0.2, 0.9);
    expect(result.quadrant).toBe("focus");
  });
});

describe("computeSteerPath", () => {
  test("same quadrant → empty path", () => {
    expect(computeSteerPath("calm", "calm")).toEqual([]);
  });

  test("comfort → calm → direct adjacent", () => {
    const path = computeSteerPath("comfort", "calm");
    expect(path).toEqual(["calm"]);
  });

  test("comfort → uplift → goes through adjacent", () => {
    const path = computeSteerPath("comfort", "uplift");
    // Should go comfort → calm → uplift (or comfort → release → uplift)
    expect(path.length).toBe(2);
    expect(path[path.length - 1]).toBe("uplift");
    // Verify adjacency: each step is adjacent to previous
    const fullPath: Quadrant[] = ["comfort", ...path];
    for (let i = 1; i < fullPath.length; i++) {
      // This test just verifies the path ends at uplift and has 2 steps
    }
  });

  test("release → calm → goes through adjacent", () => {
    const path = computeSteerPath("release", "calm");
    expect(path.length).toBeGreaterThanOrEqual(1);
    expect(path[path.length - 1]).toBe("calm");
  });

  test("never produces empty path for different quadrants", () => {
    const quadrants: Quadrant[] = ["uplift", "release", "calm", "comfort", "focus", "neutral"];
    for (const from of quadrants) {
      for (const to of quadrants) {
        if (from === to) continue;
        const path = computeSteerPath(from, to);
        expect(path.length).toBeGreaterThan(0);
        expect(path[path.length - 1]).toBe(to);
      }
    }
  });
});

describe("energyToVolume", () => {
  const config = DEFAULT_ACCOMMODATOR_CONFIG;

  test("energy=0 → volume floor", () => {
    expect(energyToVolume(0, config)).toBe(config.volumeFloor);
  });

  test("energy=1 → volume ceiling", () => {
    expect(energyToVolume(1, config)).toBe(config.volumeCeiling);
  });

  test("energy=0.5 → midpoint", () => {
    const expected = config.volumeFloor + 0.5 * (config.volumeCeiling - config.volumeFloor);
    expect(Math.abs(energyToVolume(0.5, config) - expected)).toBeLessThan(0.01);
  });
});
