import type { FeatureFunction, FeatureLayer, FeatureQuery, FeatureResult } from "./interfaces.js";

export class InMemoryFeatureLayer implements FeatureLayer {
  private readonly registry = new Map<string, FeatureFunction>();

  async register(feature: FeatureFunction): Promise<void> {
    this.registry.set(feature.id, feature);
  }

  async execute(input: FeatureQuery): Promise<FeatureResult> {
    const feature = this.registry.get(input.featureName);
    if (!feature) {
      throw new Error(`feature not found: ${input.featureName}`);
    }
    return feature.execute(input);
  }

  list(): FeatureFunction[] {
    return [...this.registry.values()];
  }
}
