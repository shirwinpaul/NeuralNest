/** Hard caps per lab so models stay small enough for modest devices. */
export interface LabLimits {
  maxLayers: number;
  maxUnits: number;
  maxFilters: number;
  maxParams: number;
  maxEpochs: number;
  batchSizes: readonly number[];
  minLearningRate: number;
  maxLearningRate: number;
  maxL2: number;
  maxDropout: number;
}

const BATCHES = [1, 4, 8, 10, 16, 32, 64, 128] as const;

export const LIMITS = {
  playground: {
    maxLayers: 5, // up to 4 hidden + output
    maxUnits: 8,
    maxFilters: 0,
    maxParams: 1000,
    maxEpochs: 2000,
    batchSizes: BATCHES,
    minLearningRate: 0.0001,
    maxLearningRate: 3,
    maxL2: 0.1,
    maxDropout: 0.5,
  },
  data: {
    maxLayers: 6,
    maxUnits: 128,
    maxFilters: 0,
    maxParams: 100_000,
    maxEpochs: 500,
    batchSizes: BATCHES,
    minLearningRate: 0.0001,
    maxLearningRate: 1,
    maxL2: 0.1,
    maxDropout: 0.6,
  },
  doodle: {
    maxLayers: 10,
    maxUnits: 128,
    maxFilters: 32,
    maxParams: 250_000,
    maxEpochs: 60,
    batchSizes: BATCHES,
    minLearningRate: 0.0001,
    maxLearningRate: 0.1,
    maxL2: 0.05,
    maxDropout: 0.6,
  },
  image: {
    maxLayers: 10,
    maxUnits: 256,
    maxFilters: 16,
    maxParams: 400_000,
    maxEpochs: 60,
    batchSizes: BATCHES,
    minLearningRate: 0.0001,
    maxLearningRate: 0.1,
    maxL2: 0.05,
    maxDropout: 0.6,
  },
} as const satisfies Record<string, LabLimits>;

export type LabKey = keyof typeof LIMITS;

/** Data / capture caps. */
export const DATA_LIMITS = {
  csvMaxBytes: 5 * 1024 * 1024,
  csvMaxRows: 50_000,
  csvMaxCols: 50,
  maxCategories: 20,
  playgroundMaxPoints: 1000,
  doodleMinClasses: 2,
  doodleMaxClasses: 5,
  doodleMaxPerClass: 50,
  imageMaxClasses: 5,
  imageMaxPerClass: 60,
  mnistMaxImages: 5000,
} as const;
