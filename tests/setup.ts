import * as tf from "@tensorflow/tfjs";
import { beforeAll } from "vitest";

// All ML tests run on the deterministic CPU backend.
beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});
