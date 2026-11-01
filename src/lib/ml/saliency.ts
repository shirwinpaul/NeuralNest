/**
 * Vanilla gradient saliency: how much does the predicted class score change when
 * each input pixel changes a little? |∂ score / ∂ pixel|, normalised to [0, 1].
 */
import * as tf from "@tensorflow/tfjs";

export function saliencyMap(model: tf.LayersModel, input: Float32Array, shape: [number, number, number], classIndex: number): Float32Array {
  return tf.tidy(() => {
    const x = tf.tensor4d(input, [1, ...shape]);
    // Use the pre-softmax-like signal: the class probability's gradient.
    const f = (inp: tf.Tensor) => (model.predict(inp) as tf.Tensor2D).gather([classIndex], 1).sum();
    const g = tf.grad(f)(x);
    const abs = g.abs().max(-1).reshape([shape[0] * shape[1]]); // max over channels
    const mx = abs.max();
    const norm = abs.div(tf.maximum(mx, tf.scalar(1e-12)));
    return norm.dataSync() as Float32Array;
  });
}
