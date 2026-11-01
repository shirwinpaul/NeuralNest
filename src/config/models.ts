/** Pre-trained model used by the Image Teacher (lazy-loaded only in that lab). */
export const MOBILENET = {
  name: "MobileNet v2 (α = 0.5, 224×224)",
  licence: "Apache-2.0 (Google)",
  // TF Hub path (redirects to Kaggle Models); the query string is kept for the weight files too.
  url: "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_050_224/classification/3/default/1/model.json?tfjs-format=file",
  approxDownloadMB: 8,
  inputSize: 224,
} as const;
