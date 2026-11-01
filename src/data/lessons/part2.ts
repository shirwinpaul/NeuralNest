import type { Lesson } from "./types";

export const LESSONS_B: Lesson[] = [
  {
    id: "regularisation",
    title: "Regularisation & dropout",
    emoji: "🧯",
    illustration: "dropout",
    summary: "Regularisation gently stops a model from relying on extreme weights or on any single neuron.",
    sections: [
      {
        heading: "L2 regularisation",
        body: [
          "L2 adds a penalty to the loss: the sum of all squared weights times a small number λ. To keep the loss low, the model prefers small weights, which usually means smoother, simpler decision boundaries that generalise better.",
          "Too much L2 and the model becomes too simple (underfits).",
        ],
      },
      {
        heading: "Dropout",
        body: [
          "During training, dropout randomly switches off a fraction of neurons (for example 25%) on each step. The network can't depend on any one neuron, so it learns more robust features. At prediction time all neurons are used.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=regularise", label: "Add L2 in the Playground" },
    quiz: [
      { q: "L2 regularisation encourages…", options: ["Large weights", "Small weights", "More layers"], answer: 1, why: "It penalises the squared size of weights." },
      { q: "Dropout randomly…", options: ["Deletes training data", "Turns off neurons during training", "Changes the labels"], answer: 1, why: "It disables neurons temporarily so the network can't over-rely on any." },
      { q: "Too much regularisation can cause…", options: ["Underfitting", "Leakage", "Faster overfitting"], answer: 0, why: "The model becomes too constrained." },
    ],
  },
  {
    id: "cnn",
    title: "Convolutional neural networks",
    emoji: "🖼️",
    illustration: "cnn",
    summary: "CNNs slide small filters over images to detect edges, curves and shapes wherever they appear.",
    sections: [
      {
        heading: "Filters",
        body: [
          "A convolution layer has small filters (like 3×3 grids of weights). Each filter slides across the image and lights up where it finds its pattern — an edge, a corner, a curve. Because the same filter is used everywhere, a CNN needs far fewer weights than a dense network on raw pixels.",
        ],
      },
      {
        heading: "Pooling and stacking",
        body: [
          "Max-pooling shrinks the image by keeping the strongest signal in each small area, so later layers see a wider view. Stacking conv → pool → conv → pool lets the network build from edges to parts to whole objects. A final dense layer with softmax gives the class probabilities.",
        ],
      },
    ],
    tryIt: { href: "/doodle", label: "Train a CNN on your doodles" },
    quiz: [
      { q: "A convolution filter is…", options: ["A small grid of weights slid over the image", "A photo editing effect", "A type of label"], answer: 0, why: "It computes a weighted sum at every position." },
      { q: "Max-pooling…", options: ["Makes images bigger", "Keeps the strongest value in each small area", "Adds colour"], answer: 1, why: "It downsamples while keeping strong signals." },
      { q: "Why do CNNs need fewer weights than dense nets on images?", options: ["Filters are shared across positions", "They ignore pixels", "They use no maths"], answer: 0, why: "Weight sharing is the key trick." },
    ],
  },
  {
    id: "transfer-learning",
    title: "Transfer learning",
    emoji: "🎁",
    illustration: "transfer",
    summary: "Reuse a network already trained on millions of images, and only train a small new part on your data.",
    sections: [
      {
        heading: "Standing on giants' shoulders",
        body: [
          "Training a big image model from scratch needs huge datasets. But a model like MobileNet, trained on over a million photos, has already learned general features — edges, textures, shapes, objects.",
          "In transfer learning we freeze that model, use it to turn each image into a list of numbers (an embedding), and train only a small dense \"head\" on top. With just a few dozen photos per class, it can work surprisingly well.",
        ],
      },
      {
        heading: "Compare for yourself",
        body: ["In the Image Teacher you can train a tiny CNN from scratch and a transfer-learning head on the same photos, then compare their test accuracy."],
      },
    ],
    tryIt: { href: "/image-teacher", label: "Try transfer learning" },
    quiz: [
      { q: "In transfer learning, the pre-trained part is usually…", options: ["Frozen", "Deleted", "Retrained from zero"], answer: 0, why: "We reuse its learned features." },
      { q: "Why does transfer learning need fewer examples?", options: ["The base model already learned general features", "It doesn't use data", "It guesses"], answer: 0, why: "Only the small head must be learned." },
      { q: "An embedding is…", options: ["A list of numbers describing an input", "A type of camera", "A loss function"], answer: 0, why: "It is the feature vector produced by the base model." },
    ],
  },
  {
    id: "classification-metrics",
    title: "Classification metrics",
    emoji: "🧮",
    illustration: "confusion",
    summary: "Accuracy alone can mislead. Precision, recall and the confusion matrix show what kind of mistakes happen.",
    sections: [
      {
        heading: "The confusion matrix",
        body: [
          "A confusion matrix is a table: rows are the true classes, columns the predicted classes. The diagonal shows correct answers; everything else shows exactly which classes get mixed up.",
        ],
      },
      {
        heading: "Precision vs. recall",
        body: [
          "Precision: of everything the model called \"positive\", how much really was? Recall: of all real positives, how many did the model find? F1 is a balance of both.",
          "For a disease test, missing a sick person (low recall) can be worse than a false alarm. Which metric matters depends on the real-world cost of each mistake.",
        ],
      },
      {
        heading: "Why accuracy can lie",
        body: [
          "If 95% of emails are not spam, a model that always says \"not spam\" gets 95% accuracy — and is useless. Always compare with a baseline like \"always predict the most common class\". The Data Lab does this for you.",
        ],
      },
    ],
    tryIt: { href: "/data-lab", label: "Compare with a baseline" },
    quiz: [
      { q: "95% of data is class A. A model always says A. Its accuracy is…", options: ["5%", "50%", "95%"], answer: 2, why: "That is why we compare with a majority baseline." },
      { q: "Recall answers…", options: ["Of real positives, how many were found?", "Of predicted positives, how many were right?", "How fast is the model?"], answer: 0, why: "Recall = TP / (TP + FN)." },
      { q: "In a confusion matrix, correct predictions are on the…", options: ["Diagonal", "First row", "Last column"], answer: 0, why: "True class = predicted class." },
    ],
  },
  {
    id: "regression-metrics",
    title: "Regression metrics",
    emoji: "📏",
    illustration: "regression",
    summary: "MAE, RMSE and R² tell you how far off number predictions are, in different ways.",
    sections: [
      {
        heading: "Measuring error",
        body: [
          "MAE (mean absolute error) is the average size of the mistakes, in the same units as the target — easy to explain. MSE squares the errors, so big mistakes count much more; RMSE is its square root, back in the original units.",
        ],
      },
      {
        heading: "R² — compared with guessing the average",
        body: [
          "R² compares your model with a baseline that always predicts the mean. R² = 1 is perfect, R² = 0 is no better than the mean, and negative R² means worse than just guessing the average!",
          "A residual plot (error vs. prediction) helps spot patterns your model is missing.",
        ],
      },
    ],
    tryIt: { href: "/data-lab?preset=study", label: "Predict exam scores" },
    quiz: [
      { q: "R² = 0 means the model is…", options: ["Perfect", "No better than predicting the mean", "Broken"], answer: 1, why: "R² compares against the mean predictor." },
      { q: "Which metric punishes big errors the most?", options: ["MAE", "MSE", "Accuracy"], answer: 1, why: "Squaring makes large errors dominate." },
      { q: "MAE is measured in…", options: ["The same units as the target", "Percent always", "Epochs"], answer: 0, why: "It averages absolute differences." },
    ],
  },
  {
    id: "bias-fairness",
    title: "Bias & fairness in data",
    emoji: "⚖️",
    illustration: "bias",
    summary: "A model trained on one-sided data will make one-sided mistakes — often on the people or cases left out.",
    sections: [
      {
        heading: "Where bias comes from",
        body: [
          "If all your training photos are taken in one room, with one lighting, of one person, the model may learn the room or the lighting instead of the thing you care about. When tested somewhere new, accuracy drops.",
          "Real systems have failed this way — for example face-recognition tools that worked worse for groups who were under-represented in their training data.",
        ],
      },
      {
        heading: "What to do",
        body: [
          "Collect varied, representative data. Test on different settings and groups separately, not just on average. Report limitations honestly. Ask: who could be harmed if this model is wrong?",
          "Try the bias activity in the Image Teacher to see it happen with your own camera.",
        ],
      },
    ],
    tryIt: { href: "/image-teacher?preset=bias", label: "Do the bias activity" },
    quiz: [
      { q: "A model trained only on photos from one room may learn…", options: ["The room instead of the object", "Nothing at all", "Perfect fairness"], answer: 0, why: "It can latch onto background clues." },
      { q: "A good way to find bias is to…", options: ["Test separately on different groups/settings", "Only look at total accuracy", "Use fewer test examples"], answer: 0, why: "Averages can hide groups where the model fails." },
      { q: "Under-represented groups in training data often get…", options: ["Better results", "Worse results", "Identical results always"], answer: 1, why: "The model saw fewer examples like them." },
    ],
  },
  {
    id: "model-cards",
    title: "Model cards & responsible AI",
    emoji: "🪪",
    illustration: "card",
    summary: "A model card is a short, honest fact sheet: what the model is for, how well it works, and where it fails.",
    sections: [
      {
        heading: "What goes in a model card",
        body: [
          "Intended use (and uses to avoid), the training data, the metrics on test data, known limitations, and ethical considerations. Model cards help others decide whether your model is safe to use for their purpose.",
        ],
      },
      {
        heading: "Responsible AI habits",
        body: [
          "Protect privacy (NeuralNest keeps your data on your device). Compare with simple baselines. Don't claim more than your test results show. Keep a human in the loop for important decisions. Document everything.",
        ],
      },
    ],
    tryIt: { href: "/data-lab", label: "Make a model card in the Data Lab" },
    quiz: [
      { q: "A model card should include…", options: ["Only the best metric", "Intended use, metrics and limitations", "The raw training data"], answer: 1, why: "It is an honest summary for others." },
      { q: "Which is a responsible AI habit?", options: ["Hiding limitations", "Comparing against a baseline", "Testing on training data"], answer: 1, why: "Baselines show whether the model really adds value." },
      { q: "Why keep a human in the loop for important decisions?", options: ["Models can be wrong", "Humans are faster", "It's required by maths"], answer: 0, why: "Even good models make mistakes." },
    ],
  },
];
