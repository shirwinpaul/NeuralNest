import type { Lesson } from "./types";

export const LESSONS_A: Lesson[] = [
  {
    id: "what-is-ml",
    title: "What is machine learning?",
    emoji: "🤖",
    illustration: "ml",
    summary: "Instead of writing rules by hand, we show a computer examples and let it find the rules itself.",
    sections: [
      {
        heading: "Rules vs. examples",
        body: [
          "In normal programming, a person writes the rules: \"if the email contains 'WIN A PRIZE', mark it as spam\". That works until spammers change their words.",
          "In machine learning (ML), we give the computer lots of examples — emails labelled spam or not spam — and an algorithm adjusts a model until its answers match the labels. The model has learned the rules from data.",
        ],
      },
      {
        heading: "What is a model?",
        body: [
          "A model is a mathematical function with adjustable numbers inside it, called parameters (or weights). Training means nudging those numbers so the model's outputs get closer to the right answers.",
          "A neural network is one kind of model. It is built from many tiny units called neurons, connected in layers. In NeuralNest you will build real ones and train them right in your browser.",
        ],
      },
      {
        heading: "Learning is not magic",
        body: [
          "A model only knows what its data shows it. If the examples are few, messy or one-sided, the model will be too. That is why good ML is as much about data as about maths.",
        ],
      },
    ],
    tryIt: { href: "/playground", label: "Train your first network" },
    quiz: [
      { q: "What does a machine-learning model learn from?", options: ["Rules typed by a programmer", "Examples (data)", "The internet automatically"], answer: 1, why: "ML models adjust their parameters to fit example data." },
      { q: "What are a model's parameters (weights)?", options: ["Adjustable numbers inside the model", "The labels of the data", "The screen size"], answer: 0, why: "Training changes these numbers so outputs match the labels." },
      { q: "If the training data is one-sided, the model will probably…", options: ["Fix the problem by itself", "Be one-sided too", "Refuse to train"], answer: 1, why: "A model can only learn patterns that are in its data." },
    ],
  },
  {
    id: "data-features-labels",
    title: "Data, features & labels",
    emoji: "🗂️",
    illustration: "features",
    summary: "Features are the clues we give the model; labels are the answers we want it to predict.",
    sections: [
      {
        heading: "Rows, features and labels",
        body: [
          "A dataset is usually a table. Each row is one example (one fruit, one student, one drawing). Each column describing it is a feature — for a fruit: weight, sweetness, colour.",
          "The label (or target) is the answer we want to predict — for example the fruit type. Training uses examples where we already know the label.",
        ],
      },
      {
        heading: "Classification vs. regression",
        body: [
          "If the label is a category (cat / dog, spam / not spam) the task is classification. If the label is a number (exam score, house price) the task is regression.",
          "Choosing good features matters a lot. In the Playground, adding the feature x₁·x₂ lets even a network with no hidden layers solve XOR — because the feature already contains the pattern.",
        ],
      },
      {
        heading: "Watch out for leaky features",
        body: [
          "A feature that secretly contains the answer (like a column copied from the label, or an ID that was sorted by label) makes a model look brilliant in testing but useless in real life. The Data Lab warns you about these.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=xor", label: "Solve XOR with a clever feature" },
    quiz: [
      { q: "Predicting a house price is…", options: ["Classification", "Regression", "Neither"], answer: 1, why: "The label is a number, so it is regression." },
      { q: "In a fruit table, 'sweetness' is a…", options: ["Label", "Feature", "Parameter"], answer: 1, why: "It is a clue used to predict the label (fruit type)." },
      { q: "A column that is an exact copy of the label is a sign of…", options: ["Leakage", "Good data", "Regularisation"], answer: 0, why: "It leaks the answer, so the model cheats." },
    ],
  },
  {
    id: "neurons-activations",
    title: "Neurons & activation functions",
    emoji: "🧠",
    illustration: "neuron",
    summary: "A neuron multiplies its inputs by weights, adds them up, then squashes the result with an activation function.",
    interactive: "activations",
    sections: [
      {
        heading: "One artificial neuron",
        body: [
          "A neuron takes inputs x₁, x₂, …, multiplies each by a weight w, adds them together with a bias b, and gets z = w₁x₁ + w₂x₂ + b.",
          "Then it applies an activation function a = f(z). Without activations, stacking layers would just give another straight-line model — activations add the bends that let networks learn curvy patterns.",
        ],
      },
      {
        heading: "Common activations",
        body: [
          "ReLU: f(z) = max(0, z). Simple and fast; the default for hidden layers in most modern networks.",
          "Sigmoid: squashes any number into 0…1, handy for a yes/no probability. Tanh: squashes into −1…1. Softmax: turns a list of scores into probabilities that add up to 1 (used for multi-class outputs).",
        ],
      },
      {
        heading: "Layers",
        body: [
          "Neurons are arranged in layers. The input layer holds the features, hidden layers transform them, and the output layer gives the prediction. More neurons and layers can learn more complex shapes — but also overfit more easily.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=neurons", label: "Change activations in the Playground" },
    quiz: [
      { q: "What does ReLU output for z = −3?", options: ["−3", "0", "3"], answer: 1, why: "ReLU is max(0, z), so negatives become 0." },
      { q: "Why do networks need activation functions?", options: ["To add non-linear bends", "To make training slower", "To store data"], answer: 0, why: "Without them, many layers collapse into one straight-line model." },
      { q: "Which activation turns scores into probabilities that sum to 1?", options: ["ReLU", "Tanh", "Softmax"], answer: 2, why: "Softmax normalises scores into a probability distribution." },
    ],
  },
  {
    id: "loss-gradient-descent",
    title: "Loss & gradient descent",
    emoji: "⛰️",
    illustration: "gradient",
    summary: "The loss measures how wrong the model is. Gradient descent walks downhill on the loss to reduce it.",
    interactive: "gradient-descent",
    sections: [
      {
        heading: "The loss",
        body: [
          "A loss function turns \"how wrong are we?\" into one number. For regression we often use mean squared error (the average of squared differences). For classification we use cross-entropy, which punishes confident wrong answers heavily.",
          "Lower loss = better fit on the data we measure it on.",
        ],
      },
      {
        heading: "Walking downhill",
        body: [
          "Imagine the loss as a hilly landscape where each position is a choice of weights. The gradient tells us which direction is uphill. Gradient descent takes a small step the opposite way — downhill — and repeats.",
          "Each step: weight ← weight − learning rate × gradient. Computers find the gradient of every weight at once with an algorithm called backpropagation.",
        ],
      },
      {
        heading: "Try it",
        body: ["Use the 1D toy below: drag the starting point and the learning rate, then step. Watch the ball roll toward the bottom of the valley — or overshoot if the steps are too big!"],
      },
    ],
    tryIt: { href: "/playground", label: "Watch the loss curve drop" },
    quiz: [
      { q: "A lower loss means the model is…", options: ["More wrong", "Less wrong on that data", "Bigger"], answer: 1, why: "Loss measures error; lower is better." },
      { q: "Gradient descent moves the weights…", options: ["Uphill", "Downhill on the loss", "Randomly"], answer: 1, why: "It steps opposite to the gradient to reduce loss." },
      { q: "The algorithm that computes gradients for all weights is…", options: ["Backpropagation", "Softmax", "Dropout"], answer: 0, why: "Backpropagation applies the chain rule layer by layer." },
    ],
  },
  {
    id: "learning-rate",
    title: "The learning rate",
    emoji: "🎚️",
    illustration: "lr",
    summary: "The learning rate is the step size. Too small is slow; too big bounces around or blows up.",
    sections: [
      {
        heading: "Step size matters",
        body: [
          "The learning rate controls how far gradient descent moves each step. With a tiny learning rate the loss creeps down very slowly. With a huge one, steps overshoot the valley and the loss jumps around — or explodes to infinity (shown as NaN).",
          "Good values depend on the optimizer: Adam often works well around 0.001–0.03; plain SGD may need larger values.",
        ],
      },
      {
        heading: "How to spot problems",
        body: [
          "Loss going down very slowly and steadily → learning rate probably too low. Loss zig-zagging or rising → probably too high. Try changing it by a factor of 3 or 10, not tiny tweaks.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=lr-high", label: "Try a learning rate that's too high" },
    quiz: [
      { q: "A loss curve that zig-zags wildly suggests the learning rate is…", options: ["Too low", "Too high", "Perfect"], answer: 1, why: "Big steps overshoot the minimum." },
      { q: "A very slow, steady loss decrease suggests…", options: ["Learning rate too low", "Learning rate too high", "Too much data"], answer: 0, why: "Tiny steps make slow progress." },
      { q: "A sensible way to tune the learning rate is to change it by…", options: ["0.0001 at a time", "Factors like ×3 or ×10", "Never change it"], answer: 1, why: "Its effect is roughly logarithmic." },
    ],
  },
  {
    id: "epochs-batches",
    title: "Epochs & batches",
    emoji: "🔁",
    illustration: "epochs",
    summary: "An epoch is one full pass over the training data; a batch is the small group used for each weight update.",
    sections: [
      {
        heading: "Batches",
        body: [
          "Instead of computing the gradient on all examples at once, we usually use small batches (for example 16 or 32 examples). Each batch gives one weight update. Small batches give noisier but more frequent updates; big batches are smoother but use more memory.",
        ],
      },
      {
        heading: "Epochs",
        body: [
          "One epoch means the model has seen every training example once. Training usually takes many epochs. Too few → the model hasn't learned enough (underfitting). Too many → it may start memorising the training data (overfitting).",
          "Early stopping watches the validation loss and stops training when it stops improving.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=batches", label: "Train with batch size 1" },
    quiz: [
      { q: "One epoch is…", options: ["One weight update", "One pass over all training data", "One neuron"], answer: 1, why: "Every training example is seen once per epoch." },
      { q: "With 100 examples and batch size 10, how many updates per epoch?", options: ["1", "10", "100"], answer: 1, why: "100 / 10 = 10 batches, one update each." },
      { q: "Early stopping stops training when…", options: ["The validation loss stops improving", "The first epoch ends", "Accuracy hits 50%"], answer: 0, why: "It prevents training long after the model stops generalising better." },
    ],
  },
  {
    id: "overfitting",
    title: "Overfitting & underfitting",
    emoji: "🎯",
    illustration: "overfit",
    summary: "Overfitting = memorising the training data; underfitting = not learning the pattern at all.",
    sections: [
      {
        heading: "Two ways to fail",
        body: [
          "Underfitting: the model is too simple (or trained too little) to capture the pattern — both training and test scores are poor.",
          "Overfitting: the model learns the training data too well, including its noise. Training accuracy is great but test accuracy is much worse. The gap between them is the warning sign.",
        ],
      },
      {
        heading: "How to fight overfitting",
        body: [
          "Get more data, use a smaller model, add regularisation (L2 or dropout), stop earlier, or use data augmentation. Always judge your model on data it did not train on.",
        ],
      },
    ],
    tryIt: { href: "/playground?preset=overfit", label: "Make it overfit, then fix it" },
    quiz: [
      { q: "Train accuracy 99%, test accuracy 70% suggests…", options: ["Underfitting", "Overfitting", "A perfect model"], answer: 1, why: "A big train–test gap means memorising." },
      { q: "Both train and test accuracy are low. That's…", options: ["Underfitting", "Overfitting", "Leakage"], answer: 0, why: "The model hasn't captured the pattern." },
      { q: "Which helps reduce overfitting?", options: ["A bigger model", "More data or regularisation", "Testing on training data"], answer: 1, why: "More data and regularisation reduce memorisation." },
    ],
  },
  {
    id: "splits",
    title: "Train / validation / test splits",
    emoji: "✂️",
    illustration: "split",
    summary: "Train on one part, tune on another, and keep a final part hidden for an honest score.",
    sections: [
      {
        heading: "Three piles",
        body: [
          "Training set: the model learns from it. Validation set: you use it to compare settings (learning rate, layers) and for early stopping. Test set: kept aside until the end to get an honest estimate of real-world performance.",
          "If you keep tweaking until the test score looks good, the test set stops being honest — you have indirectly trained on it.",
        ],
      },
      {
        heading: "Stratified and seeded splits",
        body: [
          "A stratified split keeps the same class proportions in each pile, so a rare class doesn't vanish from the test set. A seed makes the random split reproducible — run it again and you get the same piles. NeuralNest does both.",
        ],
      },
    ],
    tryIt: { href: "/data-lab", label: "See the split in the Data Lab" },
    quiz: [
      { q: "Which set gives the most honest final score?", options: ["Training", "Validation", "Test"], answer: 2, why: "It is untouched until the end." },
      { q: "A stratified split keeps…", options: ["Class proportions similar in each set", "All data in training", "Only one class"], answer: 0, why: "Each class is split separately." },
      { q: "Why use a seed for splitting?", options: ["To make results reproducible", "To make training faster", "To hide data"], answer: 0, why: "Same seed → same split every time." },
    ],
  },
];
