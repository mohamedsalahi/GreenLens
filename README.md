# GreenLens

GreenLens is a native desktop application that identifies plant diseases from
leaf images. A convolutional classifier built from a VGG16 backbone (TensorFlow
/ Keras, ImageNet transfer learning) runs in a local Python inference process;
a Rust desktop front end built with egui opens an image, sends it to the
worker, and displays the ranked class predictions with their probabilities.

Supported crops and classes depend on the dataset subset the shipped model was
trained on. See `docs/` and the "Model" section below for the exact class list
and honest evaluation numbers.

Status: work in progress. The README is updated as milestones complete.
