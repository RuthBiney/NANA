Wine Quality Prediction Using Transfer Learning
Description of the Problem Statement
The goal of this project is to apply transfer learning techniques to classify images related to alcoholic beverages, specifically focusing on wine, beer, and whiskey. The primary objective is to leverage pre-trained models (VGG16, ResNet50, and InceptionV3) to fine-tune them for this classification task, significantly reducing training time and improving performance.

The dataset used contains images categorized into three classes: wine, beer, and whiskey. The images were preprocessed, augmented, and split into training and validation sets to train and evaluate the performance of these models.

Dataset
The dataset comprises a total of 1365 images in the training set and 340 images in the validation set, with each class (wine, beer, whiskey) well represented.

Overview of Evaluation Metrics
To assess the performance of the fine-tuned models, we used the following evaluation metrics:

Accuracy: Measures the overall correctness of the model's predictions.
Loss: The error between the predicted output and the true output, minimized during training.
Precision: The ratio of correctly predicted positive observations to the total predicted positives.
Recall: The ratio of correctly predicted positive observations to the actual positives.
F1 Score: The harmonic mean of precision and recall, providing a balanced measure.

| **Model**       | **Accuracy** | **Loss** | **Precision** | **Recall** | **F1-Score** |
| --------------- | ------------ | -------- | ------------- | ---------- | ------------ |
| **VGG16**       | 0.85         | 0.45     | 0.87          | 0.82       | 0.84         |
| **ResNet50**    | 0.88         | 0.40     | 0.90          | 0.85       | 0.87         |
| **InceptionV3** | 0.86         | 0.42     | 0.88          | 0.83       | 0.85         |
