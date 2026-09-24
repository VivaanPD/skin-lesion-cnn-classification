# Skin Lesion Classification with Transfer Learning

A research project comparing **ResNet50** and **VGG16** transfer-learning models for multi-class skin-lesion image classification. The project evaluates the two architectures using accuracy, weighted F1 score, and confusion-matrix analysis.

The accompanying paper asks how the architectural differences between ResNet50 and VGG16 affect their performance on a skin-lesion classification task. Both models were initialized with ImageNet weights and fine-tuned on the same dataset and evaluation pipeline.

## Results

| Model | Training Accuracy | Test Accuracy | Weighted F1 |
|---|---:|---:|---:|
| **ResNet50** | 82.0% | **74.83%** | **0.7158** |
| **VGG16** | 79.0% | 72.7% | 0.7119 |

ResNet50 achieved the higher reported test accuracy and weighted F1 score. The confusion-matrix analysis also showed that performance varied substantially by class, with class imbalance affecting both models and ResNet50 frequently predicting the dominant `nevus` class.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── skin_lesion_cnn_experiment.ipynb
├── models/
│   ├── resnet_model.keras        # add trained model
│   └── vgg_model.keras           # add trained model
└── paper/
    └── CNN_research.pdf
```

The notebook contains the code from Appendix C of the paper, reorganized into executable cells. It also includes a clearly labeled supplementary evaluation cell for accuracy and confusion matrices because those calculations are discussed in the paper but are not fully included in the printed appendix.

## Methodology

The experiment uses a labeled skin-lesion image dataset containing **11,721 images across 8 classes**, as described in the paper.

The Appendix C preprocessing code performs stratified splitting in two stages:

1. 10% of the full dataset is reserved for testing.
2. 16.67% of the remaining 90% is reserved for validation.

That corresponds to approximately **75% training / 15% validation / 10% testing**.

Images are resized to **224 × 224** and loaded in batches of **32**. The training pipeline applies rescaling and data augmentation using rotation, translation, shear, zoom, and horizontal flipping. Validation and test images are rescaled without augmentation.

For each architecture, the original classification head is removed and replaced with:

```text
pretrained convolutional backbone
        ↓
GlobalAveragePooling2D
        ↓
Dense(num_classes, activation="softmax")
```

Both models are trained with the Adam optimizer and categorical cross-entropy loss. The Appendix C code uses early stopping with a patience of 10 epochs and checkpoints the best validation-loss model.

## Reproducing the experiment

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

The preprocessing section of the notebook expects:

- a metadata CSV containing `isic_id` and `diagnosis` columns;
- a directory containing the corresponding `.jpg` images; and
- an output directory where the train/validation/test class folders can be created.

Update the three path variables at the beginning of the preprocessing cell before running it:

```python
metadata_path = "..."
images_directory = "..."
dataset_directory = "..."
```

### 3. Train the models

Run the training section after the organized dataset has been created. The notebook uses pretrained ImageNet weights for both ResNet50 and VGG16 and saves the best checkpoints as:

```text
resnet_model.keras
vgg_model.keras
```

### 4. Evaluate

The evaluation section loads the trained checkpoints and calculates weighted F1 scores. The supplementary evaluation cell also calculates explicit test accuracy and confusion matrices.

## Trained models

The trained `.keras` checkpoints can be added to the `models/` directory. If the checkpoint files are too large for normal Git tracking, store them with **Git LFS** or attach them to a GitHub Release instead of committing the raw binaries directly.

Example with Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes models/
```

## Reproducibility notes

The original paper and its Appendix C are not perfectly aligned in two places. The prose methodology describes a 70/15/15 train/validation/test split, while the appendix code produces approximately 75/15/10. The reported test-set size is consistent with the 10% test split, so the notebook preserves the appendix implementation.

The paper's limitations section also refers to 30 training epochs, whereas the printed Appendix C training code sets `epochs=25` with early stopping. The notebook preserves the Appendix C value rather than silently changing it.

## Research paper

The full write-up is available at [`paper/CNN_research.pdf`](paper/CNN_research.pdf). It includes the motivation, CNN background, experimental design, model comparison, per-class error analysis, limitations, and the original code appendix.

## Limitations

The experiment has several important limitations discussed in the paper: the dataset is class-imbalanced, training time was constrained, and the models use pretrained architectures rather than custom CNN implementations. As a result, overall accuracy alone should not be treated as sufficient evidence of performance across all lesion classes.

## Medical-use disclaimer

This project was completed for research and educational purposes. The models are **not** clinically validated and should not be used for medical diagnosis or treatment decisions.
