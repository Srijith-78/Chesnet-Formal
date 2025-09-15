# Pneumothorax Classification and Segmentation Project

## Overview

This project provides two functionalities:

1. **Pneumothorax Classification** using pre-trained ResNet50 and DenseNet121 models.
2. **Dataset Preparation**: Cleaning the test dataset by ensuring images referenced in the CSV exist in the `images/` directory.

## 📥 Dataset Download

### 1️⃣ CheXpert Dataset (for classification)

* Download the [CXR8 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
* Place the metadata CSV `Data_Entry_2017_v2020.csv` in your working directory.
* Prepare a `test_list.txt` file containing the test image filenames, one per line. (make sure all tar.gz files are compressed and then finally put inside a single directory images)

## ⚙️ Dataset Preparation

Use the following script to filter the test CSV and ensure all images exist:

```bash
python test.py
```

This will generate `cleaned_test.csv` containing only valid image entries.

## 🚀 Model Training and Evaluation

### Install dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm
```

### Run Training and Evaluation

#### Example command to train both ResNet50 and DenseNet121:

```bash
python pneumothorax_model_compare.py --data_csv cleaned_test.csv --img_dir images/ --output_dir outputs/ --epochs 10 --batch 16 --models resnet50 densenet121 --pretrained
```

#### Important Arguments:

* `--data_csv`: Path to the dataset CSV (use `cleaned_test.csv`).
* `--img_dir`: Directory containing the images.
* `--output_dir`: Directory where models and plots will be saved.
* `--models`: List of models to train (`resnet50` and/or `densenet121`).
* `--epochs`: Number of epochs to train.
* `--batch`: Batch size.
* `--pretrained`: Use pretrained weights for faster convergence.

## 📊 Outputs

* Model weights (e.g., `resnet50_pneumo.pth`).
* Confusion matrix plot.
* ROC and PR curve plots.
* A summary printed in the terminal with Accuracy, Precision, Recall, F1, ROC AUC, and Log Loss.

## ✅ Notes

* Ensure that `cleaned_test.csv` contains only valid image paths.
* Adjust `--epochs` and `--batch` based on your hardware.

## ⚡ Contact

For further questions, feel free to open an issue on the GitHub repository.
