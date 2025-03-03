# DANCER-HTR

## Overview
This repository is an unofficial reimplementation of **DANCER** by David Castro for **Handwritten Text Recognition (HTR)**. It is based on the paper:

**"An End-to-End Approach for Handwriting Recognition: From Handwritten Text Lines to Complete Manuscripts"**  
[ResearchGate Link](https://www.researchgate.net/publication/383518443_An_End-to-End_Approach_for_Handwriting_Recognition_From_Handwritten_Text_Lines_to_Complete_Manuscripts)

## Getting Started

### Prerequisites
Ensure you have the following dependencies installed:
- **Python 3.9.1**
- **PyTorch 1.8.2**
- **CUDA 10.2** (for GPU support)

### Installation
Clone the repository:
```sh
git clone https://github.com/Laksh-Mendpara/DANCER-HTR.git
cd DANCER-HTR
```

Install required dependencies:
```sh
pip install -r requirements.txt
```

## Quick Prediction
An example script is available at:
```
OCR/document_OCR/dan/predict_examples
```
This script allows direct recognition of images using trained model weights.
(Note: Model weights will be uploaded soon.)

## Datasets
This repository uses the **READ 2016 dataset**, which was part of the ICFHR 2016 competition on handwritten text recognition.

You can find the dataset [here](https://www.read2016.com) (link placeholder).

Place the raw dataset files in the following directory:
```
Datasets/raw/READ_2016
```

## Training and Evaluation
### Step 1: Download the Dataset
Download and place the dataset in `Datasets/raw/READ_2016`.

### Step 2: Format the Dataset
Run the following command to format the dataset:
```sh
python -m Datasets.dataset_formatters.read2016_formatter
```

### Step 3: Add Custom Fonts
Add any `.ttf` font files to the `Fonts` directory to use them for synthetic data generation.

### Step 4: Generate Synthetic Line Dataset for Pre-Training
Run:
```sh
python -m ctc_training.main_syn_line
```
Modify the following lines in the script to match the dataset used:
```python
model.generate_syn_line_dataset("READ_2016_syn_line")
dataset_name = "READ_2016"
```

### Step 5: Pre-Training on Synthetic Lines
Run:
```sh
python -m ctc_training.main_line_ctc
```
Modify the following lines in the script:
```python
dataset_name = "READ_2016"
"output_folder": "FCN_read_line_syn"
```
Weights and evaluation results will be stored in:
```
./outputs
```

### Step 6: Training the DANCER Model
Run:
```sh
python -m doc_training.main
```

## License
This project is licensed under the **MIT License**.

