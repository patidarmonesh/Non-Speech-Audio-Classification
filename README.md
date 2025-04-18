
# 🎤 Non-Speech Human Sound Classification 🎧

The goal of the Non-Speech Human Sound Classification project is to classify various non-speech human sounds such as coughing, laughing, sneezing, and others. This project utilizes Mel Spectrogram features and a Custom Convolutional Neural Network (CNN) model built in PyTorch, trained on Kaggle's TPU for accelerated training.

[![Open In Colab][colab-shield]][colab-url]
<p align="center">
  <img src="images/spectrogram_example.png" alt="Sample Mel Spectrogram" width="600"/>
  <br/><em>Mel Spectrogram representation of an audio sample</em>
</p>
## 📝 Table of Contents

* [Introduction](#-introduction)
* [Dataset](#-dataset)
* [Feature Extraction](#-feature-extraction)
* [Model Architecture](#-model-architecture)
* [Training & Evaluation](#-training--evaluation)
    * [TPU Acceleration](#tpu-acceleration)
    * [Cross-Validation](#cross-validation)
    * [Final Training](#final-training)
* [Results](#-results)
    * [Metrics](#metrics)
    * [Visualizations](#visualizations)
* [🚀 How to Run](#-how-to-run)
* [🛠️ Tools & Technologies Used](#️-tools--technologies-used)
* [🏁 Conclusion](#-conclusion)
* [💡 Future Scope](#-future-scope)
* [📜 License](#-license)
* [Acknowledgements](#acknowledgements)

## 📖 Introduction

This project focuses on classifying non-speech human sounds. Understanding and classifying such sounds can be beneficial in various applications, such as:

* Healthcare monitoring (detecting cough patterns)
* Smart environments (understanding user activity)
* Content analysis (tagging events in audio/video)
* Accessibility tools

We used Mel Spectrograms to process audio signals and designed a custom CNN model incorporating Residual Blocks for classification. Training was accelerated using Google Colab's TPUs.

## 💿 Dataset

The dataset used in this project is Nonspeech7k Dataset (link https://zenodo.org/records/6967442) . It consists of `.wav` files containing various non-speech human sounds.

* **Classes (Example):** Coughing, Laughing, Sneezing, Crying, Snoring, Clapping, Sighing (Based on `num_classes=7` in the code, list the 7 class names here).
* **Preprocessing:**
    * All audio files were resampled to 32,000 Hz.
    * Each audio file was standardized to a fixed length of 4 seconds (using padding or truncation).
    * Converted to mono channel.
* **Dataset Split:** The data was divided into training and testing sets. Stratified K-Fold Cross-Validation (3 Folds) was used on the training set.

## ✨ Feature Extraction

Extracting meaningful features from raw waveforms is crucial for audio classification. We used Mel Spectrograms:

1.  **Loading Audio:** `.wav` files were loaded using the `torchaudio` library.
2.  **Mel Spectrogram:** Each audio waveform was converted into a Mel Spectrogram (`torchaudio.transforms.MelSpectrogram`). This time-frequency representation is closer to human auditory perception.
    * Parameters: `n_fft=2048`, `hop_length=512`, `n_mels=128`.
3.  **Logarithmic Scaling:** Amplitude was converted to Decibels (dB) using `torchaudio.transforms.AmplitudeToDB`.
4.  **Normalization:** The spectrograms were normalized using the mean and standard deviation calculated from the training data to help the model converge better.

## 🧠 Model Architecture

We designed a custom Convolutional Neural Network (CNN) incorporating Residual Blocks to improve performance:

1.  **Initial Convolution:** A standard Conv2D layer (`in_channels=1`, `out_channels=32`) for initial feature extraction, followed by Batch Normalization, ReLU activation, and Max Pooling.
2.  **Residual Blocks:** The model includes 3 Residual Blocks (`ResidualBlock`) that help train deeper networks and mitigate the vanishing gradient problem. The number of channels increases progressively (32 -> 64 -> 128 -> 256). Max Pooling layers are used to reduce spatial dimensions.
3.  **Adaptive Pooling:** An `AdaptiveAvgPool2d` layer converts the feature maps to a fixed-size output (1x1) before flattening.
4.  **Classifier Head:** A Fully Connected (Linear) layer block comprising one hidden layer (512 units), ReLU activation, Dropout (0.5 for regularization), and a final output layer (`num_classes` units) providing class probabilities.

Okay, let's break this down step-by-step in English.

Here's what you need to create a great GitHub repository for your project:

What is Needed:

A README.md File: This is the most important file. It acts as the front page of your project, explaining what it is, how it works, and how others can use it. It must be in Markdown format (.md).
Your Jupyter Notebook: The tpu-with-visuals.ipynb file containing your code.
Visuals Folder: A folder (e.g., named images) inside your repository to store graphs and other images you want to show in the README.
(Optional but Recommended) requirements.txt File: Lists the Python libraries needed to run your code (pip freeze > requirements.txt can help create this).
(Optional but Recommended) LICENSE File: Specifies how others can use your code (e.g., MIT License is common).
(Optional but Recommended) .gitignore File: Tells Git which files or folders to ignore (like large datasets, temporary files).
Step-by-Step Guide: What To Do

Step 1: Create/Update the README.md File

Why Markdown? GitHub uses a file named README.md to display information on your repository's main page. It must be a Markdown file (.md), not a Word document (.docx) or Google Doc. Markdown is a simple text format used for styling text on the web.
How to Create/Edit:
Option A (GitHub Website): Go to your repository page on GitHub.
If README.md doesn't exist, click "Add file" -> "Create new file".
If it exists, click the pencil icon (Edit) on the README.md file.
Name the file README.md.
Copy the complete Markdown content provided below and paste it into the editor.
Click "Commit changes".
Option B (Local Editor - Recommended):
Use a text editor (like VS Code, Sublime Text, Notepad++) on your computer.
Create a new file.
Copy the complete Markdown content below and paste it into the file.
Save the file with the exact name README.md in your project's main folder.
Use Git commands to upload it:
Bash

git add README.md
git commit -m "Add or update README file"
git push
Step 2: Prepare and Add Visuals (Graphs/Images)

Save Your Plots: In your Jupyter notebook (tpu-with-visuals.ipynb), find where you create plots (like the confusion matrix, loss curves, example spectrograms/waveforms). Before the plt.show() command for each plot, add:
Python

plt.savefig('confusion_matrix.png') # Use a descriptive filename
# e.g., plt.savefig('loss_curve.png')
# e.g., plt.savefig('spectrogram_example.png')
plt.show()
Run these cells and download the saved .png image files.
Create an images Folder: In your local project folder (the one you pushed to GitHub), create a new folder named images.
Move Images: Put all the .png files you just saved into this images folder.
Upload the Folder: Use Git to upload this folder and its contents:
Bash

git add images/*
git commit -m "Add graphs and visual examples"
git push
Now the images are part of your repository and can be linked in the README.
Step 3: Add Badges to README.md

Badges make your README look professional and give quick info.
Go to Shields.io.
Search for badges you want (e.g., Python version, PyTorch, license).
Customize them if needed.
Copy the generated Markdown code for each badge.
Paste these badge codes near the top of your README.md file (right after the main title is a good place). The template below already includes some examples.
Step 4: Set the Social Preview (Thumbnail)

This is the image shown when you share your repository link.
Go to your repository on GitHub -> Settings tab -> General section.
Scroll down to Social preview.
Click Edit and upload an image that represents your project (e.g., a key graph, title slide).
Step 5: Organize Your Repository

Ensure your main .ipynb file is present.
Consider adding a requirements.txt file (list dependencies).
Consider adding a LICENSE file (e.g., create one named LICENSE and paste the MIT license text into it).
Ensure the images folder with your graphs is present.
README Content (Copy and Paste into README.md)

Important Note: The following text is formatted in Markdown. You need to save this exact text into a file named README.md. GitHub requires this format to display your project page correctly. Do not save this as a .docx or Google Docs file for your repository's main page.

Markdown

# 🎤 Non-Speech Human Sound Classification 🎧

[![Python Version][python-shield]][python-url]
[![PyTorch Version][pytorch-shield]][pytorch-url]
[![License: MIT][license-shield]][license-url]
[![Open In Colab][colab-shield]][colab-url]

The goal of the Non-Speech Human Sound Classification project is to classify various non-speech human sounds such as coughing, laughing, sneezing, and others. This project utilizes Mel Spectrogram features and a Custom Convolutional Neural Network (CNN) model built in PyTorch, trained on Google Colab TPUs for accelerated training.

This repository was created for a final year engineering project.

[![Open In Colab][colab-shield]][colab-url]
<p align="center">
  <img src="images/spectrogram_example.png" alt="Sample Mel Spectrogram" width="600"/>
  <br/><em>Mel Spectrogram representation of an audio sample</em>
</p>
## 📝 Table of Contents

* [Introduction](#-introduction)
* [Dataset](#-dataset)
* [Feature Extraction](#-feature-extraction)
* [Model Architecture](#-model-architecture)
* [Training & Evaluation](#-training--evaluation)
    * [TPU Acceleration](#tpu-acceleration)
    * [Cross-Validation](#cross-validation)
    * [Final Training](#final-training)
* [Results](#-results)
    * [Metrics](#metrics)
    * [Visualizations](#visualizations)
* [🚀 How to Run](#-how-to-run)
* [🛠️ Tools & Technologies Used](#️-tools--technologies-used)
* [🏁 Conclusion](#-conclusion)
* [💡 Future Scope](#-future-scope)
* [📜 License](#-license)
* [Acknowledgements](#acknowledgements)

## 📖 Introduction

This project focuses on classifying non-speech human sounds. Understanding and classifying such sounds can be beneficial in various applications, such as:

* Healthcare monitoring (detecting cough patterns)
* Smart environments (understanding user activity)
* Content analysis (tagging events in audio/video)
* Accessibility tools

We used Mel Spectrograms to process audio signals and designed a custom CNN model incorporating Residual Blocks for classification. Training was accelerated using Google Colab's TPUs.

## 💿 Dataset

The dataset used in this project is [State your dataset source here - e.g., Kaggle dataset name/link or 'A custom dataset collected from various sources']. It consists of `.wav` files containing various non-speech human sounds.

* **Classes (Example):** Coughing, Laughing, Sneezing, Crying, Snoring, Clapping, Sighing (Based on `num_classes=7` in the code, list the 7 class names here).
* **Preprocessing:**
    * All audio files were resampled to 32,000 Hz.
    * Each audio file was standardized to a fixed length of 4 seconds (using padding or truncation).
    * Converted to mono channel.
* **Dataset Split:** The data was divided into training and testing sets. Stratified K-Fold Cross-Validation (3 Folds) was used on the training set.

## ✨ Feature Extraction

Extracting meaningful features from raw waveforms is crucial for audio classification. We used Mel Spectrograms:

1.  **Loading Audio:** `.wav` files were loaded using the `torchaudio` library.
2.  **Mel Spectrogram:** Each audio waveform was converted into a Mel Spectrogram (`torchaudio.transforms.MelSpectrogram`). This time-frequency representation is closer to human auditory perception.
    * Parameters: `n_fft=2048`, `hop_length=512`, `n_mels=128`.
3.  **Logarithmic Scaling:** Amplitude was converted to Decibels (dB) using `torchaudio.transforms.AmplitudeToDB`.
4.  **Normalization:** The spectrograms were normalized using the mean and standard deviation calculated from the training data to help the model converge better.

<p align="center">
  <img src="images/waveform_example.png" alt="Sample Waveform" width="45%"/>
  <img src="images/spectrogram_example.png" alt="Sample Mel Spectrogram" width="45%"/>
  <br/><em>Left: Sample Audio Waveform, Right: Corresponding Mel Spectrogram</em>
</p>
## 🧠 Model Architecture

We designed a custom Convolutional Neural Network (CNN) incorporating Residual Blocks to improve performance:

1.  **Initial Convolution:** A standard Conv2D layer (`in_channels=1`, `out_channels=32`) for initial feature extraction, followed by Batch Normalization, ReLU activation, and Max Pooling.
2.  **Residual Blocks:** The model includes 3 Residual Blocks (`ResidualBlock`) that help train deeper networks and mitigate the vanishing gradient problem. The number of channels increases progressively (32 -> 64 -> 128 -> 256). Max Pooling layers are used to reduce spatial dimensions.
3.  **Adaptive Pooling:** An `AdaptiveAvgPool2d` layer converts the feature maps to a fixed-size output (1x1) before flattening.
4.  **Classifier Head:** A Fully Connected (Linear) layer block comprising one hidden layer (512 units), ReLU activation, Dropout (0.5 for regularization), and a final output layer (`num_classes` units) providing class probabilities.

## 🏋️ Training & Evaluation
TPU Acceleration
Google Colab TPUs were utilized to accelerate model training. The torch_xla library enabled running PyTorch code on TPUs, significantly reducing the training time per epoch.

Cross-Validation
To assess the model's robustness and generalization, Stratified 3-Fold Cross-Validation was performed on the training data. Stratified sampling ensures that the class distribution in each fold mirrors the original dataset. The best model for each fold was saved based on the F1-score.

Final Training
After cross-validation, a final model was trained on the entire training dataset (num_epochs = 50) to leverage the maximum available data.

Training Details:

Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
Loss Function: CrossEntropyLoss (Class weights were used to handle class imbalance)
Learning Rate Scheduler: ReduceLROnPlateau (Reduces learning rate when validation loss stops improving)
Batch Size: 32
Epochs: 50 (for CV folds and final training)
Data Augmentation: Frequency Masking, Time Masking, Time Shifting, Scaling, and Noise Injection (AugmentDataset) were applied during training to improve the model's generalization ability.

📊 Results
Metrics
The final model was evaluated on the Test Set. The metrics below reflect the performance of the final model on the Test Set:
Test Set Metrics: Loss=0.6264, F1=0.8753, Precision=0.8612, Recall=0.8969

Cross-Validation Average Metrics (on Validation Sets):
3-Fold CV Average: F1=0.9136, Precision=0.9098, Recall=0.9201

🚀 How to Run
Clone Repository: git clone [https://github.com/patidarmonesh/Non-Speech-Audio-Classification.git](https://github.com/patidarmonesh/Non-Speech-Audio-Classification.git)
cd Non-Speech-Audio-Classification

Dataset:
Ensure your dataset (.wav files and metadata .csv files) is available.
Place the metadata of train set.csv and metadata of test set.csv in the location expected by the notebook (e.g., root, or update path in code).
Place the audio files in audio_files/train/ and audio_files/test/ relative to where the metadata CSVs are, or update the paths in the AudioDataset class in the notebook.

Open in Kaggle:
Upload the tpu-with-visuals.ipynb notebook to Kaggle.

Setup Environment:
In Colab: Runtime -> Change runtime type -> Select TPU as Hardware accelerator.

Run Notebook: Execute the cells sequentially. Make sure dataset paths are correct within the notebook.

Results: View training progress, final metrics, and generated plots in the notebook output. Saved models (e.g., best_foldX.pth) will appear in the Kaggle environment's working directory.

🛠️ Tools & Technologies Used
Programming Language: Python 3.x
Deep Learning: PyTorch, PyTorch XLA (for TPU)
Audio Processing: Torchaudio
Data Handling: Pandas, NumPy
Machine Learning: Scikit-learn (for Cross-Validation, Metrics)
Plotting: Matplotlib
Environment: Kaggle (with TPU), Jupyter Notebook

🏁 Conclusion
This project successfully developed a CNN model for classifying non-speech human sounds using Mel Spectrograms. The model, accelerated by TPUs, achieved an F1-score of .8754 on the test set. Techniques like residual connections, data augmentation, and class weighting contributed to the performance.

💡 Future Scope
Expand Dataset: Incorporate more data and potentially more diverse sound classes.
Advanced Architectures: Explore Transformer-based models (e.g., Audio Spectrogram Transformer) or other CNN variants.
Real-time Application: Optimize the model for deployment in real-time audio processing scenarios.
Fine-tuning: Investigate fine-tuning pre-trained audio models like VGGish or YAMNet.
Error Analysis: Perform a deeper analysis of misclassified samples to guide model improvements.

