# nyc_taxi_xgboost_lab
GPU-accelerated machine learning workflow on 2023 NYC Yellow Taxi using XGBoost, cuDF Pandas, and RMM.

## Video Tutorial 🎥
<a href="https://youtu.be/F_8RKstP2X8" target="_blank"><img width="600" alt="Real-World Machine Learning Project thumbnail" src="https://github.com/user-attachments/assets/ec3ac34f-7a0c-4bcf-9de8-40ab1ef8de3b" /></a>

---

## Overview 📚

Ever wondered what makes people tip more in taxis? 💵  
In this hands-on machine learning project, we’ll build a complete workflow on **real NYC Yellow Taxi data** — cleaned, engineered, and trained entirely on **GPU using XGBoost CUDA and cuDF Pandas**. 🐼

You’ll see how professionals approach problems, handle massive data, and fix memory errors — designing real data-science pipelines step-by-step.

---

## Project Files 🗃️

- **Distilled_2023_Yellow_Taxi_Trip_Data.csv**
  <br>a smaller, 5 million records, version of the <a href="https://data.cityofnewyork.us/Transportation/2023-Yellow-Taxi-Trip-Data/4b4i-vvec/about_data" target="_blank">2023 Yellow Taxi Trip Data</a> from the City of New York.
- **TutorialWorkflow.ipynb**
  <br>Tne official tutorial notebook from the video. A step by step workflow for beginners.
- **AdvancedWorkflow.ipynb**
  <br>A "Best Practices" version of the tutorial workflow above with K-Fold validation and hyperparameter tuning.
  
---

## Environment Setup 🚀 

You can run this project in two ways:

### 1️⃣ Google Colab  
- Change runtime to **T4 GPU**  
- Use the smaller version of the dataset (≈ 5 million rows)  
- Download from repo and store in your Google Drive `Distilled_2023_Yellow_Taxi_Trip_Data.csv`
- Use the following code snippet to mount your goodle drive:
```
# mount your google drive - connect it to your notebook
from google.colab import drive
drive.mount('/content/drive')
```
- Use the following code snippet to load your dataset:
```
%load_ext cudf.pandas
import pandas as pd

data = pd.read_csv("/content/drive/MyDrive/path_to_your_dataset/Distilled_2023_Yellow_Taxi_Trip_Data.csv")
data.tail()
```

### 2️⃣ Local Setup  
- Requires a **CUDA-compatible GPU**  
- Use the full **NYC Taxi Dataset (≈ 38 million rows)**  
  [Download from NYC Open Data](https://data.cityofnewyork.us/Transportation/2023-Yellow-Taxi-Trip-Data/4b4i-vvec/about_data)
- Use **WSL + Miniforge/Conda** (other setups may fail)  
- Install modules using the [RAPIDS installation guide](https://docs.rapids.ai/install/)  
- For example: the **xgb_env** environment setup command from my video:
```
conda create -n xgb_env -c rapidsai-nightly -c conda-forge -c nvidia  \
    cudf=25.12 python=3.13 'cuda-version=13.0' \
    jupyterlab xgboost matplotlib
```

---

## Credits 🌟

Created with ❤️ by **[Mariya Sha](https://www.linkedin.com/in/mariyasha888)**  
Part of the **[Python Simplified](https://youtube.com/@PythonSimplified)** educational YouTube channel.

Beautiful thumbnail icon assembled from: https://flaticon.com

---

### Keywords 🔑
machine learning, data science, xgboost, cudf, rapids, python, pandas, gpu acceleration, big data, beginner project, real-world dataset, nyc taxi, kaggle alternative, portfolio project, ai tutorial, python simplified

