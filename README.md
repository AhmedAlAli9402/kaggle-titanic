# Titanic - Machine Learning from Disaster 🚢

This project is based on the classic Kaggle challenge: predicting survival on the Titanic using machine learning.

## 📁 Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
├── notebook/
│   └── titanic-notebook.ipynb
├── scripts/
│   ├── main.py
│   ├── module.py
│   └── utils/
│       ├── preprocess_data.py
│       ├── create_model.py
│       ├── handle_test_data.py
│       ├── split_scale_data.py
│       └── plot_graphs.py
```

## ⚙️ Setup

1. Clone the repo and navigate to the directory.
2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate kaggle-titanic
```

3. Launch the notebook:

```bash
jupyter lab
```

## 🧠 Model Summary

The model combines classical machine learning (e.g., XGBoost, Logistic Regression) and deep learning (via Keras) to predict Titanic survivors based on features like `Sex`, `Age`, `Pclass`, and `Family Size`.

## ✍️ Author

Ahmed AlAli – Kaggle Titanic Challenge Project

---

Happy modeling! 🚀
