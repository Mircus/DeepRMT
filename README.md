# 📊 Random Matrix Theory for Deep Models

> Opening the hood of deep networks using spectral statistics and matrix ensembles

📖 **Related article**: [Random Matrix Theory: Opening the Hood of Deep Models](https://medium.com/@polymathicus/random-matrix-theory-opening-the-hood-of-deep-models-6b3764b5032b)

---

## 🧮 What is this project about?

This repo provides tools and experiments that apply **Random Matrix Theory (RMT)** to analyze the inner workings of deep learning models — particularly their **weight matrices**, **Jacobian spectra**, and **training dynamics**.

We explore how concepts like the **Marchenko-Pastur law**, **Wigner semicircle**, and **spectral entropy** can be used to diagnose learning behaviors, detect overfitting, or even forecast generalization.

---

## 🔬 Features

* 📉 Spectral density plots of weight and Jacobian matrices
* 🧠 Comparison with RMT distributions (Wigner, MP, Tracy-Widom)
* 🔍 Metrics: eigenvalue outliers, condition number, entropy
* 📊 Temporal tracking of spectral statistics during training
* 📦 Works with PyTorch-trained models

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Mircus/DeepRMT.git
cd DeepRMT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch analysis notebook

```bash
jupyter notebook RMT_Analysis.ipynb
```

---

## 🧠 How It Works

1. Extract weight matrices (or Jacobians) layer by layer
2. Compute eigenvalues or singular values
3. Compare empirical spectrum to RMT laws
4. Visualize and interpret the results

> This method reveals the "spectral fingerprint" of your model's learning stage and health.

---

## 📚 Background

We build on insights from:

* Terence Tao, Alan Edelman — random matrix theory
* Martin & Mahoney — heavy-tailed self-regularization in DNNs
* Pennington et al. — Jacobian analysis of deep nets

---

## 🧰 Tech Stack

* Python, PyTorch
* NumPy, SciPy, Matplotlib, Seaborn
* Jupyter Notebook

---

## 📄 License

MIT License — see `LICENSE` file.

---

## 🙋‍♂️ Contributions

Pull requests and suggestions are welcome. If you build on this work in a paper or article, please link back to the Medium post or this repository.

> *Part of the Holomathics research ecosystem.*
