# 🚀 Fast-Warmer

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Fast-Warmer** solves the **cold start problem** for video hosting platforms using **ALS**, **BERT**, and **KNN** models. The project includes a **Streamlit frontend** to visualize personalized video recommendations.

---

## 🗂️ Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)

---

## 📖 Introduction

**Fast-Warmer** provides a hybrid recommendation model using:
- **Collaborative filtering (ALS)**
- **Content-based filtering (BERT)**
- **Neighborhood-based filtering (KNN)**

It generates video recommendations for new users or videos without previous interaction data.

## ✨ Features

- Hybrid recommendation engine (ALS + BERT + KNN)
- Solves cold start with personalized suggestions
- Real-time predictions via a **Streamlit** UI

## 🏗️ Architecture

The hybrid system combines **ALS**, **BERT**, and **KNN** models into a single core for generating recommendations, visualized through the **Streamlit** frontend.

```mermaid
graph TD
    A[ALS Collaborative Filtering] --> B[Fast-Warmer Core]
    C[BERT Content-Based Filtering] --> B
    D[KNN Neighborhood-Based Filtering] --> B
    B --> E[Streamlit Frontend]
```

## 🛠️ Technologies 
- Backend: Python, ALS, BERT, KNN
- Frontend: Streamlit (main.py)
- Libraries: pandas, torch, scikit-learn

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fast-warmer.git
cd fast-warmer
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the app:**
```bash
streamlit run main.py
```
