# CardioAI

>I can't replace a cardiologist. But I built something that knows when you need one.

An end-to-end ML pipeline for cardiovascular risk assessment — benchmarked five classification algorithms, selected KNN at 88.6% accuracy, and deployed as a production-ready Streamlit web app with real-time risk scoring and interactive clinical analytics.

🌐 **Live Demo:** [cardio-ai-bishwajitpattanaik.streamlit.app](https://cardio-ai-bishwajitpattanaik.streamlit.app)

---

## 💻 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| Streamlit | 1.57.0 | Web app framework |
| Scikit-learn | 1.8.0 | ML model training & evaluation |
| Pandas | 3.x | Data preprocessing |
| NumPy | 2.x | Numerical computation |
| Plotly | 6.x | Interactive data visualizations |
| Joblib | 1.5.x | Model serialization |

**Cloud Services**

| Service | Purpose |
|---|---|
| Streamlit Cloud | App deployment & hosting |
| GitHub | Source control & CI/CD trigger |

---

## ✨ Features

**🔍 Risk Assessment Tab**
- 3-step guided clinical input form (Demographics → Cardiovascular Vitals → Clinical Indicators)
- Real-time KNN prediction with probability score
- Color-coded result banner (red = high risk, green = low risk)
- Animated gauge chart showing risk percentage
- Feature contribution bar chart (top 6 biomarkers)
- Medical disclaimer section

**📊 Data Insights Tab**
- Dataset-level statistics (918 patients, disease split, avg age, avg max HR)
- Age distribution by heart disease status (histogram)
- Chest pain type breakdown (grouped bar)
- Cholesterol vs Max Heart Rate scatter plot
- ST Slope distribution (stacked bar)
- Raw dataset preview (first 20 records)

---

## 🤖 ML Pipeline

```
Raw Dataset (heart.csv)
        │
        ▼
┌───────────────────────┐
│  Data Preprocessing   │
│  pd.get_dummies()     │
│  drop_first=True      │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Train / Test Split   │
│  80% train, 20% test  │
│  stratify=y           │
│  random_state=42      │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Feature Scaling     │
│   StandardScaler      │
│   fit on train only   │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Model Benchmarking   │
│  Logistic Regression  │
│  K-Nearest Neighbors  │
│  Naive Bayes          │
│  Decision Tree        │
│  SVM (RBF kernel)     │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   KNN Selected        │
│   Accuracy: 88.6%     │
│   F1 Score:  90%      │
└──────────┬────────────┘
           │
           ▼
  Streamlit Web App
  (Live Prediction)
```

---

## 📊 Model Comparison

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 87.5% | 88.78% |
| **K-Nearest Neighbors** | **88.6%** | **90%** |
| Naive Bayes | 86.96% | 87.88% |
| Decision Tree | 87.5% | 76.04% |
| SVM (RBF kernel) | 85.41% | 88.04% |

> KNN was selected for its highest accuracy and F1 score combination on the UCI Heart Failure dataset.

---

## 📁 Project Structure

```
cardio-ai/
│
├── app.py                  # Main Streamlit app (UI + ML pipeline)
├── heart.csv               # UCI Heart Failure dataset (918 records)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

> **Note:** No `.pkl` files needed. The model trains on startup from `heart.csv` using `@st.cache_resource` — version-safe across all Python/sklearn environments.

---

## 🧬 Dataset

**Source:** UCI Heart Failure Prediction Dataset

| Property | Value |
|---|---|
| Records | 918 patients |
| Features | 11 clinical biomarkers |
| Target | HeartDisease (0 = No, 1 = Yes) |
| Disease split | ~55% positive, ~45% negative |

**Input Features**

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Patient age in years |
| Sex | Categorical | M / F |
| ChestPainType | Categorical | ATA / NAP / TA / ASY |
| RestingBP | Numerical | Resting blood pressure (mm Hg) |
| Cholesterol | Numerical | Serum cholesterol (mg/dL) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dL |
| RestingECG | Categorical | Normal / ST / LVH |
| MaxHR | Numerical | Maximum heart rate achieved |
| ExerciseAngina | Binary | Exercise-induced angina Y / N |
| Oldpeak | Numerical | ST depression value |
| ST_Slope | Categorical | Up / Flat / Down |

---

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/bishwajitpattanaik/cardio-ai.git
cd cardio-ai
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

> App runs on `http://localhost:8501`

**4. Open in browser**

Visit `http://localhost:8501` to use CardioAI locally.

> Note: `heart.csv` must be in the same directory as `app.py`. The model trains automatically on first launch.

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
```

---

## 🚀 Deployment

Deployed on **Streamlit Cloud** via GitHub integration.

| Layer | Platform | URL |
|---|---|---|
| App | Streamlit Cloud | [cardio-ai-bishwajitpattanaik.streamlit.app](https://cardio-ai-bishwajitpattanaik.streamlit.app) |
| Source | GitHub | [github.com/bishwajitpattanaik/cardio-ai](https://github.com/bishwajitpattanaik/cardio-ai) |

**How it works:**
- Every push to the `main` branch triggers an automatic redeploy on Streamlit Cloud.
- The app reads `heart.csv`, trains the KNN model at startup, and caches it for the session using `@st.cache_resource`.
- No pickle files — the pipeline is fully reproducible across Python versions.

---

## 👤 Author

Built with ❤️ by **Bishwajit Pattanaik**

- 🔗 GitHub: [github.com/bishwajitpattanaik](https://github.com/bishwajitpattanaik)
- 💼 LinkedIn: [linkedin.com/in/bishwajit-pattanaik-717818320](https://www.linkedin.com/in/bishwajit-pattanaik-717818320/)

---

## 🛠️ Support

For issues or questions, open an issue in the repository — [github.com/bishwajitpattanaik/cardio-ai/issues](https://github.com/bishwajitpattanaik/cardio-ai/issues)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
