# 🎵 Music Genre Classification using Audio Features

Classify music tracks into genres using **precomputed audio features** from the FMA dataset. This end-to-end ML pipeline loads features, trains a model, and predicts genres — all without needing audio files.

---

## 🚀 Project Overview

- 🔍 Uses `features.csv` (MFCCs, spectral features)
- 🧠 Trains a **Random Forest Classifier** with Scikit-learn
- 📊 Evaluates model with accuracy & F1-score
- 💾 Saves model (`.pkl`) for later use
- ✅ No need for raw audio or spectrograms

---

## 🗂️ Dataset Required

- `fma_metadata/features.csv`
- `fma_metadata/tracks.csv`

Download from: [FMA GitHub](https://github.com/mdeff/fma)

---

🧠 Training Steps (Colab Compatible)
python
Copy
Edit
# 1. Load Data
import pandas as pd, numpy as np
features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0,1,2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0,1])
genres = tracks['track']['genre_top'].dropna()

# 2. Clean & Align
features = features.loc[genres.index].dropna()
genres = genres.loc[features.index]

# 3. Encode Labels
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(genres)

# 4. Prepare Features
X = features.copy()
X.columns = ['_'.join(col).strip() for col in X.columns.values]
X = X.values

# 5. Train Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save Model
import joblib
joblib.dump(model, 'genre_rf_model.pkl')
🧪 Predict Genres with Saved Model
python
Copy
Edit
import pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder

# Load CSV with same format as features.csv
df = pd.read_csv('your_new_features.csv', index_col=0, header=[0,1,2])
df.columns = ['_'.join(col).strip() for col in df.columns.values]
X_new = df.values

# Load model
model = joblib.load('genre_rf_model.pkl')
predictions = model.predict(X_new)

# Map predictions back to genre names (if label encoder saved)
print("Predicted genres:", predictions)
---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn joblib

📝 License
MIT License © 2025
Built with ❤️ using Scikit-learn and the FMA dataset

📬 Credits
Dataset: FMA by Michaël Defferrard

Model: Scikit-learn Random Forest

yaml
Copy
Edit

---

Let me know if you also want:

- `requirements.txt`
- A ready-to-run `predict_genre.py` file
- GitHub repo name/description suggestion

Would you like me to generate those files as well?
