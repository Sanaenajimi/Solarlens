# ☀️ SolarLens — Projet complet (reconstitué)

Ce zip contient **tout le code** du projet : entraînement du modèle,
app Streamlit de démo, et version web statique pour GitHub Pages.

⚠️ **Ce qui manque** (normal, ça ne se transmet pas dans un zip de code) :
le modèle déjà entraîné (`checkpoints/solarlens_best.pth`). Il faut
ré-entraîner — mais avec un GPU gratuit (Google Colab), ça prend
15-20 minutes, pas des heures.

## 📁 Contenu

```
SolarLens_complet/
├── train.py                 ← entraîne le modèle (à lancer en premier)
├── export_onnx.py           ← convertit le modèle entraîné pour le web
├── strip_checkpoint.py      ← allège le fichier модèle (optionnel)
├── verify_onnx.py           ← vérifie que l'export ONNX fonctionne
├── app.py                   ← app Streamlit de démo (test en local)
├── requirements.txt
├── models/
│   └── solarlens_model.py   ← architecture ResNet-50 + Squeeze-Excitation
├── utils/                   ← modules utilisés par app.py
├── assets/                  ← images du hero (maison 3D)
└── web/                     ← à déployer sur GitHub Pages (voir plus bas)
    ├── index.html
    ├── css/style.css
    ├── js/app.js
    └── assets/
```

## 🚀 Étape 1 — Ré-entraîner le modèle (Google Colab, GPU gratuit)

1. Va sur **colab.research.google.com** → nouveau notebook
2. **Runtime → Change runtime type → GPU (T4)**
3. Uploade tout ce dossier dans Colab (ou juste `train.py` + `models/`)
4. Récupère le dataset :
   ```python
   !pip install kaggle
   # Upload ton kaggle.json (Kaggle > Account > Create New API Token)
   !kaggle datasets download -d pythonafroz/solar-panel-images
   !unzip solar-panel-images.zip -d data/pv-panel-defect/
   ```
5. Lance l'entraînement :
   ```python
   !python train.py --data_dir ./data/pv-panel-defect
   ```
   Sur GPU T4 : ~15-20 secondes par époque (au lieu de 3-5 minutes sur CPU).
   Le script s'arrête automatiquement (early stopping) autour de l'époque 70-80.
6. **Télécharge immédiatement** `checkpoints/solarlens_best.pth` sur ton disque
   (clique-droit dans le panneau fichiers de Colab → Download). L'environnement
   Colab est effacé à la fermeture, donc ne saute pas cette étape.

## 🖥️ Étape 2 (optionnel) — Tester l'app Streamlit en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Étape 3 — Déployer la version web sur GitHub Pages

Le dossier `web/` est déjà prêt (design chrome/silver, 4 onglets, caméra live,
inférence 100% dans le navigateur via ONNX Runtime Web).

### 3a. Exporter le modèle entraîné en ONNX

```bash
python export_onnx.py --checkpoint checkpoints/solarlens_best.pth --output checkpoints/solarlens_web.onnx
```
Ça produit `solarlens_web.onnx` (~100 Mo) et `solarlens_web_labels.json`.

### 3b. Pousser `web/` sur un repo GitHub

```bash
cd web
git init
git add .
git commit -m "SolarLens web"
git branch -M main
git remote add origin https://github.com/TON-USERNAME/solarlens.git
git push -u origin main
```

### 3c. Créer une GitHub Release avec le modèle attaché

Sur ton repo GitHub → **Releases** → **Create a new release** :
- Tag : `v1.0`
- Dans **Attach binaries**, glisse les 2 fichiers de l'étape 3a
- **Publish release**

GitHub te donne une URL stable du type :
```
https://github.com/TON-USERNAME/solarlens/releases/download/v1.0/solarlens_web.onnx
```

### 3d. Brancher ces URLs et activer Pages

Ouvre `web/js/app.js`, remplace les 2 premières constantes par tes vraies URLs :
```js
const MODEL_URL = "https://github.com/TON-USERNAME/solarlens/releases/download/v1.0/solarlens_web.onnx";
const LABELS_URL = "https://github.com/TON-USERNAME/solarlens/releases/download/v1.0/solarlens_web_labels.json";
```

Commit, push, puis sur GitHub : **Settings → Pages → Source: Deploy from a
branch → main → / (root) → Save**.

Ton site sera en ligne à : `https://TON-USERNAME.github.io/solarlens/`

## 🧪 Tester la version web en local avant de déployer

```bash
cd web
python -m http.server 8000
```
Ouvre `http://localhost:8000` (jamais en double-cliquant sur le fichier —
la caméra et le chargement du modèle ne marchent qu'avec un vrai serveur HTTP).

## 🐛 Problèmes fréquents

| Symptôme | Solution |
|---|---|
| Entraînement très lent | Vérifie que le runtime Colab est bien en GPU (Runtime → Change runtime type) |
| "Impossible de charger le modèle" (web) | Vérifie les 2 URLs dans `app.js` |
| 404 sur le fichier .onnx | Vérifie que la Release est "Published", pas en brouillon |
| Erreur CORS en local | Utilise `python -m http.server`, jamais `file://` |
