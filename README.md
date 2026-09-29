# ☀️ SolarLens

Ce zip contient **tout le code** du projet : entraînement du modèle,
app Streamlit de démo, et version web statique pour GitHub Pages.

## 📁 Contenu

```
SolarLens_complet/
├── train.py                 ← entraîne le modèle 
├── export_onnx.py           ← convertit le modèle entraîné pour le web
├── strip_checkpoint.py      ← allège le fichier моdèle 
├── verify_onnx.py           ← vérifie que l'export ONNX fonctionne
├── app.py                   ← app Streamlit de démo (test en local)
├── requirements.txt
├── models/
│   └── solarlens_model.py   ← architecture ResNet-50 + Squeeze-Excitation
├── utils/                   ← modules utilisés par app.py
├── assets/                  ← images du hero (maison 3D)
└── web/                    
    ├── index.html
    ├── css/style.css
    ├── js/app.js
    └── assets/
```
## Comment Refaire le projet par vous même ?
## 🚀 Étape 1 — Ré-entraîner le modèle (Google Colab, GPU gratuit)

1. pour réentrainer le modèle vous pouvez utiliser sur **colab.research.google.com** → nouveau notebook
2. **Runtime → Change runtime type → GPU (T4)**
3. Uploadez tout ce dossier dans Colab (ou juste `train.py` + `models/`)
4. Récupèrez le dataset :
   ```python
   !pip install kaggle
   # Uploadez votre kaggle.json (Kaggle > Account > Create New API Token)
   !kaggle datasets download -d pythonafroz/solar-panel-images
   !unzip solar-panel-images.zip -d data/pv-panel-defect/
   ```
5. Lancez l'entraînement :
   ```python
   !python train.py --data_dir ./data/pv-panel-defect
   ```
   Sur GPU T4 : ~15-20 secondes par époque (au lieu de 3-5 minutes sur CPU).
   Le script s'arrête automatiquement (early stopping) autour de l'époque 70-80.
6. **Téléchargez immédiatement** `checkpoints/solarlens_best.pth` sur votre disque
   (clique-droit dans le panneau fichiers de Colab → Download). L'environnement
   Colab est effacé à la fermeture, donc ne saute pas cette étape.

## 🖥️ Étape 2 (optionnel) — Tester l'app Streamlit en local

```bash
pip install -r requirements.txt
streamlit run app.py
``
###  Créer une GitHub Release avec le modèle attaché

Sur votre repo GitHub → **Releases** → **Create a new release** :
- Tag : `v1.0`
- Dans **Attach binaries**, glisse les 2 fichiers de l'étape 3a
- **Publish release**

GitHub vous donne une URL stable du type :
```
https://github.com/TON-USERNAME/solarlens/releases/download/v1.0/solarlens_web.onnx que vous devez utiliser pour remplacez les urls dans js/app.js
```
