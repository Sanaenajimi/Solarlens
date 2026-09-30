/**
 * SolarLens — Inference client-side avec ONNX Runtime Web.
 *
 * Aucun serveur : le modèle ResNet-50 + SE (exporté en ONNX) est
 * téléchargé depuis Hugging Face Hub puis exécuté directement dans
 * le navigateur via WebAssembly. Les images de l'utilisateur ne
 * quittent jamais son appareil.
 */

// ─────────────────────────────────────────────────────────────
// CONFIGURATION — remplace par l'URL de ta GitHub Release
// (Settings → onglet "Releases" de ton repo → uploader les 2 fichiers
// générés par export_onnx.py comme "assets" de la release)
// ─────────────────────────────────────────────────────────────
const MODEL_URL = "model/solarlens_web.onnx";
const LABELS_URL = "model/solarlens_web_labels.json";
// Fallback si le JSON des labels n'a pas pu être chargé
const FALLBACK_CLASS_NAMES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered"
];

// Normalisation ImageNet (doit être identique à l'entraînement PyTorch)
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const INPUT_SIZE = 224;

// ─────────────────────────────────────────────────────────────
// COPIE MÉTIER — même contenu que la version Streamlit
// ─────────────────────────────────────────────────────────────
const CLASS_COPY = {
    "Clean": {
        label: "Tout va bien", severity: "low",
        severityCopy: "Aucun problème — entretien de routine",
        summary: "Votre panneau est en excellent état. Rien à signaler pour le moment.",
        advice: "Surveillez simplement votre production mensuelle. Un rinçage annuel suffit en région sèche.",
    },
    "Dusty": {
        label: "Poussière accumulée", severity: "medium",
        severityCopy: "À traiter prochainement",
        summary: "De la poussière et des salissures recouvrent la surface et empêchent la lumière d'atteindre les cellules.",
        advice: "Rincez à l'eau claire un matin frais — vous récupérerez généralement 10 à 15 % de production.",
    },
    "Bird-drop": {
        label: "Fientes d'oiseaux", severity: "medium",
        severityCopy: "À traiter prochainement",
        summary: "Des fientes localisées font de l'ombre sur une partie du panneau et peuvent créer un point chaud à terme.",
        advice: "Un rinçage doux à l'eau maintenant. Si le problème revient, installez discrètement des dissuadeurs d'oiseaux.",
    },
    "Snow-Covered": {
        label: "Recouvert de neige", severity: "low",
        severityCopy: "Aucun problème — entretien de routine",
        summary: "La neige bloque temporairement la lumière du soleil. La production reprendra dès la fonte.",
        advice: "Laissez fondre naturellement si possible. Ne dégagez à la main que si la neige persiste plusieurs jours.",
    },
    "Electrical-damage": {
        label: "Défaut électrique", severity: "high",
        severityCopy: "Appelez un professionnel",
        summary: "Signes d'un défaut électrique — probablement un point chaud, une diode de bypass ou une dégradation. Cela peut s'aggraver rapidement.",
        advice: "Ne touchez pas au panneau. Contactez un installateur certifié sous 48 h pour une inspection sur site.",
    },
    "Physical-Damage": {
        label: "Dommage physique", severity: "high",
        severityCopy: "Appelez un professionnel",
        summary: "Le panneau présente un dommage physique : fissure, verre cassé ou délamination. L'humidité peut s'infiltrer et aggraver le problème.",
        advice: "Photographiez les dégâts sous plusieurs angles. Contactez votre installateur cette semaine pour vérifier la garantie.",
    },
};

const SEVERITY_PALETTE = {
    low:    { main: "var(--sev-low)",    bg: "var(--sev-low-bg)",    border: "var(--sev-low-border)",    chip: "var(--sev-low-chip)",    icon: "✓" },
    medium: { main: "var(--sev-medium)", bg: "var(--sev-medium-bg)", border: "var(--sev-medium-border)", chip: "var(--sev-medium-chip)", icon: "⚠" },
    high:   { main: "var(--sev-high)",   bg: "var(--sev-high-bg)",   border: "var(--sev-high-border)",   chip: "var(--sev-high-chip)",   icon: "⚡" },
};

// ─────────────────────────────────────────────────────────────
// ÉTAT GLOBAL
// ─────────────────────────────────────────────────────────────
let ortSession = null;
let classNames = FALLBACK_CLASS_NAMES;
let modelReady = false;
let cameraStream = null;

// ─────────────────────────────────────────────────────────────
// CHARGEMENT DU MODÈLE (au démarrage de la page)
// ─────────────────────────────────────────────────────────────
async function loadModel() {
    const statusDot = document.getElementById("model-status-dot");
    const statusText = document.getElementById("model-status-text");

    try {
        statusText.textContent = "Téléchargement du modèle (~100 Mo)...";

        // ONNX Runtime Web : on utilise le backend WASM (fonctionne partout,
        // pas besoin de WebGPU/WebGL même si c'est plus lent qu'un serveur).
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

        // Charger les labels et le modèle en parallèle
        const [labelsResp, session] = await Promise.all([
            fetch(LABELS_URL).then(r => r.ok ? r.json() : null).catch(() => null),
            ort.InferenceSession.create(MODEL_URL, {
                executionProviders: ["wasm"],
                graphOptimizationLevel: "all",
            }),
        ]);

        if (labelsResp && Array.isArray(labelsResp)) {
            classNames = labelsResp;
        }
        ortSession = session;
        modelReady = true;

        statusDot.classList.add("ready");
        statusText.textContent = "Modèle chargé — prêt pour l'analyse";
        enableInferenceUI();
    } catch (err) {
        console.error("Erreur de chargement du modèle :", err);
        statusDot.classList.add("error");
        statusText.textContent = "Impossible de charger le modèle. Vérifiez votre connexion et réessayez.";
    }
}

function enableInferenceUI() {
    document.querySelectorAll(".requires-model").forEach(el => {
        el.disabled = false;
    });
}

// ─────────────────────────────────────────────────────────────
// PRÉTRAITEMENT D'IMAGE — reproduit exactement les transforms PyTorch
// (Resize 224x224 → ToTensor → Normalize ImageNet)
// ─────────────────────────────────────────────────────────────
function preprocessImage(imageElement) {
    const canvas = document.createElement("canvas");
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext("2d");

    // Resize direct à 224x224 (équivalent à transforms.Resize((224,224)) en mode val)
    ctx.drawImage(imageElement, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = imageData; // Uint8ClampedArray RGBA, longueur = 224*224*4

    // Le modèle attend un tenseur CHW (Channel, Height, Width), normalisé.
    // C'est l'ordre PyTorch standard, différent de l'ordre HWC du canvas.
    const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const pixelCount = INPUT_SIZE * INPUT_SIZE;

    for (let i = 0; i < pixelCount; i++) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;

        // Normalisation ImageNet : (valeur - mean) / std, canal par canal
        chw[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];                    // canal R
        chw[pixelCount + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];       // canal G
        chw[2 * pixelCount + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];   // canal B
    }

    return new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ─────────────────────────────────────────────────────────────
// INFÉRENCE
// ─────────────────────────────────────────────────────────────
async function runInference(imageElement) {
    if (!modelReady) {
        throw new Error("Le modèle n'est pas encore chargé.");
    }

    const inputTensor = preprocessImage(imageElement);
    const feeds = { image: inputTensor };
    const results = await ortSession.run(feeds);

    // Le graphe ONNX exporté inclut déjà le softmax (voir export_onnx.py),
    // donc results.probabilities contient directement des probabilités
    // entre 0 et 1 qui somment à 1 — pas besoin de softmax en JS.
    const outputName = ortSession.outputNames[0];
    const probs = Array.from(results[outputName].data);

    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    return {
        predClass: classNames[maxIdx],
        confidence: probs[maxIdx] * 100,
        probs: classNames.map((name, i) => [name, probs[i] * 100]),
    };
}

// ─────────────────────────────────────────────────────────────
// RENDU DU DIAGNOSTIC (langage clair, couleurs par sévérité)
// ─────────────────────────────────────────────────────────────
function renderDiagnosisCard(result, containerId) {
    const container = document.getElementById(containerId);
    const copy = CLASS_COPY[result.predClass] || CLASS_COPY["Clean"];
    const palette = SEVERITY_PALETTE[copy.severity];

    const urgentBadge = copy.severity === "high"
        ? `<span class="urgent-badge" style="background:${palette.main};">⚡ Urgent</span>`
        : "";
    const pulseClass = copy.severity === "high" ? "pulse-alert" : "";

    container.innerHTML = `
        <div class="chapter-card animate-rise-in ${pulseClass}"
             style="border-left:6px solid ${palette.border}; background:linear-gradient(135deg, ${palette.bg}, var(--card));">
            <div class="diag-header" style="color:${palette.main};">
                <span style="font-size:16px;">${palette.icon}</span>
                <span>NOTRE DIAGNOSTIC</span>
            </div>
            <div class="diag-verdict" style="color:${palette.main};">${copy.label}${urgentBadge}</div>
            <div class="diag-meta" style="color:${palette.main};">
                ${copy.severityCopy} · confiance : ${result.confidence.toFixed(1)}%
            </div>
            <p class="diag-summary">${copy.summary}</p>
            <div class="diag-advice" style="background:${palette.chip}; border-left:3px solid ${palette.border};">
                <div class="diag-advice-label" style="color:${palette.main};">CE QUE VOUS POUVEZ FAIRE</div>
                <div class="diag-advice-text">${copy.advice}</div>
            </div>
        </div>
    `;

    // Alerte "prédiction incertaine" si le modèle hésite entre 2 classes
    const sorted = [...result.probs].sort((a, b) => b[1] - a[1]);
    const gap = sorted[0][1] - sorted[1][1];
    if (result.confidence < 85 || gap < 15) {
        const warning = document.createElement("div");
        warning.className = "warning-box orange animate-rise-in";
        warning.innerHTML = `
            <span style="font-size:20px;">⚠️</span>
            <div>
                <div class="warning-box-title" style="color:#f59e0b;">Prédiction incertaine</div>
                <div class="warning-box-text" style="color:#92400e;">
                    L'IA n'est pas hautement confiante sur ce diagnostic. Ce résultat peut refléter
                    un cas hors du domaine d'entraînement (angle inhabituel, environnement très présent, etc.).
                    <span style="color:var(--muted-foreground);">Une inspection humaine est recommandée pour confirmation.</span>
                </div>
            </div>
        `;
        container.prepend(warning);
    }
}

function renderProbabilityBars(result, containerId) {
    const container = document.getElementById(containerId);
    const copy = CLASS_COPY[result.predClass] || CLASS_COPY["Clean"];
    const topColor = SEVERITY_PALETTE[copy.severity].main;

    const sorted = [...result.probs].sort((a, b) => b[1] - a[1]);
    const bars = sorted.map(([label, value], i) => {
        const isTop = i === 0;
        const fill = isTop ? topColor : "linear-gradient(90deg, #cbd5e1, #94a3b8)";
        const color = isTop ? topColor : "var(--foreground)";
        const weight = isTop ? "700" : "400";
        return `
            <div class="prob-row">
                <div class="prob-row-header">
                    <span style="color:${color};font-weight:${weight};">${label.replace("-", " ")}</span>
                    <span class="prob-row-value" style="color:${color};font-weight:${weight};">${value.toFixed(1)}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:${value.toFixed(1)}%; background:${fill};"></div>
                </div>
            </div>
        `;
    }).join("");

    container.innerHTML = `
        <div class="chapter-card animate-rise-in" style="margin-top:16px;">
            <div class="diag-header">PROBABILITÉS PAR CLASSE</div>
            <div style="margin-top:20px;">${bars}</div>
        </div>
    `;
}

function showLoading(containerId, message) {
    document.getElementById(containerId).innerHTML = `
        <div class="loading-row"><span class="spinner"></span> ${message}</div>
    `;
}

function showError(containerId, message) {
    document.getElementById(containerId).innerHTML = `
        <div class="warning-box orange">
            <span style="font-size:20px;">⚠️</span>
            <div>
                <div class="warning-box-title" style="color:#f59e0b;">Erreur</div>
                <div class="warning-box-text" style="color:#92400e;">${message}</div>
            </div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────
// TABS (navigation single-page)
// ─────────────────────────────────────────────────────────────
function initTabs() {
    const buttons = document.querySelectorAll(".tab-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
            document.getElementById(target).classList.add("active");

            if (target === "page-camera") {
                startCamera();
            } else {
                stopCamera();
            }
        });
    });
}

// ─────────────────────────────────────────────────────────────
// UPLOAD (drag & drop + click)
// ─────────────────────────────────────────────────────────────
function initUpload() {
    const zone = document.getElementById("upload-zone");
    const input = document.getElementById("file-input");

    zone.addEventListener("click", () => input.click());

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Merci de choisir un fichier image (JPG ou PNG).");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = async () => {
            document.getElementById("scan-preview-wrap").style.display = "block";
            document.getElementById("scan-preview").src = e.target.result;
            document.getElementById("scan-results").style.display = "block";

            showLoading("scan-diagnosis", "Analyse du panneau…");
            document.getElementById("scan-probs").innerHTML = "";

            try {
                const result = await runInference(img);
                renderDiagnosisCard(result, "scan-diagnosis");
                renderProbabilityBars(result, "scan-probs");
            } catch (err) {
                console.error(err);
                showError("scan-diagnosis", "L'analyse a échoué. " + err.message);
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────────────────────
// CAMÉRA (getUserMedia + capture d'une frame)
// ─────────────────────────────────────────────────────────────
async function startCamera() {
    const video = document.getElementById("camera-video");
    if (cameraStream) return; // déjà démarrée

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 1280 } },
            audio: false,
        });
        video.srcObject = cameraStream;
        document.getElementById("camera-capture-btn").disabled = false;
    } catch (err) {
        console.error("Erreur caméra :", err);
        document.getElementById("camera-error").style.display = "block";
        document.getElementById("camera-error").textContent =
            "Impossible d'accéder à la caméra. Vérifiez les permissions du navigateur, " +
            "ou utilisez l'onglet 'Analyser une photo' à la place.";
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

function initCamera() {
    const captureBtn = document.getElementById("camera-capture-btn");
    const video = document.getElementById("camera-video");
    const canvas = document.getElementById("camera-canvas");

    captureBtn.addEventListener("click", async () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        document.getElementById("camera-results").style.display = "block";
        showLoading("camera-diagnosis", "Analyse en cours…");
        document.getElementById("camera-probs").innerHTML = "";

        const img = new Image();
        img.onload = async () => {
            try {
                const result = await runInference(img);
                renderDiagnosisCard(result, "camera-diagnosis");
                renderProbabilityBars(result, "camera-probs");
            } catch (err) {
                console.error(err);
                showError("camera-diagnosis", "L'analyse a échoué. " + err.message);
            }
        };
        img.src = canvas.toDataURL("image/jpeg");
    });
}

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initUpload();
    initCamera();
    loadModel();
});
