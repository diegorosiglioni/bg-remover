# Guida GPU — abilitare la Phase 2 (NVIDIA / CUDA)

Questa guida spiega, passo per passo, come modificare il progetto per usarlo con
una scheda video NVIDIA invece che in CPU. È la "Phase 2" descritta in
[`ARCHITECTURE.md`](ARCHITECTURE.md) §8.

> **In breve:** non è un interruttore. Oggi i componenti GPU sono *stub* che
> lanciano `NotImplementedError`. Per attivare la GPU servono 4 cose:
> **1)** hardware NVIDIA, **2)** installare le dipendenze torch, **3)** scrivere
> il codice dei 2 stub + 1 piccola modifica alla pipeline, **4)** cambiare la
> config. Mettere solo `engine=torch` senza i passi 2–3 fa crashare.

---

## 0. Cosa cambia e cosa NO

| Componente | Phase 1 (CPU, ora) | Phase 2 (GPU) |
|---|---|---|
| Segmentazione | `RembgSegmenter` (ONNX) | `TorchBiRefNetSegmenter` (torch) |
| Matting refine | `PyMattingRefiner` | `ViTMatteRefiner` |
| Defringe / trimap / composite | invariati | **identici** |
| Dipendenze | rembg, onnxruntime | + torch, transformers, ... |

I moduli che eliminano l'alone (`trimap.py`, `defringe.py`, `composite.py`) NON
si toccano. Si sostituiscono solo i 2 "seam": il segmentatore e il refiner.

---

## 1. Requisiti hardware

- GPU **NVIDIA** (consigliata ≥ 8 GB VRAM per BiRefNet HR a 2048²; 6 GB ok con
  risoluzione più bassa).
- Driver NVIDIA aggiornati + toolkit CUDA compatibile con la build di torch.
- Senza GPU NVIDIA, la Phase 2 non si applica: resta la CPU (Phase 1).

Verifica che il sistema veda la GPU:

```powershell
nvidia-smi
```

Se il comando stampa la tabella con la tua GPU e la versione CUDA driver, sei a
posto. Se "non riconosciuto", installa prima i driver NVIDIA.

---

## 2. Installare le dipendenze GPU

Queste **non** sono in `requirements.txt` (che resta CPU-only). Crea un file
separato `requirements-gpu.txt`:

```text
# Phase 2 — GPU (torch + transformers). Da installare SOPRA requirements.txt.
torch
torchvision
transformers
huggingface-hub
accelerate
```

Installa torch dalla index CUDA giusta (esempio CUDA 12.1 → `cu121`; adatta alla
tua versione vista con `nvidia-smi`):

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

Verifica che torch veda la GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Deve stampare `True ...`. Se stampa `False`, la build di torch è quella CPU:
disinstalla e reinstalla dalla index `cuXXX` corretta.

---

## 3. Scrivere il codice (la parte vera)

### 3a. Segmentatore — `app/segmentation.py`

Sostituisci il corpo dello stub `TorchBiRefNetSegmenter` (oggi alle righe ~147)
con un'implementazione che carica BiRefNet HR una volta sola e gira in FP16.

```python
class TorchBiRefNetSegmenter:
    """BiRefNet HR su torch/CUDA (Phase 2)."""

    def __init__(self, config: Config) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        self._config = config
        self._device = config.device  # "cuda"
        self._model = AutoModelForImageSegmentation.from_pretrained(
            config.model_name,            # es. "ZhengPeng7/BiRefNet_HR"
            trust_remote_code=True,
        )
        self._model.to(self._device).eval()
        self._torch = torch
        # input quadrato del modello (HR = 2048)
        self._size = 2048

    def predict(self, image: np.ndarray) -> np.ndarray:
        _validate_image(image)
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        h, w = image.shape[:2]
        tf = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            transforms.Resize((self._size, self._size), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),  # ImageNet
        ])
        x = tf(image).unsqueeze(0).to(self._device)

        with torch.no_grad(), torch.autocast(self._device, dtype=torch.float16):
            pred = self._model(x)[-1].sigmoid()          # (1,1,S,S) in [0,1]
        pred = F.interpolate(pred, size=(h, w), mode="bilinear",
                             align_corners=False)
        alpha = pred[0, 0].float().clamp(0, 1).cpu().numpy().astype(np.float32)
        return alpha
```

Output: `(H, W) float32` in `[0,1]`, soft — esattamente quello che si aspetta il
resto della pipeline (stesso contratto di `RembgSegmenter.predict`).

### 3b. Refiner — `app/matting.py`

Sostituisci il corpo dello stub `ViTMatteRefiner` (oggi alle righe ~71):

```python
class ViTMatteRefiner:
    """ViTMatte su torch/CUDA (Phase 2)."""

    _MODEL_ID = "hustvl/vitmatte-small-composition-1k"

    def __init__(self, config: Config) -> None:
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self._config = config
        self._device = config.device
        self._processor = VitMatteImageProcessor.from_pretrained(self._MODEL_ID)
        self._model = (
            VitMatteForImageMatting.from_pretrained(self._MODEL_ID)
            .to(self._device)
            .eval()
        )
        self._torch = torch

    def refine(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        _validate(image, trimap)
        import torch

        h, w = image.shape[:2]
        # il processor pad-a a multipli di size_divisor: poi si ricroppa a HxW
        inputs = self._processor(images=image, trimaps=trimap, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            alphas = self._model(**inputs).alphas      # (1,1,H_pad,W_pad)
        alpha = alphas[0, 0, :h, :w].float().clamp(0, 1).cpu().numpy()
        return alpha.astype(np.float32)
```

> Nota: il `VitMatteImageProcessor` vuole il trimap con valori in `[0,1]`
> (`0`=bg, `0.5`=unknown, `1`=fg), che è esattamente ciò che produce
> `trimap.py`. Se la versione di `transformers` lo vuole in `{0,128,255}`,
> moltiplica per 255 prima di passarlo.

### 3c. Collegare il refiner nella pipeline — `app/pipeline.py`

**Importante:** oggi `pipeline.py:113` usa SEMPRE `PyMattingRefiner`, anche con
`engine=torch`. Va reso selezionabile, altrimenti ViTMatte non viene mai usato.

Aggiungi un builder in `app/matting.py`:

```python
def build_refiner(config: Config):
    if config.engine == "torch":
        return ViTMatteRefiner(config)
    return PyMattingRefiner(config)
```

Poi in `app/pipeline.py` cambia l'import e la riga 113:

```python
# prima
from app.matting import PyMattingRefiner
...
self._refiner: MattingRefiner = PyMattingRefiner(config)

# dopo
from app.matting import build_refiner
...
self._refiner: MattingRefiner = build_refiner(config)
```

Il segmentatore invece è già selezionato da `build_segmenter` via
`config.engine` (`segmentation.py:173`) — non serve toccarlo.

---

## 4. Cambiare la configurazione

La GPU richiede la modalità `modular` (la one-shot esiste solo per rembg).
Imposta le variabili d'ambiente `BGR_` prima di lanciare:

```powershell
$env:BGR_ENGINE = "torch"
$env:BGR_DEVICE = "cuda"
$env:BGR_PIPELINE_MODE = "modular"
$env:BGR_MODEL_NAME = "ZhengPeng7/BiRefNet_HR"
```

Oppure in un file `.env` nella root del progetto:

```text
BGR_ENGINE=torch
BGR_DEVICE=cuda
BGR_PIPELINE_MODE=modular
BGR_MODEL_NAME=ZhengPeng7/BiRefNet_HR
```

In codice, equivalente:

```python
from app import Config, remove_background

cfg = Config(engine="torch", device="cuda",
             pipeline_mode="modular", model_name="ZhengPeng7/BiRefNet_HR")
rgba = remove_background("foto.jpg", cfg)
```

---

## 5. Eseguire

Identico a prima — cambia solo che ora gira su GPU:

```bash
python -m app.cli input output --mode modular
```

Il primo avvio scarica i pesi da Hugging Face (alcuni GB) e li mette in cache;
le run successive sono offline.

---

## 6. Alternativa leggera: onnxruntime-gpu (senza torch)

Se vuoi solo accelerare il modello ONNX attuale senza riscrivere gli stub, puoi
restare su rembg ma su GPU:

```powershell
pip uninstall onnxruntime
pip install onnxruntime-gpu
$env:BGR_DEVICE = "cuda"   # segmentation.py usa già CUDAExecutionProvider
```

Il codice in `segmentation.py:31` (`_PROVIDERS["cuda"]`, dict che parte a
riga 29) prova già `CUDAExecutionProvider`. Questo dà un boost alla
segmentazione ONNX ma **non**
porta la qualità di BiRefNet HR + ViTMatte (Phase 2 vera). Buon compromesso se
non vuoi scrivere codice.

---

## 7. Wrapper FastAPI (opzionale, Phase 2 completa)

Per esporre il tutto come servizio (passo 5 di `ARCHITECTURE.md` §8): un'app
FastAPI carica i modelli una volta in un handler `lifespan` ed espone
`POST /remove` (upload → PNG RGBA) e `GET /health`, chiamando lo *stesso*
`remove_background`. Dockerfile su base `nvidia/cuda`, richiede NVIDIA Container
Toolkit. Non incluso in questo repo.

---

## Checklist riassuntiva

- [ ] `nvidia-smi` mostra la GPU
- [ ] torch installato dalla index `cuXXX` giusta; `torch.cuda.is_available()` → `True`
- [ ] `requirements-gpu.txt` creato e installato
- [ ] `TorchBiRefNetSegmenter` implementato (`segmentation.py`)
- [ ] `ViTMatteRefiner` implementato (`matting.py`)
- [ ] `build_refiner` aggiunto e collegato in `pipeline.py`
- [ ] config: `engine=torch`, `device=cuda`, `pipeline_mode=modular`, `model_name=ZhengPeng7/BiRefNet_HR`
- [ ] prima run scarica i pesi, poi gira su GPU
