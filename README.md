# Physiognomy

An experimental project exploring facial geometry and text generation from structured data. The repository includes an image dataset, feature tables, a data-processing notebook, and a Gradio interface for reading reports.

The project pairs measurements extracted from images with text labels supplied by an external service. Labels about personality, careers, relationships, future outcomes, and wealth have not been independently validated as facts about a person. Do not use these interpretations to assess people or make consequential decisions.

## How it works

```text
Face image → Geometry extraction → Feature CSV
                                       ↓
                           Semantic representation → Text
                                                       ↓
                                              Gradio report
```

- **Geometry data:** facial landmarks provide measurements of distances, angles, and proportions.
- **Dataset preparation:** processing scripts join feature tables with provider labels using normalized image filenames.
- **Text generation:** experiments use templates or FLAN-T5 to produce text across five categories.
- **Report viewer:** five sections have a contents menu, copy buttons, and downloadable CSV/JSON files.
- **Light and dark mode:** a switch in the top-right corner of the header remembers your browser preference.

The landmark illustration in the interface is a schematic. It does not show measurements extracted from your uploaded image.

## What you need

- Git and Git LFS to download the source and datasets.
- A Python environment with the appropriate dependencies to open the notebook or try the interface.
- Several GB of free disk space for images and tables. The `Humans/` directory alone is about 2.06 GB.
- Jupyter or VS Code with notebook support to read the `.ipynb` file.

The commands below use PowerShell and run from the repository root. Internet access is needed to clone the project, download LFS data, and install dependencies.

## Setup

### 1. Clone the project and datasets

```powershell
git lfs install
git clone https://github.com/ElderOne9977/Physiognomy.git
cd Physiognomy
git lfs pull
```

Git LFS stores images and large datasets outside regular Git history. To browse the source before downloading the data, you can skip the initial LFS download:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone https://github.com/ElderOne9977/Physiognomy.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

When you need the datasets, enter the `Physiognomy` directory and run `git lfs pull`. This replaces small pointer files with the actual data. Downloads depend on the GitHub account's available LFS storage and bandwidth.

### 2. Open the notebook and explore the data

Start with [Physiognomy_VSCode_Colab_Features.ipynb](Physiognomy_VSCode_Colab_Features.ipynb). The notebook retains its existing cells and outputs so you can follow the processing steps. Some paths refer to the original environment; review each cell and adjust those paths before running it.

| Dataset | Contents |
| --- | --- |
| `Humans/` | 7,472 input images. |
| `faces_full_features.csv` | Original feature table. |
| `faces_full_features_clean.csv` | Cleaned table with 6,988 rows and 452 columns. |
| `labels_enriched.jsonl` | 7,467 provider-label records. |
| `training_dataset.csv`, `merged_full_features.csv` | Two joined versions of the features and labels. |
| `semantic_schema.json`, `semantic_labels.csv` | Definitions and encoded labels for 45 semantic concepts. |
| `semantic_training_dataset.csv` | 6,983 matched feature and semantic-label rows. |
| `training_*.jsonl`, `text_generation_*.jsonl` | Text datasets in the formats used by earlier experiments. |
| `Source/text_generation_controlled_train.jsonl` | Structured-concept and text dataset for the controlled-generation experiment. |
| `oof_semantic_*` | Saved out-of-fold predictions and metadata. |

Row counts differ because not every image has both extracted features and matching labels. Each processing stage also has its own schema: the cleaned table has 452 columns, while the semantic model consumes 95 base features and 21 missing-value indicators in the saved order.

### 3. Prepare the environment

Dependency files serve different parts of the project:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Playwright and HTML parsing tools from the earlier data-collection workflow. Playwright is also used for browser tests. |
| `requirements-demo.txt` | The Gradio version used by the interface. This file does not include every dependency imported by the project. |
| `requirements-cpu.txt` | Data, image-processing, and model libraries for the CPU environment. |

The CPU requirements use version ranges rather than a complete environment lock. Model weights, library caches, and local experiment outputs are not included in this upload. Cloning the repository therefore does not provide a complete offline inference bundle; some evaluation scripts also require artifacts kept separately on the original machine.

## Preview the interface

From a Python environment that already has the project's dependencies installed, run:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/experiments/ui_reading_20261014/mock_preview.py
```

Open **http://127.0.0.1:7862/** in your browser. Upload an image to try the layout, progress indicator, contents menu, copy buttons, and downloads. Press **Ctrl+C** in the terminal to stop the server. Run only one preview on this port.

The preview returns **mock text** without loading models or performing real predictions. Its downloadable CSV/JSON files are also fixtures for testing the interface.

## Testing

To run presentation tests in the prepared environment:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/test_demo_reading_view.py
```

Browser tests require Playwright and Chromium. Start the mock preview in one terminal, then run `python -u -B Source/test_demo_ui_browser.py` in another. The test uses synthetic images with the mock preview only. Stop the preview when testing is complete.

`Source/test_demo_gradio.py` also checks for model artifacts and therefore needs those local files. Mock tests verify software behavior, not accuracy on previously unseen faces.

## Troubleshooting

- **A dataset contains only a few lines of text:** confirm that Git LFS is installed, then run `git lfs pull` from the repository directory.
- **LFS downloads fail:** check your connection, access permissions, and available GitHub LFS storage and bandwidth.
- **Python reports a missing dependency:** check the active environment and the appropriate requirements file. Installing Gradio alone does not cover all project imports.
- **The notebook cannot find an image:** update the cell's paths and confirm that the `Humans/` directory contains the downloaded LFS data.
- **Port 7862 is already in use:** stop the previous preview with `Ctrl+C`, then start it again.
- **Model weights or experiment artifacts are missing:** these files are kept locally and are not part of this source-and-dataset release.

## Current limitations

Provider labels are *weak supervision*. They support experiments with representations and text generation, but are not independently verified ground truth.

The saved text test set contains **702 records** representing **686 distinct semantic vectors**. The original FLAN-T5 checkpoint produced only **5 distinct texts** on that set, indicating repetitive output. Templates may follow selected concepts more closely, but this does not establish that personal traits can be inferred from appearance.

Historical experiment results are not guarantees of quality or deployment readiness.

## Project layout

```text
Physiognomy/
├── Humans/                                  # Image corpus, stored with Git LFS
├── Physiognomy_VSCode_Colab_Features.ipynb    # Data-processing notebook
├── faces_full_features_clean.csv            # Cleaned features
├── labels_enriched.jsonl                    # Provider labels
├── semantic_training_dataset.csv            # Joined semantic dataset
├── training_*.jsonl                         # Text datasets
├── requirements*.txt                        # Dependencies by purpose
└── Source/
    ├── assets/                             # CSS, JavaScript, vector illustration
    ├── demo_gradio.py                       # Report interface
    ├── physiognomy.py                       # Image processing and geometry
    ├── build_*.py                          # Dataset preparation
    ├── test_*.py                           # Tests
    └── experiments/ui_reading_20261014/
        └── mock_preview.py                 # Mock interface preview
```

README is the only Markdown document published in the repository. Internal documentation, logs, backups, caches, personal test images, model weights, trial outputs, and Docker configuration that depends on local resources are kept outside Git.

## Built with

- **NumPy, pandas, and scikit-learn** for tabular processing and semantic experiments.
- **InsightFace, facexlib, and OpenCV** for image processing and facial geometry.
- **PyTorch, Transformers, and FLAN-T5** for text-generation experiments.
- **Gradio** for the interface and **Playwright** for browser tests.
- **Git LFS** for images and large datasets.

Libraries, models, and images have their own usage terms. The repository does not include a verified dataset license; check the applicable permissions before redistributing or reusing these resources.
