<div align="center">
  <a href="https://github.com/Nokia-Bell-Labs/papagei-foundation-model">
    <img width="20%" height="250%" src="figures/papagei-logo.png" alt="PaPaGei Logo">
  </a>
  <h1>PaPaGei</h1>
  <h2>Open Foundation Models for Optical Physiological Signals</h2>
  <p>
    <a href="https://arxiv.org/abs/2410.20542"><img src="https://img.shields.io/badge/arXiv-2410.20542-b31b1b.svg" alt="ArXiv"></a>
    <a href="https://zenodo.org/records/13983110"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13983110.svg" alt="DOI"></a>
    </p>
</div>

## 📖 Overview

Photoplethysmography (PPG) is a non-invasive optical technique widely used for monitoring biosignals and cardiovascular health, prevalent in both clinical settings and consumer wearable devices. Current machine learning models for PPG signals are often task-specific and struggle with generalizability. Many prior works utilize single-device datasets, neglect out-of-domain generalization, or do not publicly release their models, thereby limiting reproducibility and broader research progress.

**PaPaGei** is the **first open foundation model for PPG signals**. It is pre-trained on over **57,000 hours** of 20 million unlabeled PPG segments, exclusively using publicly available datasets. We benchmark PaPaGei against popular time-series foundation models and other methods across **20 tasks from 10 diverse datasets**, covering cardiovascular health, sleep disorders, pregnancy monitoring, and wellbeing assessment.

Our novel architecture incorporates representation learning approaches that capitalize on morphological differences in PPG signals across individuals, enabling it to **capture richer representations** than traditional contrastive learning methods. PaPaGei demonstrates significant improvements, boosting classification and regression performance by an average of **6.3% and 2.9%**, respectively, compared to other leading time-series foundation models in at least 14 tasks. Notably, PaPaGei is **more data- and parameter-efficient**, outperforming models up to 70x larger.

Beyond accuracy, we investigate robustness against different skin tones, establishing a benchmark for evaluating bias in future models. PaPaGei can be readily used as a **feature extractor** or an **encoder** for multimodal models, paving the way for new advancements in multimodal health monitoring.

<div align="center">
  <img src="figures/model-overview.png" alt="PaPaGei Model Overview" width="70%"/>
</div>

---

## 🚀 Updates

* **Jan 22, 2025**: PaPaGei accepted to the International Conference on Learning Representations (ICLR). [Read the latest version of the paper](https://arxiv.org/pdf/2410.20542v2).
* **Dec 15, 2024**: PaPaGei received the 🏆 **Best Paper Award** at the NeurIPS workshop on Time Series in the Age of Large Models (TSALM). [See accepted papers](https://neurips-time-series-workshop.github.io/accepted-papers/).
* **Oct 29, 2024**: Paper available on [arXiv](https://arxiv.org/abs/2410.20542).
* **Oct 24, 2024**: Access the model weights on Zenodo ([here](https://zenodo.org/records/13983110)).
* **Oct 15, 2024**: Code released! 🎉

---

## 🛠️ How to Use PaPaGei

PaPaGei offers versatility for developers and researchers:

1.  **Out-of-the-Box Feature Extraction**: Use PaPaGei to extract transferable features for your machine learning tasks, replacing handcrafted features.
2.  **PPG Encoder Integration**: Incorporate PaPaGei as a PPG encoder into larger frontier models (e.g., LLMs like [AnyMAL](https://arxiv.org/abs/2309.16058)).

### 📦 Installation

1.  **Create a Conda Environment:**
    ```bash
    conda create -n papagei_env python=3.10
    conda activate papagei_env
    ```
2.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install pyPPG Package:**
    ```bash
    pip install pyPPG==1.0.41
    ```
    *Note: This might show a `wfdb` package conflict, but it should still function correctly.*
4.  **Restore the Pinned numpy Version:**
    ```bash
    pip install numpy==1.26.4 --force-reinstall
    ```
    *Note: `pyPPG==1.0.41` hard-pins `numpy==1.23.2`, which lacks the `numpy.dtypes` submodule required by `torch_ecg`, causing an `AttributeError: module 'numpy' has no attribute 'dtypes'` on import (see [issue #6](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/issues/6)). This step re-installs the numpy version pinned in `requirements.txt` after pyPPG downgrades it.*

### 🧠 Downloading Model Weights

Model weights are hosted on Zenodo by Arvind Pillai.
* **Download Link**: [Zenodo Record 13983110](https://zenodo.org/records/13983110)
* For feature extraction, save the downloaded model weights (e.g., `papagei_s.pt`) into a folder named `weights/` in your project directory, or update the path accordingly in your scripts.

### ✨ Extracting Embeddings: Quick Start

Here’s a brief example of how to load the PaPaGei-S model and extract embeddings:

1.  **Import Necessary Packages:**
    ```python
    import numpy as np
    import torch
    from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
    from preprocessing.ppg import preprocess_one_ppg_signal
    from segmentations import waveform_to_segments
    from torch_ecg._preprocessors import Normalize
    from models.resnet import ResNet1DMoE
    ```

2.  **Load the PaPaGei-S Model:**
    ```python
    # Define Model Configuration
    model_config = {
        'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512, # Embedding dimension
        'n_experts': 3
    }

    # Initialize Model
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=model_config['base_filters'],
        kernel_size=model_config['kernel_size'],
        stride=model_config['stride'],
        groups=model_config['groups'],
        n_block=model_config['n_block'],
        n_classes=model_config['n_classes'],
        n_experts=model_config['n_experts']
    )

    # Load Pre-trained Weights
    model_path = "weights/papagei_s.pt" # Ensure this path is correct
    model = load_model_without_module_prefix(model, model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded on {device}")
    ```

3.  **Pre-process a PPG Signal:**
    ```python
    # Example PPG Signal
    fs = 500  # Original sampling frequency in Hz
    fs_target = 125 # Target sampling frequency in Hz
    segment_duration_seconds = 10 # Duration of each segment in seconds
    signal_duration_seconds = 60 # Total duration of the example signal

    signal = np.random.randn(signal_duration_seconds * fs) # Example: 60s signal at 500Hz
    print(f"Original PPG dimensions: {signal.shape}")

    # Clean and segment the signal
    signal_processed, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=fs)
    
    segment_length_original_fs = fs * segment_duration_seconds
    segmented_signals = waveform_to_segments(
        waveform_name='ppg', # Can be any name, not strictly used in this function
        segment_length=segment_length_original_fs,
        clean_signal=signal_processed
    )
    
    # Resample segments
    resampled_segments = resample_batch_signal(
        segmented_signals, 
        fs_original=fs, 
        fs_target=fs_target, 
        axis=-1
    )
    print(f"After segmentation and resampling: {resampled_segments.shape}") # (num_segments, segment_length_target_fs)

    # Convert to PyTorch Tensor
    signal_tensor = torch.Tensor(resampled_segments).unsqueeze(dim=1).to(device) # (num_segments, 1, segment_length_target_fs)
    ```

4.  **Extract Embeddings:**
    ```python
    with torch.inference_mode():
        outputs = model(signal_tensor)
        # PaPaGei-S returns a tuple (embeddings, expert_outputs, gating_weights)
        # We are interested in the first element: embeddings
        embeddings = outputs[0].cpu().detach().numpy()
    print(f"Embedding dimensions: {embeddings.shape}") # (num_segments, n_classes)
    ```

👉 For a comprehensive end-to-end example, including feature extraction and downstream task evaluation on the `ppg-bp` dataset, please refer to the Jupyter Notebook: [`example_papagei.ipynb`](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/example_papagei.ipynb).

**Important Considerations:**
* **Model Variability**: No single model excels across all tasks and datasets. We release the models that achieved the most wins in our evaluations.
* **Confidence Intervals**: Instead of fixed random seeds, we use bootstrapping (500 iterations) to compute 95% confidence intervals, providing a performance range.

---

## ⚙️ Workflow & Modules

The end-to-end workflow of PaPaGei involves several key stages:

<div align="center">
  <img src="figures/PaPaGei.png" alt="PaPaGei Workflow Diagram" width="80%"/>
</div>

1.  **PPG Data Pre-processing** (`preprocessing/`, `segmentations.py`):
    * `preprocessing/flatline.py`: Detects flatline sections in PPG signals using the `BioBSS` package.
    * `preprocessing/ppg.py`:
        * `preprocess_one_ppg_signal`: Applies a bandpass filter to raw signals.
        * Includes I/O functions for batch processing and saving.
    * `segmentations.py`:
        * `waveform_to_segments`: Segments filtered PPG signals based on specified segment lengths.
        * Utility functions for saving segments.

2.  **Morphology Augmentation Module Computation** (`morphology.py`):
    * Computes morphological features like Stress-Induced Vascular Response Index (sVRI), Inflection Point Area ratio (IPA), and Signal Quality Index (SQI).
    * `extract_svri`: Calculates sVRI.
    * `skewness_sqi`: Calculates SQI.
    * `compute_ipa`: Calculates IPA.
    * Includes batch processing utilities.

3.  **Dataset Handling and Time-Series Augmentations** (`dataset.py`, `augmentations.py`):
    * `dataset.py`:
        * `PPGDatasetLabelsArray`: A PyTorch custom `Dataset` class used for PaPaGei-S training. DataLoaders are set up in `training_mt.py`.
    * `augmentations.py`:
        * Provides time-series augmentation techniques implemented as `torch.nn.Module` classes for easy on-the-fly transformations during training.

4.  **Model Training** (`models/resnet.py`, `training_mt.py`):
    * `models/resnet.py`: Contains the model architecture. `ResNet1DMoE` is the PaPaGei-S model.
    * `training_mt.py`: Manages end-to-end distributed training for PaPaGei-S.
        * `train_step`: Defines a single training step, including loss computation for PaPaGei-S.
        * `training`: Orchestrates the training loop, checkpointing, and model saving.
        * `main`: Entry point for distributed training.

5.  **Feature Extraction** (`feature_extraction.py`):
    * `compute_signal_embeddings`: Extracts embeddings using the pre-trained model.
    * `save_embeddings`: Utility for saving extracted embeddings.

6.  **Linear Evaluation**:
    * The embeddings extracted in Step 5 can be used as input to linear models or shallow Artificial Neural Networks (ANNs) for various downstream classification or regression tasks.

---

## 🙏 Acknowledgements

We gratefully acknowledge the contributions of the following projects, which were instrumental in the evaluation of PaPaGei:

* **Chronos**: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
* **Moment**: [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment)
* **REGLE**: [Google-Health/genomics-research](https://github.com/Google-Health/genomics-research)
* **TF-C**: [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining)
* **BYOL (for PPG Quality)**: [chengding0713/SiamQuality](https://github.com/chengding0713/SiamQuality/tree/main)
* **Morphology (PPG features)**: [qiriro/PPG](https://github.com/qiriro/PPG)

---

## 📜 Citation

If you use PaPaGei models, code, or ideas from this project in your research, please cite our paper:

```bibtex
@inproceedings{pillai2025papagei,
  title={{PaPaGei: Open Foundation Models for Optical Physiological Signals}},
  author={Arvind Pillai and Dimitris Spathis and Fahim Kawsar and Mohammad Malekzadeh},
  booktitle={The Thirteenth International Conference on Learning Representations, {ICLR} 2025},
  year={2025},
  month={April},
  address={Singapore},
  note={Accepted. arXiv preprint arXiv:2410.20542},
  url={[https://arxiv.org/abs/2410.20542](https://arxiv.org/abs/2410.20542)}
}

