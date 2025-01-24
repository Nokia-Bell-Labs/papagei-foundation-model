<div id="toc">
   <ul align="center" style="list-style: none;">
  <a href="[https://github.com/evelyn0414/OPERA](https://github.com/Nokia-Bell-Labs/papagei-foundation-model"> <img width="20%" height="20%" src="figures/papagei-logo.png"></a>
  <summary>
     <h1>PaPaGei</h1> <br>
    <h2> Open Foundation Models for Optical Physiological Signals </h2>
  </summary>
   </ul>
</div>


## :rocket: Updates
- Jan 22nd 2025: PaPaGei is [accepted](https://openreview.net/forum?id=kYwTmlq6Vn&referrer) to the International Conference on Learning Representations (ICLR)
- Dec 15th 2024: PaPaGei recevied [Best Paper Award](https://neurips-time-series-workshop.github.io/accepted-papers/) 🏆 at NeurIPS workshop on Time Series in the Age of Large Models (TSALM)
- Oct 29th 2024: The paper is available on [Arxiv](https://arxiv.org/abs/2410.20542).
- Oct 24th 2024: Visit Arvind's page on Zenodo ([here](https://zenodo.org/records/13983110)) to access the models!
- Oct 15th 2024: The code is now available! 
  
## :book: Summary
Photoplethysmography (PPG) is the most widely used non-invasive technique for monitoring biosignals and cardiovascular health, with applications in both clinical settings and consumer health through wearable devices. Current machine learning models trained on PPG signals are mostly task-specific and lack generalizability. Previous works often used single-device datasets, did not explore out-of-domain generalization, or did not release their models, hindering reproducibility and further research. We introduce PaPaGei, **the first open foundation model for PPG signals**. PaPaGei is **pre-trained on more than 57,000 hours** of 20 million unlabeled segments of PPG signals using publicly available datasets exclusively. We evaluate against popular time-series foundation models and other benchmarks on **20 tasks of 10 diverse datasets spanning cardiovascular health, sleep disorders, pregnancy monitoring, and wellbeing assessment**. Our architecture incorporates novel representation learning approaches that leverage differences in PPG signal morphology across individuals, enabling it to **capture richer representations** than traditional contrastive learning methods. Across 20 tasks, PaPaGei improves classification and regression performance by an average of 6.3\% and 2.9\%, respectively, compared to other competitive time-series foundation models in at least 14 tasks. **PaPaGei is more data- and parameter-efficient than other foundation models or methods**, as it outperforms 70x larger models. Beyond accuracy, we also investigate robustness against different skin tones, establishing a benchmark for bias evaluations of future models. Notably, PaPaGei can be used out of the box as both a **feature extractor** and an **encoder** for other multimodal models, opening up new opportunities for multimodal health monitoring.

<div align="center">
  <img src="figures/model-overview.png" alt="Project Screenshot"/>
</div>

## :chart_with_upwards_trend: How to use

PaPaGei can be useful in multiple ways:

1. Developers and researchers can use it out-of-the-box to extract transferrable features for ML (instead of handcrafted features).
2. It can also be used as a PPG encoder that plugs into other powerful frontier models (LLMs such as [AnyMAL](https://arxiv.org/abs/2309.16058) etc).

#### Installation 

1. Create a conda environment: ```conda create -n papagei_env python==3.10```
2. Activate the environment: ```conda activate papagei_env```
3. Install the required packages in the environment: ```pip install -r requirements.txt ```
4. Install the pyPPG package: ```pip install pyPPG==1.0.41``` (While this may result in wfdb package conflict, it will still work).

#### Downloading the model weights

To access the model weights, you can download them from [Zenodo](https://zenodo.org/records/13983110) hosted by [Arvind Pillai](https://arvindpillai.io/). For feature extraction, please save it in a folder called ```weights``` and/or change the path when loading.

#### Extracting embeddings

A quick example demonstrating how to extract embeddings from the encoder:

1. Import the necessary packages
```python
import numpy as np
import torch
from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments
from torch_ecg._preprocessors import Normalize
from models.resnet import ResNet1DMoE
```
2. Load the PaPaGei-S model
```python
### Load Model ###

model_config = {'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            'n_experts': 3
            }

model = ResNet1DMoE(in_channels=1, 
            base_filters=model_config['base_filters'], 
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts'])

model_path = "weights/papagei_s.pt"
model = load_model_without_module_prefix(model, model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

3. Clean and pre-process a PPG signal
```python
### Clean Signal ###
fs = 500
fs_target = 125
segment_length = fs * 10 
signal = np.random.randn(30000) # a 60 second signal at 500Hz

print(f"PPG dimensions : {signal.shape}")
norm = Normalize(method='z-score')
signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal,
                                        frequency=fs)
signal = waveform_to_segments(waveform_name='ppg',
                   segment_length=segment_length,
                   clean_signal=signal)
signal = resample_batch_signal(signal, fs_original=fs, fs_target=fs_target, axis=-1)
print(f"After segmentation : {signal.shape}")
signal = torch.Tensor(signal).unsqueeze(dim=1)
```

4. Extract the embeddings
```python
### Extract Features ###
model.eval()
with torch.inference_mode():
    signal = signal.to(device)
    outputs = model(signal)
    embeddings = outputs[0].cpu().detach().numpy()
print(f"Embedding dimensions : {embeddings.shape}")
```

An example notebook describing end-to-end feature extraction and downstream evaluation on the ppg-bp dataset is available [here](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/example_papagei.ipynb).

## Brief description of important modules

<div align="center">
  <img src="figures/PaPaGei.png" alt="Project Screenshot"/>
</div>

We describe the end-to-end workflow below.

#### Step 1: PPG Data Pre-processing
The code required to preprocess the raw PPG signal is in the ```preprocessing``` folder:
- ```flatline.py```: Check if PPG signal has flatline sections using the ```BioBSS``` package.
- ```ppg.py```:
  - ```preprocess_one_ppg_signal```: Function to apply a bandpass filter on raw signals.
  - Other necessary IO functions for batch processing and saving signals. 

```segmentations.py``` contains the code needed to segment the filtered PPG signal:
- ```waveform_to_segment```: Function to segment signal based on segment length
- Other utilities for saving segments.

#### Step 2: Morphology Augmentation Module Computation

```morphology.py``` contains code required to compute stress-Induced vascular response index (sVRI), Inflection Point Area ratio (IPA), and Signal Quality Index (SQI):

- ```extract_svri```: Function to compute sVRI.
- ```skewness_sqi```: Function to compute SQI.
- ```compute_ipa```: Function to compute IPA.
- Contains utility functions to compute these metrics for all segments in batches and save them.

#### Step 3: Dataset and Time series Augmentations

- ```dataset.py```: The ```PPGDatasetLabelsArray``` is a PyTorch custom dataset class that is used in PaPaGei-S. Note that the dataloader is created before training in ```training_mt.py```.
- ```augmentations.py```: Contains time series augmentation code as ```torch.nn.Module``` classes for easier on the fly transforms. 

#### Step 4: Training
The ```resnet.py``` file in the ```models``` folders contains the model architecture. In particular, the ```ResNet1DMoE``` is the PaPaGei-S model.

```training_mt.py``` contains the end-to-end distributed training code for PaPaGei-S.
  - ```train_step```: Function for a single train step that computes PaPaGei-S loss.
  - ```training```: Trains for pre-defined steps, and checkpoints and saves the model.
  - ```main```: The main function for distributed training.

#### Step 5: Feature Extraction
After training, we extract embeddings using the ```feature_extraction.py``` files.

- ```compute_signal_embeddings```: Function to extract embeddings from the pre-trained model.
- ```save_embeddings```: Utility Function to save the model.
  
#### Step 6: Linear Evaluation

The saved embeddings from Step 5 can be passed to a linear model or shallow ANN for classification/regression.

## :file_folder: Other Repositories

We gratefully acknowledge the work from the following projects that made the evaluation of our model possible:

- **[Chronos](https://github.com/amazon-science/chronos-forecasting)**
- **[Moment](https://github.com/moment-timeseries-foundation-model/moment?tab=readme-ov-file)**
- **[REGLE](https://github.com/Google-Health/genomics-research)** 
- **[TF-C](https://github.com/mims-harvard/TFC-pretraining)** 
- **[BYOL](https://github.com/chengding0713/SiamQuality/tree/main)** 
- **[Morphology](https://github.com/qiriro/PPG)** 

## Citation
If you use models, code, or ideas from this project, please cite our [paper](https://arxiv.org/abs/2410.20542):
```bibtex
@misc{pillai2024papagei,
      title={PaPaGei: Open Foundation Models for Optical Physiological Signals}, 
      author={Arvind Pillai and Dimitris Spathis and Fahim Kawsar and Mohammad Malekzadeh},
      year={2024},
      url={https://arxiv.org/abs/2410.20542}, 
}
```
