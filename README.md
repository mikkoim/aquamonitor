# AquaMonitor

This is the main repository for the AquaMonitor dataset. 
AquaMonitor is a large, multimodal, and multi-view image sequence dataset of aquatic invertebrates, collected during two years of operational environmental monitoring. It allows benchmarking computer vision algorithms for fine-grained classification, open-set detection, out-of-distribution detection and domain adaptation, all which are problems encountered in real-life monitoring situations. The dataset has 2.7M images from 43,189 specimens, DNA sequences for 1358 specimens, and dry mass and size measurements for 1494 specimens.

The dataset is available at Huggingface Datasets: https://huggingface.co/datasets/mikkoim/aquamonitor.

The codes used to produce the dataset are in a separate repository: https://github.com/mikkoim/aquamonitor-codes. This repository contains also vast amounts of metadata related to the dataset.

Baseline models and predictions are in https://huggingface.co/mikkoim/aquamonitor-baselines. These will be made more easily available in the near future.

This repository contains a utility library for handling AquaMonitor sequences and metadata.

# Downloading the data

Using Huggingface `datasets`:
```python
import datasets
ds = datasets.load_dataset("mikkoim/aquamonitor", data_dir="images", split="train", cache_dir="aquamonitor")
```
The full dataset will consume \~100GB of disk space, and it is recommended to cache it to a known location.

For testing, you can use the thumbnail dataset (~10GB):
```python
ds_thumbs = datasets.load_dataset("mikkoim/aquamonitor", data_dir="thumbnail", split="train", cache_dir="aquamonitor")
```

You can also download the raw `.tar` partitions from [here](https://huggingface.co/datasets/mikkoim/aquamonitor/tree/main/images)

The metadata can be accessed straight from Huggingface using pandas:

```python
import pandas as pd
df = pd.read_parquet("https://huggingface.co/datasets/mikkoim/aquamonitor/resolve/main/aquamonitor-monitor.parquet.gzip")
df_train = df.query("fold0 == 'train'")
df_val = df.query("fold0 == 'val'")
```

The benchmark splits are in separate files:

```python
df_classif = pd.read_parquet("https://huggingface.co/datasets/mikkoim/aquamonitor/resolve/main/aquamonitor-classif.parquet.gzip")
df_fewshot = pd.read_parquet("https://huggingface.co/datasets/mikkoim/aquamonitor/resolve/main/aquamonitor-fewshot.parquet.gzip")
```

See the dataset repository for details on metadata columns.

# AquaMonitor library

This repository contains a utility library that makes dataset handling a bit easier. It makes it also possible to retrieve synced image pairs from the dataset.

## Installation

```bash
pip install git+https://github.com/mikkoim/aquamonitor.git
```

The `AquaMonitorDataset` needs and index defined from the metadata.
```python
import aquamonitor
index = {f"{k}.jpg":i for i,k in enumerate(ds["train"]["__key__"])}
am = aquamonitor.AquaMonitorDataset(df_train,
                                    ds["train"].rename_column("jpg", "x"),
                                    index=index)
```

See `demo.ipynb` for a full example using the lightweight biomass subset of the dataset.
