import pytest
import datasets
import pandas as pd

@pytest.fixture
def aquamonitor_jyu_dataset():
    ds = datasets.load_dataset("mikkoim/aquamonitor-jyu", cache_dir="hf")
    return ds

@pytest.fixture
def aquamonitor_jyu_metadata():
    df = pd.read_parquet("https://huggingface.co/datasets/mikkoim/aquamonitor-jyu/resolve/main/aquamonitor-jyu.parquet.gzip")
    return df