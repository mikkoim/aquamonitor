import pytest
from aquamonitor import AquaMonitorDataset

@pytest.mark.usefixtures("aquamonitor_jyu_dataset", "aquamonitor_jyu_metadata")
def test_overall(aquamonitor_jyu_dataset, aquamonitor_jyu_metadata):
    ds = aquamonitor_jyu_dataset
    df = aquamonitor_jyu_metadata

    df = df.query("fold0 == 'val'")
    am = AquaMonitorDataset(df, ds["validation"])

    assert len(am.images) == len(df["img"].unique())
    assert len(am.imaging_runs) == len(df["imaging_run"].unique())
    assert len(am.individuals) == len(df["individual"].unique())

    individual



def test_overall(aquamonitor_jyu_dataset, aquamonitor_jyu_metadata):
    ds = aquamonitor_jyu_dataset
    df = aquamonitor_jyu_metadata

    df = df.query("fold0 == 'val'")
    am = AquaMonitorDataset(df, ds["validation"])

    am.load()