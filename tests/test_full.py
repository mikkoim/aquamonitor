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

@pytest.mark.usefixtures("aquamonitor_jyu_dataset", "aquamonitor_jyu_metadata")
def test_overall_full(aquamonitor_jyu_dataset, aquamonitor_jyu_metadata):
    ds = aquamonitor_jyu_dataset["validation"]
    df = aquamonitor_jyu_metadata

    df = df.query("fold0 == 'val'")

    index = {f"{k}.jpg":i for i,k in enumerate(ds["__key__"])}
    am = AquaMonitorDataset(df,
                            ds.rename_column("jpg", "x"),
                            index=index)
    image_id = am.images[0]
    imagepair_id = am.imagepairs[0]
    imaging_run_id = am.imaging_runs[0]
    individual_id = am.individuals[0]
    am.load(image=image_id)
    am.load(imagepair=imagepair_id)
    am.load(individual=individual_id)[0]["image"]
    am.show(image=image_id)
    am.show(imagepair=imagepair_id)
    am.show(imaging_run=imaging_run_id)
    am.show(imaging_run=imaging_run_id, imagepairs=True)