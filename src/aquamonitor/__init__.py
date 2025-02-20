from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from aquamonitor.utils import stack_images

@dataclass(frozen=True)
class AquaMonitorImage:
    id: str
    imaging_run: str
    area: int
    perimeter: int
    holes: int
    max_feret_diameter: int
    area_holes: int
    roi_left: int
    roi_top: int
    roi_right: int
    roi_bottom: int
    width: int
    height: int
    imaging_time: datetime

@dataclass(frozen=True)
class ImagePair:
    id: str
    images: tuple[AquaMonitorImage, AquaMonitorImage]

@dataclass(frozen=True)
class ImagingRun:
    id: str
    individual: str
    camera1: list[AquaMonitorImage]
    camera2: list[AquaMonitorImage]
    imagepairs: list[ImagePair]

@dataclass(frozen=True)
class Individual:
    id: str
    imaging_runs: list[ImagingRun]
    taxon: str
    taxon_group: str
    taxon_label: str
    taxon_code: str
    lake: str
    site: str
    sample: str
    plate: str
    position: str
    dataset: str
    year: int
    is_bulk: bool
    has_dna: bool
    has_biomass: bool
    fold0: str
    fold1: str
    fold2: str
    fold3: str
    fold4: str


def show_file_list(images: list[Image.Image], nrows=8, img_size=128):
    batch = []
    for img in images:
        img = ImageOps.pad(img, (img_size,img_size))
        I = np.asarray(img)
        batch.append(I)

    # Add empty patches if row count and image count don't match
    n_zeros = nrows - len(images) % nrows
    if n_zeros != nrows:
        for _ in range(n_zeros):
            batch.append(np.zeros((img_size,img_size,3), np.uint8))
    batch = np.array(batch)
            
    # Reshape to a single image
    b, h, w, c = batch.shape
    ncol = b // nrows

    img_grid = (batch.reshape(nrows, ncol, h, w, c)
              .swapaxes(1,2)

              .reshape(h*nrows, w*ncol, 3))
    return Image.fromarray(img_grid)

def sync_timestamps(list1, list2):
    """
    Given two lists of timestamps, returns a list of tuples containing the closest
    paired timestamps between the two lists. Any extra timestamps that do not have a pair
    are discarded.

    Args:
        list1 (list of float): First list of timestamps.
        list2 (list of float): Second list of timestamps.

    Returns:
        list of tuples: Each tuple is (timestamp_from_list1, timestamp_from_list2)
    """
    # Sort both lists (if not already sorted)
    list1 = sorted(list1)
    list2 = sorted(list2)
    
    pairs = []
    i, j = 0, 0

    # Use two pointers to iterate over both lists.
    while i < len(list1) and j < len(list2):
        current_diff = abs(list1[i] - list2[j])
        
        # Check if the next element in list1 is a closer match to list2[j]
        if i + 1 < len(list1) and abs(list1[i+1] - list2[j]) < current_diff:
            i += 1
        # Otherwise, check if the next element in list2 is a closer match to list1[i]
        elif j + 1 < len(list2) and abs(list1[i] - list2[j+1]) < current_diff:
            j += 1
        else:
            # If neither next element improves the match, then current pair is best
            pairs.append((list1[i], list2[j]))
            i += 1
            j += 1

    return pairs

class AquaMonitorDataset():
    def __init__(self, df: pd.DataFrame, ds):
        self.df = df
        self.ds = ds
        self.index = {f"{k}.jpg":i for i,k in enumerate(ds["__key__"])}
        self.init_images()
        self.init_imaging_runs()
        self.init_individuals()

    def init_images(self):
        """Initializes the AquaMonitorImage dictionary"""
        self.image_dict = {}
        print("Initializing images...")
        records = self.df.to_dict(orient="records")
        for row in records:
            image = AquaMonitorImage(
                                    id=row["img"],
                                    imaging_run=row["imaging_run"],
                                    area=row["area"],
                                    perimeter=row["perimeter"],
                                    holes=row["holes"],
                                    max_feret_diameter=row["max_feret_diameter"],
                                    area_holes=row["area_holes"],
                                    roi_left=row["roi_left"],
                                    roi_top=row["roi_top"],
                                    roi_right=row["roi_right"],
                                    roi_bottom=row["roi_bottom"],
                                    width=row["width"],
                                    height=row["height"],
                                    imaging_time=row["imaging_time"])
            self.image_dict[row["img"]] = image
        print(f"Done. {len(self.image_dict)} images.")
    
    def _create_imagepairs(self, imaging_run_id, camera1, camera2) -> list[ImagePair]:
        """Creates synced imagepairs"""
        timestamps1 = [image.imaging_time for image in camera1]
        timestamps2 = [image.imaging_time for image in camera2]
        img_map1 = {x.imaging_time: x for x in camera1}
        img_map2 = {x.imaging_time: x for x in camera2}
        timepairs = sync_timestamps(timestamps1, timestamps2)
        imagepairs = []
        for i, timepair in enumerate(timepairs):
            imagepair_id = f"{imaging_run_id}_pair{i:04d}"
            image1 = img_map1[timepair[0]] # AquaMonitorImage
            image2 = img_map2[timepair[1]] # AquaMonitorImage
            imagepair = ImagePair(id=imagepair_id,
                                  images=(image1, image2))
            self.imagepair_dict[imagepair_id] = imagepair
            imagepairs.append(imagepair)
        return imagepairs
    
    def init_imaging_runs(self):
        """Initializes ImagingRun and ImagePair dictionaries"""
        self.imaging_run_dict = {}
        self.imagepair_dict = {}
        print("Initializing imaging runs...")
        for imaging_run_id in tqdm(self.df.imaging_run.unique()):
            # Fetch imaging run rows
            imaging_run_df = self.df.query(f"imaging_run == '{imaging_run_id}'").sort_values("imaging_time")
            individual_id = imaging_run_df.individual.iloc[0]

            # Separate cameras
            camera1_imgs = []
            camera2_imgs = []
            for image_id, camera in zip(imaging_run_df.img, imaging_run_df.camera):
                image = self.image_dict[image_id] # Fetch reference to AquaMonitorImage
                if camera == 1:
                    camera1_imgs.append(image)
                else:
                    camera2_imgs.append(image)
            
            # Create image pair objects
            imagepairs = self._create_imagepairs(imaging_run_id, camera1_imgs, camera2_imgs)

            imaging_run = ImagingRun(id=imaging_run_id,
                                     individual=individual_id,
                                     camera1=camera1_imgs, 
                                     camera2=camera2_imgs,
                                     imagepairs=imagepairs)
            self.imaging_run_dict[imaging_run_id] = imaging_run
        print(f"Done. {len(self.imaging_run_dict)} imaging_runs.")
    
    def init_individuals(self):
        self.individual_dict = {}
        print("Initializing individuals...")
        records = self.df.groupby("individual").first().reset_index().to_dict(orient="records")
        imaging_run_id_df = self.df.groupby("individual")["imaging_run"].unique()
        for row in records:
            imaging_run_ids = imaging_run_id_df[row["individual"]]
            imaging_runs = [self.imaging_run_dict[imaging_run_id] for imaging_run_id in imaging_run_ids]
            individual = Individual(id=row["individual"],
                                    imaging_runs=imaging_runs,
                                    taxon=row["taxon"],
                                    taxon_group=row["taxon_group"],
                                    taxon_label=row["taxon_label"],
                                    taxon_code=row["taxon_code"],
                                    lake=row["lake"],
                                    site=row["site"],
                                    sample=row["sample"],
                                    plate=row["plate"],
                                    position=row["position"],
                                    dataset=row["dataset"],
                                    year=row["year"],
                                    is_bulk=row["is_bulk"],
                                    has_dna=row["has_dna"],
                                    has_biomass=row["has_biomass"],
                                    fold0=row["fold0"],
                                    fold1=row["fold1"],
                                    fold2=row["fold2"],
                                    fold3=row["fold3"],
                                    fold4=row["fold4"])
            self.individual_dict[row["individual"]] = individual
        print(f"Done. {len(self.individual_dict)} individuals.")
    
    def __repr__(self):
        return f"AquaMonitorDataset with {len(self.individual_dict)} individuals, {len(self.imaging_run_dict)} imaging runs and {len(self.image_dict)} images."
    
    def __call__(self, image=None, imagepair=None, imaging_run=None, individual=None):
        if image is not None:
            return self.image_dict[image]
        if imagepair is not None:
            return self.imagepair_dict[imagepair]
        if imaging_run is not None:
            return self.imaging_run_dict[imaging_run]
        if individual is not None:
            return self.individual_dict[individual]
        raise ValueError("Provide either image, imaging_run or individual.")
    
    def _load_image(self, image_id):
        return self.ds[self.index[image_id]]["jpg"]
    
    def load(self, image=None, imagepair=None, imaging_run=None, individual=None, camera=None, imagepairs=False):
        if image is not None:
            image_object = self(image=image)
            return {"image": self._load_image(image), "data": asdict(image_object)}

        if imagepair is not None:
            imagepair_object = self(imagepair=imagepair)
            image1 = imagepair_object.images[0]
            image2 = imagepair_object.images[1]
            return {"image": (self._load_image(image1.id),
                               self._load_image(image2.id)),
                    "data": (asdict(image1), asdict(image2))}

        if imaging_run is not None and imagepairs:
            imaging_run_object = self(imaging_run=imaging_run)
            image_tuples = []
            data_tuples = []
            for imagepair_object in imaging_run_object.imagepairs:
                image1 = imagepair_object.images[0]
                image2 = imagepair_object.images[1]
                image_tuples.append((self._load_image(image1.id), self._load_image(image2.id)))
                data_tuples.append((asdict(image1), asdict(image2)))
            return {"image": image_tuples, "data": data_tuples}

        if imaging_run is not None:
            imaging_run_object = self(imaging_run=imaging_run)
            images1 = []
            data1 = []
            images2 = []
            data2 = []
            if len(imaging_run_object.camera1) > 0:
                images1 = [self._load_image(image.id) for image in imaging_run_object.camera1]
                data1 = [asdict(image) for image in imaging_run_object.camera1]

            if len(imaging_run_object.camera2) > 0:
                images2 = [self._load_image(image.id) for image in imaging_run_object.camera2]
                data2 = [asdict(image) for image in imaging_run_object.camera2]

            if camera == 1:
                return {"image": images1, "data": data1}
            if camera == 2:
                return {"image": images2, "data": data2}
            return {"image": images1 + images2, "data": data1 + data2}

        if individual is not None:
            individual = self(individual=individual)
            imaging_runs = []
            for imaging_run_object in individual.imaging_runs:
                imaging_run_record = self.load(imaging_run=imaging_run_object.id)
                record = {"imaging_run": imaging_run_object.id,
                          "image": imaging_run_record["image"],
                          "data": imaging_run_record["data"]}
                imaging_runs.append(record)
            return imaging_runs
    
    def show(self, image=None, imagepair=None, imaging_run=None, individual=None, camera=None, imagepairs=False, n_rows=8, img_size=128):
        if image is not None:
            return self._load_image(image)

        if imagepair is not None:
            imagepair_object = self(imagepair=imagepair)
            image1 = self._load_image(imagepair_object.images[0].id)
            image2 = self._load_image(imagepair_object.images[1].id)
            return stack_images([image1,image2], orientation="vertical")
        
        if imaging_run is not None and imagepairs:
            imagepair_objects = self.load(imaging_run=imaging_run, imagepairs=True)
            pair_images = []
            for pair in imagepair_objects["image"]:
                img1 = pair[0]
                img2 = pair[1]
                pair_images.append(stack_images([img1, img2], orientation="vertical"))
            return stack_images(pair_images, orientation="horizontal")

        if imaging_run is not None:
            images = self.load(imaging_run=imaging_run, camera=camera, imagepairs=False)["image"]
            return show_file_list(images, nrows=n_rows, img_size=img_size)

        if individual is not None:
            imaging_run_images = []
            imaging_runs = self(individual=individual).imaging_runs
            for imaging_run in imaging_runs:
                pil_image = self.show(imaging_run=imaging_run.id, camera=camera, imagepairs=imagepairs, n_rows=n_rows, img_size=img_size)
                imaging_run_images.append(pil_image)
            return imaging_run_images
    @property
    def imaging_runs(self):
        return list(self.imaging_run_dict.keys())
    
    @property
    def individuals(self):
        return list(self.individual_dict.keys())

    @property
    def images(self):
        return list(self.image_dict.keys())
    
    @property
    def imagepairs(self):
        return list(self.imagepair_dict.keys())