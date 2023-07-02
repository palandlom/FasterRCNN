import os
import shutil
import sys
import logging as log

log.basicConfig(level=log.INFO, filename="dsconverter.log", filemode="w")
# log.basicConfig(level=log.DEBUG, filename="dsconverter.log", filemode="w")

from typing import List

import lxml
from lxml import objectify

from dsconverter.models import DatasetSample

# me
print("sys.argv ", sys.argv)


def convert(dataset_path, out_path) -> List[str]:
    dataset_path = os.path.abspath(dataset_path)
    dataset_train_path = os.path.join(dataset_path, "train")
    dataset_test_path = os.path.join(dataset_path, "test")

    # Collect samples
    train_samples = get_samples_with_id_assigning(dataset_train_path, first_id=1)
    test_samples = get_samples_with_id_assigning(dataset_test_path, first_id=int(train_samples[-1].id) + 1)
    val_samples = []
    trainval_samples = []

    # Collect classes of objects
    obj_classes = get_object_classes(train_samples)

    # === Create output folders
    image_sets_dir = os.path.join(out_path, "ImageSets")
    main_subdir = os.path.join(image_sets_dir, "Main")
    layout_subdir = os.path.join(image_sets_dir, "Layout")
    segmentation_subdir = os.path.join(image_sets_dir, "Segmentation")

    all_annotations_dir = os.path.join(out_path, "Annotations")
    all_jpeg_images_dir = os.path.join(out_path, "JPEGImages")

    create_dir(out_path)
    create_dir(image_sets_dir)
    create_dir(main_subdir)
    create_dir(layout_subdir)
    create_dir(segmentation_subdir)

    create_dir(all_annotations_dir)
    create_dir(all_jpeg_images_dir)

    # === Copy samples of all subsets to common jpeg and annot folders ...
    train_samples = update_and_copy_samples_to_output_folders(train_samples, all_annotations_dir, all_jpeg_images_dir)
    test_samples = update_and_copy_samples_to_output_folders(test_samples, all_annotations_dir, all_jpeg_images_dir)

    # === Fill ImagesSets/Main folder
    for obj_class in obj_classes:
        create_sample_id_to_is_class_exist_file(obj_class, train_samples,
                                                os.path.join(main_subdir, f"{obj_class}_train.txt"))
        create_sample_id_to_is_class_exist_file(obj_class, test_samples,
                                                os.path.join(main_subdir, f"{obj_class}_test.txt"))
        create_sample_id_to_is_class_exist_file(obj_class, val_samples,
                                                os.path.join(main_subdir, f"{obj_class}_val.txt"))
        create_sample_id_to_is_class_exist_file(obj_class, trainval_samples,
                                                os.path.join(main_subdir, f"{obj_class}_trainval.txt"))

    # === Fill ImagesSets/Layout folder ! maybe useless
    create_file_with_sample_ids(train_samples, os.path.join(layout_subdir, f"train.txt"))
    create_file_with_sample_ids(test_samples, os.path.join(layout_subdir, f"test.txt"))
    create_file_with_sample_ids(trainval_samples, os.path.join(layout_subdir, f"trainval.txt"))
    create_file_with_sample_ids(val_samples, os.path.join(layout_subdir, f"val.txt"))

    # === Fill ImagesSets/Segmentation folder ! maybe useless
    create_file_with_sample_ids(train_samples, os.path.join(segmentation_subdir, f"train.txt"))
    create_file_with_sample_ids(test_samples, os.path.join(segmentation_subdir, f"test.txt"))
    create_file_with_sample_ids(trainval_samples, os.path.join(segmentation_subdir, f"trainval.txt"))
    create_file_with_sample_ids(val_samples, os.path.join(segmentation_subdir, f"val.txt"))

    return obj_classes


def create_sample_id_to_is_class_exist_file(obj_class, samples, out_file_path):
    sampleid_to_is_exist = []
    for sample in samples:
        class_is_in_sample = "-1"
        for obj in sample.objects:
            if obj.name == obj_class:
                class_is_in_sample = "1"
                break  # to the next sample

        sampleid_to_is_exist.append((sample.id, class_is_in_sample))

    with open(out_file_path, mode="w") as f:
        for pair in sampleid_to_is_exist:
            f.write(f"{pair[0]} {pair[1]}\n")

    return sampleid_to_is_exist


def create_file_with_sample_ids(samples: List[DatasetSample], out_file_path):
    with open(out_file_path, mode="w") as f:
        for s in samples:
            f.write(f"{s.id}\n")


def update_and_copy_samples_to_output_folders(dataset_samples, AnnotationsDir, jpegImagesDir) -> List[
    DatasetSample]:
    # === Fill AnnotationsDir
    # ! just copy annot files - thea are already in appropriate format

    updated_samples = []
    for n, sample in enumerate(dataset_samples):
        new_annot_path = os.path.join(AnnotationsDir, sample.id + ".xml")
        new_pict_path = os.path.join(jpegImagesDir, sample.id + ".jpg")

        shutil.copyfile(sample.annot_path, new_annot_path)
        shutil.copyfile(sample.pict_path, new_pict_path)

        # set paths from new locations
        sample.pict_path = new_pict_path
        sample.annot_path = new_annot_path
        updated_samples.append(sample)

    return updated_samples


def create_dir(path):
    if os.path.exists(path):
        log.info(f"{path} exists")
    else:
        os.mkdir(path)


def get_samples_with_id_assigning(dataset_path, first_id=1) -> List[DatasetSample]:
    samples = []
    n = first_id
    for filename in os.listdir(dataset_path):
        f = os.path.join(dataset_path, filename)

        # Construct samples from label-files ...
        if os.path.isfile(f) and filename.endswith(".xml"):
            objects = get_objects_from_annotation_file(f)
            log.debug(f"from {filename} got {len(objects)} objects")

            sample_id = str(n).zfill(6)
            n = n + 1

            pict_path = os.path.splitext(f)[0] + ".jpg"
            samples.append(DatasetSample(sample_id, pict_path=pict_path, objects=objects, annot_path=f))

    log.info(f"== From {dataset_path} got {len(samples)} dataset samples ===")
    return samples


def get_objects_from_annotation_file(label_file_path) -> List:
    with open(label_file_path, 'r') as file:
        data = file.read()

    annotation = objectify.fromstring(data)
    return annotation.object


def get_object_classes(samples: List[DatasetSample]) -> List[str]:
    cls = set()
    for sample in samples:
        for obj in sample.objects:
            cls.add(obj.name)
    log.info(f"Get {len(cls)} classes of objects ")
    return list(cls)
