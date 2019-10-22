import argparse
import errno
import os
import tempfile
from typing import List
from xml.etree import ElementTree

import contextlib2
from object_detection.dataset_tools.tf_record_creation_util import open_sharded_output_tfrecords
from object_detection.utils import dataset_util
from vod_converter.converter import convert
from vod_converter.kitti import KITTIEgestor
from vod_converter.voc import VOCIngestor
import tensorflow as tf

from vod_convert.utils import matching_ids


# ------------------------------------------------------------------------------
def _create_tf_example(
        pascal_file_path: str,
        image_file_path: str,
        labels: List[str],
        include_image_data: bool = False,
) -> tf.train.Example:

    # get the encoded image data as a string object
    with tf.io.gfile.GFile(image_file_path, "rb") as fid:
        image_string = fid.read()

    # parse the file name and encoding from the image path
    image_file_name = image_file_path.split(os.path.sep)[-1]
    if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg"):
        image_format = b'jpeg'
        decoded_image = tf.image.decode_jpeg(image_string)
    elif image_file_name.endswith(".png"):
        image_format = b'png'
        decoded_image = tf.image.decode_png(image_string)
    else:
        raise ValueError(f"Unsupported image file extension: {image_file_path}")
    (height, width) = decoded_image.shape[:2]

    # lists of bounding box coordinates and class labels/indices, the final
    # number of elements in each list will equal the number of bounding boxes
    xmins = []  # normalized left x coordinate of bounding box
    xmaxs = []  # normalized right x coordinate of bounding box
    ymins = []  # normalized top y coordinate of bounding box
    ymaxs = []  # normalized bottom y coordinate of bounding box
    class_labels = []  # object class name (label) of object within bounding box
    class_ids = []  # integer class id of object within bounding box

    # parse the PASCAL VOC XML into an ElementTree and get the root
    tree = ElementTree.parse(pascal_file_path)
    root = tree.getroot()

    # get the image dimensions
    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)
    if (image_height != height) or (image_width != width):
        raise ValueError(
            f"Annotation dimension(s) [H: {image_height}, W: {image_width}] "
            f"are inconsistent with the actual image [H: {height}, W: {width}]")

    # for each bounding box get the coordinates and class label/index
    # as well as the bounding box's width and height in terms of
    # a decimal fraction of the total image dimension
    for obj in root.iter("object"):

        # get the label index based on the annotation's object label
        object_label = obj.find("name")
        if object_label:
            label = object_label.text.strip()
            try:
                # we add one to the index since TFRecords are typically used in
                # conjunction with protobuf text files that use 1-based indices
                label_index = labels.index(label) + 1
            except ValueError:
                # we didn't find the box's label in our list
                # of known labels so we skip the box altogether
                continue
        else:
            # we somehow have an object without a name/label, skip it
            continue

        # get the bounding box coordinates, normalized (between 0.0 and 1.0)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text) / width
        ymin = int(bbox.find("ymin").text) / height
        xmax = int(bbox.find("xmax").text) / width
        ymax = int(bbox.find("ymax").text) / height

        # TODO perform validation (sanity check) on bounding box values

        # add to the bounding box lists
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        class_labels.append(label)
        class_ids.append(label_index)

    # convert our labels to bytes objects to facilitate loading as a bytes list feature
    bytes_labels = [label.encode('utf-8') for label in labels]

    # create a feature dictionary that we can use to create a Features object
    feature = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_file_name.encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(image_file_name.encode('utf-8')),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(bytes_labels),
      'image/object/class/label': dataset_util.int64_list_feature(class_ids),
    }
    if include_image_data:
        encoded_image_data = bytes(image_string)
        feature['image/encoded'] = dataset_util.bytes_feature(encoded_image_data)

    # create a training example object from the features described by the annotation
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


# ------------------------------------------------------------------------------
def pascal_to_tfrecords(
        pascal_dir: str,
        images_dir: str,
        tfrecord_path_base: str,
        labels: List[str],
        include_image_data: bool = False,
        num_shards: int = 1,
) -> int:
    """
    Builds TFRecord annotation files from PASCAL VOC annotation XML files.

    :param pascal_dir: directory containing input PASCAL VOC annotation
        XML files, all XML files in this directory matching to corresponding JPG
        files in the images directory will be converted to TFRecord format
    :param images_dir: directory containing image files corresponding to the
        PASCAL VOC annotation files to be converted to TFRecord format
    :param tfrecord_path_base: base file path for output TFRecord annotation
        files, final files will have this base path/name with shard number(s)
        appended -- for example <tfrecord_path_base>-00000-00002
    :param labels: list of object class labels
    :param include_image_data: whether or not the TFRecords should include
        the image data
    :param num_shards: the number of "shard" files that will result
    :return: 0 indicates success
    """

    # get corresponding image and annotation file IDs, i.e. the file IDs
    # that are shared between the images and PASCAL annotations directories,
    # these will be the file pairs that will be used to create TFExamples
    # that will be included in the final set of TFRecord files
    annotation_ext = ".xml"
    image_ext = ".jpg"
    file_ids = matching_ids(pascal_dir, images_dir, annotation_ext, image_ext)

    # in case the output directory path doesn't exist we create it here
    try:
        os.makedirs(os.path.dirname(tfrecord_path_base))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # use an ExitStack context manager
    with contextlib2.ExitStack() as tf_record_close_stack:

        # open all TFRecord shards for writing and add them to the exit stack
        output_tfrecords = \
            open_sharded_output_tfrecords(
                tf_record_close_stack,
                tfrecord_path_base,
                num_shards,
            )

        # for each file ID create a corresponding Example
        # and write it into one of the TFRecord shards
        for index, file_id in enumerate(file_ids):
            annotation_path = os.path.join(pascal_dir, file_id + annotation_ext)
            image_path = os.path.join(images_dir, file_id + image_ext)
            tf_example = _create_tf_example(annotation_path, image_path, labels, include_image_data)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

    return 0


# ------------------------------------------------------------------------------
def pascal_to_kitti(
        pascal_dir: str,
        images_dir: str,
        kitti_dir: str,
) -> int:
    """
    Builds KITTI annotation files from PASCAL VOC annotation XML files.

    :param pascal_dir: directory containing input PASCAL VOC annotation
        XML files, all XML files in this directory matching to corresponding JPG
        files in the images directory will be converted to KITTI format
    :param images_dir: directory containing image files corresponding to the
        PASCAL VOC annotation files to be converted to KITTI format
    :param kitti_dir: directory where output KITTI annotation files will
        be written, will be created if it does not yes exist
    :return: the number of PASCAL VOC annotation files converted to KITTI format
    """

    # get list of file IDs of the PASCAL VOC annotations and corresponding images
    file_ids = matching_ids(pascal_dir, images_dir, ".xml", ".jpg")

    # create a temporary directory to contain everything in the structure
    # (VOC2012) expected by the vod_converter package we'll use for conversion
    with tempfile.TemporaryDirectory() as base_dir:

        # recreate the VOC2012 directory structure
        voc2012_dir = os.path.join(base_dir, "VOC2012")
        os.mkdir(voc2012_dir)
        jpegimages_dir = os.path.join(voc2012_dir, "JPEGImages")
        os.mkdir(jpegimages_dir)
        imagesets_dir = os.path.join(voc2012_dir, "ImageSets")
        os.mkdir(imagesets_dir)
        main_dir = os.path.join(imagesets_dir, "Main")
        os.mkdir(main_dir)
        voc_annotations_dir = os.path.join(voc2012_dir, "Annotations")
        os.mkdir(voc_annotations_dir)

        # link annotation and image files into the expected locations
        for file_id in file_ids:
            pascal_file_name = file_id + ".xml"
            image_file_name = file_id + ".jpg"
            pascal_file_path = os.path.join(pascal_dir, pascal_file_name)
            image_file_path = os.path.join(images_dir, image_file_name)
            voc_link = os.path.join(voc_annotations_dir, pascal_file_name)
            image_link = os.path.join(jpegimages_dir, image_file_name)
            os.symlink(pascal_file_path, voc_link)
            os.symlink(image_file_path, image_link)

        # create the "trainval.txt" file
        trainval_path = os.path.join(main_dir, "trainval.txt")
        with open(trainval_path, "w+") as trainval_file:
            for file_id in file_ids:
                trainval_file.write(f"{file_id}\n")

        # utilize the vod_converter to convert PASCAL VOC to KITTI
        success, msg = convert(
            from_path=base_dir,
            ingestor=VOCIngestor(),
            to_path=kitti_dir,
            egestor=KITTIEgestor(),
            select_only_known_labels=False,
            filter_images_without_labels=True,
        )
        if not success:
            raise ValueError(f"Failed to convert from PASCAL to KITTI: {msg}")

    # return the number of annotations converted
    return len(file_ids)


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="path to directory containing input annotation files to be converted",
    )
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to directory containing input image files",
    )
    args_parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="path to directory for output annotation files after conversion",
    )
    args_parser.add_argument(
        "--in_format",
        required=True,
        type=str,
        default="pascal",
        choices=["pascal", ],
        help="format of output annotations",
    )
    args_parser.add_argument(
        "--out_format",
        required=True,
        type=str,
        default="tfrecord",
        choices=["kitti", "tfrecord"],
        help="format of output annotations",
    )
    args_parser.add_argument(
        "--shards",
        required=False,
        type=int,
        default=1,
        help="number of shard TFRecord files to produce",
    )
    args_parser.add_argument(
        "--labels",
        required=False,
        type=str,
        nargs='+',
        help="list of object class labels",
    )
    args_parser.add_argument(
        "--tfr_base_name",
        required=False,
        type=str,
        help="base name of output TFRecord files (shard numbering will be appended)",
    )
    args = vars(args_parser.parse_args())

    if (args["in_format"] == "pascal") and (args["out_format"] == "kitti"):

        # perform PASCAL to KITTI conversion
        pascal_to_kitti(args["annotations_dir"], args["images_dir"], args["out_dir"])

    elif (args["in_format"] == "pascal") and (args["out_format"] == "tfrecord"):

        # perform PASCAL to TFRecord conversion
        pascal_to_tfrecords(
            args["annotations_dir"],
            args["images_dir"],
            os.path.join(args["out_dir"], args["tfr_base_name"]),
            args["labels"],
            num_shards=args["shards"],
        )
    else:
        raise ValueError(
            "Unsupported format conversion: "
            f"{args['in_format']} to {args['out_format']}",
        )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: PASCAL to KITTI
    $ python convert.py --annotations_dir ~/datasets/handgun/annotations/pascal \
        --out_dir ~/datasets/handgun/annotations/kitti \
        --in_format pascal --out_format kitti \
        --images_dir ~/datasets/handgun/images

    Usage: PASCAL to TFRecord
    $ python convert.py --annotations_dir ~/datasets/handgun/annotations/pascal \
        --out_dir ~/datasets/handgun/annotations/tfrecord \
        --in_format pascal --out_format tfrecord \
        --images_dir ~/datasets/handgun/images \
        --shards 4 \
        --labels handgun rifle \
        --tfr_base_name weapons.tfrecord
    """

    main()
