#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import sys

import boxes as bboxs_util
import cv2
import h5py
import scipy.misc
import segms as segms_util
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *


def instances2dict(imageFileList, verbose=False):
    imgCount = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            instanceObj = Instance(imgNp, instanceId)
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                _, contour, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict["contours"] = polygons
            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=" ")
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict


def main(argv):
    fileList = []
    if len(argv) > 2:
        for arg in argv:
            if "png" in arg:
                fileList.append(arg)
    instances2dict(fileList, True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("--dataset", help="cityscapes_car_only", default=None, type=str)
    parser.add_argument(
        "--outdir", help="output dir for json files", default=None, type=str
    )
    parser.add_argument(
        "--datadir",
        help="data dir for annotations to be converted",
        default=None,
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


# for Cityscapes
def getLabelID(self, instID):
    if instID < 1000:
        return instID
    else:
        return int(instID / 1000)


def convert_cityscapes_car_only(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        "gtFine_val",
        "gtFine_train",
        # 'gtFine_test',
        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        "gtFine_trainvaltest/gtFine/val",
        "gtFine_trainvaltest/gtFine/train",
        # 'gtFine_trainvaltest/gtFine/test',
        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = "caronly_filtered_%s.json"
    ends_in = "%s_polygons.json"
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        "car",
    ]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print("Starting %s" % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in % data_set.split("_")[0]):
                    if len(images) % 50 == 0:
                        print(
                            "Processed %s images, %s annotations"
                            % (len(images), len(annotations))
                        )
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image["id"] = img_id
                    img_id += 1

                    image["width"] = json_ann["imgWidth"]
                    image["height"] = json_ann["imgHeight"]
                    image["file_name"] = (
                        filename[: -len(ends_in % data_set.split("_")[0])]
                        + "leftImg8bit.png"
                    )
                    image["seg_file_name"] = (
                        filename[: -len(ends_in % data_set.split("_")[0])]
                        + "%s_instanceIds.png" % data_set.split("_")[0]
                    )
                    images.append(image)

                    fullname = os.path.join(root, image["seg_file_name"])
                    objects = instances2dict([fullname], verbose=False)[fullname]

                    for object_cls in objects:
                        if object_cls not in category_instancesonly:
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:
                            if obj["contours"] == []:
                                print("Warning: empty contours.")
                                continue  # skip non-instance categories

                            len_p = [len(p) for p in obj["contours"]]
                            if min(len_p) <= 4:
                                print("Warning: invalid contours.")
                                continue  # skip non-instance categories

                            ann = {}
                            ann["id"] = ann_id
                            ann_id += 1
                            ann["image_id"] = image["id"]
                            ann["segmentation"] = obj["contours"]

                            if object_cls not in category_dict:
                                category_dict[object_cls] = cat_id
                                cat_id += 1
                            ann["category_id"] = category_dict[object_cls]
                            ann["iscrowd"] = 0
                            ann["area"] = obj["pixelCount"]
                            ann["bbox"] = bboxs_util.xyxy_to_xywh(
                                segms_util.polys_to_boxes([ann["segmentation"]])
                            ).tolist()[0]

                            annotations.append(ann)

        ann_dict["images"] = images
        categories = [
            {"id": category_dict[name], "name": name} for name in category_dict
        ]
        ann_dict["categories"] = categories
        ann_dict["annotations"] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        print(categories)
        with open(os.path.join(out_dir, json_name % data_set), "w") as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "cityscapes_car_only":
        convert_cityscapes_car_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
