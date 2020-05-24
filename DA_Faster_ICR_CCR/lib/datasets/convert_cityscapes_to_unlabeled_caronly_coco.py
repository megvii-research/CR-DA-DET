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

import cityscapesscripts.evaluation.instances2dict as cs
import h5py
import scipy.misc


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument(
        "--dataset", help="cityscapes_unlabeled_car_only", default=None, type=str
    )
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
    json_name = "caronly_filtered_unlabeled_%s.json"
    ends_in = "%s_polygons.json"
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        "car",
    ]
    category_dict["car"] = cat_id

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
                    objects = cs.instances2dict([fullname], verbose=False)[fullname]

                    bbox = []
                    # x
                    bbox.append(0)
                    # y
                    bbox.append(0)
                    # w
                    bbox.append(int(json_ann["imgWidth"]))
                    # h
                    bbox.append(int(json_ann["imgHeight"]))

                    seg = []
                    # bbox[] is x,y,w,h
                    # left_top
                    seg.append(bbox[0])
                    seg.append(bbox[1])
                    # left_bottom
                    seg.append(bbox[0])
                    seg.append(bbox[1] + bbox[3])
                    # right_bottom
                    seg.append(bbox[0] + bbox[2])
                    seg.append(bbox[1] + bbox[3])
                    # right_top
                    seg.append(bbox[0] + bbox[2])
                    seg.append(bbox[1])

                    ann = {}
                    ann["id"] = ann_id
                    ann_id += 1
                    ann["image_id"] = image["id"]
                    ann["segmentation"] = [seg]
                    ann["category_id"] = 1
                    ann["iscrowd"] = 0
                    ann["area"] = bbox[2] * bbox[3]
                    ann["bbox"] = bbox

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
    if args.dataset == "cityscapes_unlabeled_car_only":
        convert_cityscapes_car_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
