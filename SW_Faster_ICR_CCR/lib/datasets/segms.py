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

"""Functions for interacting with segmentation masks in the COCO format.

The following terms are used in this module
    mask: a binary mask encoded as a 2D numpy array
    segm: a segmentation mask in one of the two COCO formats (polygon or RLE)
    polygon: COCO's polygon format
    RLE: COCO's run length encoding format
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pycocotools.mask as mask_util

# Type used for storing masks in polygon format
_POLY_TYPE = list
# Type used for storing masks in RLE format
_RLE_TYPE = dict


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys
