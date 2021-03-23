import math
import numpy as np


def intersection_over_union(a_quad_list, b_quad_list):
    import Polygon

    poly_a = Polygon.Polygon(np.array(a_quad_list).reshape(-1, 2))
    poly_b = Polygon.Polygon(np.array(b_quad_list).reshape(-1, 2))

    poly_inter = poly_a & poly_b

    area_a = poly_a.area()
    area_b = poly_b.area()

    area_inter = poly_inter.area()
    area_union = area_a + area_b - area_inter

    area_min = min(area_a, area_b)
    if area_min < area_inter < area_min * 1.0000000001:
        area_inter = area_min

    jaccard_index = area_inter / area_union

    return jaccard_index


def intersection_over_union_with_ground_truth_normalization(prediction, ground_truth, tpl_size):
    import Polygon
    import cv2
    # Reference : https://github.com/jchazalon/smartdoc15-ch1-eval
    #MIT License
    #
    #Copyright (c) 2015 Joseph Chazalon - University of La Rochelle, France
    #                   joseph(dot)chazalon(at)univ-lr(dot)fr
    #Copyright (c) 2015 Marçal Rusiñol <marcal(at)cvc(dot)uab(dot)cat>
    #                   CVC / Universitat Autònoma de Barcelona, Spain
    #
    #Permission is hereby granted, free of charge, to any person obtaining a copy
    #of this software and associated documentation files (the "Software"), to deal
    #in the Software without restriction, including without limitation the rights
    #to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    #copies of the Software, and to permit persons to whom the Software is
    #furnished to do so, subject to the following conditions:
    #
    #The above copyright notice and this permission notice shall be included in all
    #copies or substantial portions of the Software.
    #
    #THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    #IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    #FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    #AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    #LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    #OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    #SOFTWARE.
    
    target_width = tpl_size[0]
    target_height = tpl_size[1]

    object_coord_target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
                                   np.float32)

    matrix = cv2.getPerspectiveTransform(np.array(ground_truth, np.float32).reshape(-1, 2),
                                         object_coord_target.reshape(-1, 2))

    test_coords = cv2.perspectiveTransform(np.array(prediction, np.float32).reshape(-1, 1, 2), matrix)

    poly_target = Polygon.Polygon(object_coord_target.reshape(-1, 2))
    poly_test = Polygon.Polygon(test_coords.reshape(-1, 2))
    poly_inter = poly_target & poly_test

    area_target = poly_target.area()
    area_test = poly_test.area()
    area_inter = poly_inter.area()

    area_union = area_test + area_target - area_inter

    area_min = min(area_target, area_test)
    if area_min < area_inter < area_min * 1.0000000001:
        area_inter = area_min
    
    jaccard_index = area_inter / area_union
    return jaccard_index


def mean_intersection_over_union(a, b, image_size):
    import cv2

    width = image_size[0]
    height = image_size[1]

    a = np.array(a)
    b = np.array(b)

    img1 = np.zeros((height, width))
    a = a.astype(np.int32)
    cv2.fillConvexPoly(img1, a, 1)
    not_img1 = np.logical_not(img1)

    img2 = np.zeros((height, width))
    b = b.astype(np.int32)
    cv2.fillConvexPoly(img2, b, 1)
    not_img2 = np.logical_not(img2)

    inter = img1 * img2
    union = np.logical_or(img1, img2)
    if float(np.sum(union)) < 1:
        print(1, a, b, float(np.sum(union)))
        breakpoint()
    iou = float(np.sum(inter)) / float(np.sum(union))

    inter_back = not_img1 * not_img2
    union_back = np.logical_or(not_img1, not_img2)

    if float(np.sum(union_back)) < 1:
        print(2, a, b, float(np.sum(union_back)))
        breakpoint()
    iou_back = float(np.sum(inter_back)) / float(np.sum(union_back))
    return (iou + iou_back) / 2


def euc_distance(p1, p2):
    assert len(p1) == 2
    assert len(p2) == 2
    return math.sqrt(sum([(c1 - c2)**2 for (c1, c2) in zip(p1, p2)]))


def corner_residuals(run, ideal):
    return [euc_distance(p1, p2) for (p1, p2) in zip(run, ideal)]


def as_rect(size):
    assert len(size) == 2
    return [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]


def transform_point(h, p):
    d = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2]
    assert d != 0

    x = (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / d
    y = (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / d
    return x, y


def template_perimeter(size):
    return (size[0] + size[1]) * 2


def provide_regular_orientation(quad):
    q = np.array(quad)
    if np.cross(q[1] - q[0], q[2] - q[1]) > 0:
        return quad
    else:
        return [quad[1], quad[0], quad[3], quad[2]]


def residual_metric(run, ideal, size):
    import cv2

    run = provide_regular_orientation(run)
    ideal = provide_regular_orientation(ideal)

    matrix = cv2.getPerspectiveTransform(np.array(run, np.float32).reshape(-1, 2),
                                         np.array(as_rect(size), np.float32).reshape(-1, 2))

    ta = [transform_point(matrix, p) for p in ideal]
    residuals = corner_residuals(ta, as_rect(size))

    ta_rotated = [ta[3], ta[0], ta[1], ta[2]]
    residuals_rotated = corner_residuals(ta_rotated, as_rect(size))
    if max(residuals) > max(residuals_rotated):
        residuals = residuals_rotated
    ta_rotated = [ta[2], ta[3], ta[0], ta[1]]
    residuals_rotated = corner_residuals(ta_rotated, as_rect(size))
    if max(residuals) > max(residuals_rotated):
        residuals = residuals_rotated
    ta_rotated = [ta[1], ta[2], ta[3], ta[0]]
    residuals_rotated = corner_residuals(ta_rotated, as_rect(size))
    if max(residuals) > max(residuals_rotated):
        residuals = residuals_rotated

    score = max(residuals) / template_perimeter(size)

    return score
