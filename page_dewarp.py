#!/usr/bin/env python
######################################################################
# page_dewarp.py - Proof-of-concept of page-dewarping based on a
# "cubic sheet" model. Requires OpenCV (version 3 or greater),
# PIL/Pillow, and scipy.optimize.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: MIT License (see LICENSE.txt)
######################################################################

import os
import sys
import datetime
import cv2
import Image
import numpy as np
import scipy.optimize

PAGE_MARGIN_X = 5       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 15      # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0       # how much to zoom output relative to *original* image
OUTPUT_DPI = 300        # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16     # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55     # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15     # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2     # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5   # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10 # max reduced px thickness of detected text contour

EDGE_MAX_OX = 1.0       # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0 # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0  # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5    # maximum change in angle allowed between adjacent contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30     # minimum reduced px width for span

SPAN_PX_PER_STEP = 20   # reduced px spacing for sampling along spans

FOCAL_LENGTH = 1.2      # normalized focal length of camera

DEBUG_LEVEL = 0         # 0=none, 1=some, 2=lots, 3=all

DEBUG_OUTPUT = 'file'   # file, screen, both

WINDOW_NAME = 'Dewarp'  # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)

######################################################################
# show debug image

def debug_show(basefile, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = basefile + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass

######################################################################
# round to nearest multiple of factor, used for choosing output size

def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem

######################################################################
# convert pixel coordinates to normalized coordinates

def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
    return (pts - offset) * scl

######################################################################
# convert normalized coordinates to pixel coordinates

def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height], dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval

######################################################################
# tuple from flattened array

def fltp(point):
    return tuple(point.astype(int).flatten())

######################################################################
# draw point correspondences from points in normalized coords and
# return marked-up copy of image

def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display

######################################################################
# construct default parameter vector from camera projection matrix
# and points

def get_default_params(corners, ycoords, xcoords):

    # page width and height
    pw = np.linalg.norm(corners[1] - corners[0])
    ph = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (pw, ph)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [pw, 0, 0],
        [pw, ph, 0],
        [0, ph, 0]])

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    kp_lengths = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))

    return rough_dims, kp_lengths, params

######################################################################
# project 2D xy points to 3D using cubic slopes and
# intrinsics/extrinsics.

def project_xy(xy, p):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(p[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])

    xy = xy.reshape((-1, 2))
    z = np.polyval(poly, xy[:, 0])

    objpoints = np.hstack((xy, z.reshape((-1, 1))))

    ipts, _ = cv2.projectPoints(objpoints,
                                p[RVEC_IDX],
                                p[TVEC_IDX],
                                K, np.zeros(5))

    return ipts

######################################################################
# project all of the 2D keypoints stored in the parameter vector to 3D
# using the function above

def project_keypoints(kp_lengths, p):

    nspans = len(kp_lengths)

    ycoords = p[8:8+nspans]
    xcoords = []

    start = 8+nspans
    npts = 1
    for l in kp_lengths:
        end = start+l
        xcoords.append(p[start:end])
        npts += l
        start = end

    assert end == len(p)

    xy = np.zeros((npts, 2))

    start = 1

    for y, xc in zip(ycoords, xcoords):
        end = start + len(xc)
        xy[start:end, 0] = xc
        xy[start:end, 1] = y
        start = end

    return project_xy(xy, p)

######################################################################
# downscale a big image to fit onto the screen, return the scale too

def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    h, w = src.shape[:2]

    fx = float(w)/maxw
    fy = float(h)/maxh

    scl = int(np.ceil(max(fx, fy)))

    if scl > 1.0:
        f = 1.0/scl
        img = cv2.resize(src, (0, 0), None, f, f, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img

######################################################################
# box-shaped morphological structuring element

def box(w, h):
    return np.ones((h, w), dtype=np.uint8)

######################################################################
# disc-shaped morphological structuring element

def disc(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (r, r), (-1, -1))

######################################################################
# get usable portion of image. in earlier versions of the program, I
# had tried to automate this, it didn't work very well, so now I just
# have predefined page margins that get discarded.

def get_page_extents(small):

    h, w = small.shape[:2]

    x0 = PAGE_MARGIN_X
    y0 = PAGE_MARGIN_Y
    x1 = w-PAGE_MARGIN_X
    y1 = h-PAGE_MARGIN_Y

    page = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(page, (x0, y0), (x1, y1), (255, 255, 255), -1)

    outline = np.array([
        [x0, y0],
        [x0, y1],
        [x1, y1],
        [x1, y0]])

    return page, outline

######################################################################
# extract mask of text or horiz. lines from image

def get_mask(basename, small, pagemask, masktype):

    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.3, 'eroded', mask)

    else:

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show(basename, 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


######################################################################
# given two intervals of the form (lo, hi) compute their overlap
# (positive) or distance (negative)

def interval_measure_overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])

######################################################################
# get angular distance mod 2pi

def angle_dist(b, a):

    diff = b - a

    while diff > np.pi:
        diff -= 2*np.pi

    while diff < -np.pi:
        diff += 2*np.pi

    return np.abs(diff)

######################################################################
# class to hold information about a contour

class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask

        m = cv2.moments(contour)

        s00 = m['m00']
        s10 = m['m10']
        s01 = m['m01']
        c20 = m['mu20']
        c11 = m['mu11']
        c02 = m['mu02']

        mx = s10 / s00
        my = s01 / s00

        A = np.array([
            [c20, c11],
            [c11, c02]
        ]) / s00

        _, U, _ = cv2.SVDecomp(A)

        ux = U[0, 0]
        uy = U[1, 0]

        self.center = np.array([mx, my])
        self.tangent = np.array([ux, uy])

        c, s = tuple(self.tangent)
        self.angle = np.arctan2(s, c)

        clx = map(self.proj_x, contour)

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.p0 = self.center + self.tangent * lxmin
        self.p1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None
        self.pred_dist = None
        self.succ_dist = None

    def proj_x(self, pt):
        return np.dot(self.tangent, pt.flatten()-self.center)

    def local_overlap(self, other):
        x0 = self.proj_x(other.p0)
        x1 = self.proj_x(other.p1)
        return interval_measure_overlap(self.local_xrng, (x0, x1))

######################################################################
# generate a candidate edge between two contours

def generate_candidate_edge(cinfo_a, cinfo_b):

    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    if cinfo_a.p0[0] > cinfo_b.p1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    oax = cinfo_a.local_overlap(cinfo_b)
    obx = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    c, s = tuple(overall_tangent)
    overall_angle = np.arctan2(s, c)

    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

    # we want the largest overlap in x to be small
    ox = max(oax, obx)

    dist = np.linalg.norm(cinfo_b.p0 - cinfo_a.p1)

    if (dist > EDGE_MAX_LENGTH or
            ox > EDGE_MAX_OX or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle*EDGE_ANGLE_COST
        return  (score, cinfo_a, cinfo_b)

######################################################################
# make a tight mask

def make_tight_mask(contour, x, y, w, h):

    tight_mask = np.zeros((h, w), dtype=np.uint8)
    tight_contour = contour - np.array((x, y)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask

######################################################################
# obtain contours from a text mask

def get_contours(basename, small, pagemask, masktype):

    mask = get_mask(basename, small, pagemask, masktype)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        if (w < TEXT_MIN_WIDTH or
                h < TEXT_MIN_HEIGHT or
                w < TEXT_MIN_ASPECT*h):
            continue

        tight_mask = make_tight_mask(contour, x, y, w, h)

        thickness = tight_mask.sum(axis=0).max()
        if thickness > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(basename, small, contours_out)

    return contours_out

######################################################################
# a span is a list of ContourInfo objects. this gets its width by
# summing up the width of each one.

def span_width(span):
    return sum([cinfo.local_xrng[1] - cinfo.local_xrng[0]
                for cinfo in span])

######################################################################
# given a list of ContourInfo objects, assemble them into spans.
# this is a greedy algorithm that runs in the worst-case O(n^2) time
# for n spans in the list

def assemble_spans(basename, small, pagemask, orig_cinfo_list):

    # sort list
    cinfo_list = sorted(orig_cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generate all candidate edges
    candidate_edges = []

    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    candidate_edges.sort()

    # for each candidate edge
    for _, cinfo_a, cinfo_b in candidate_edges:
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generate list of spans as output
    spans = []

    # until we have removed everything from the list
    while cinfo_list:

        # get the first on the list
        cinfo = cinfo_list[0]

        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []

        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            # set successor
            cinfo = cinfo.succ

        # add if long enough
        if span_width(cur_span) > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(basename, small, pagemask, spans)

    return spans

######################################################################

def sample_spans(shape, spans):

    all_points = []

    for span in spans:

        points = []

        for cinfo in span:

            cmask = cinfo.mask
            csum = cmask.sum(axis=0)
            maskx = (cmask.sum(axis=0) != 0)
            assert np.all(maskx)

            yvals = np.arange(cmask.shape[0]).reshape((-1, 1))
            totals = (yvals * cmask).sum(axis=0)
            means = totals / csum

            x0, y0 = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            start = ((len(csum)-1) % step) / 2

            points += [(x+x0, means[x]+y0)
                       for x in range(start, len(maskx), step)]

        points = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        points = pix2norm(shape, points)

        all_points.append(points)

    return all_points

######################################################################

def keypoints_from_samples(basename, small, pagemask, page_outline,
                           span_points):

    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:

        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)

        weight = np.linalg.norm(points[-1] - points[0])

        all_evecs += evec * weight
        all_weights += weight

    evec = all_evecs / all_weights

    bx = evec.flatten()
    if bx[0] < 0: bx = -bx

    c, s = tuple(bx)

    by = np.array([-s, c])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px = np.dot(pagecoords, bx)
    py = np.dot(pagecoords, by)

    px0 = px.min()
    px1 = px.max()

    py0 = py.min()
    py1 = py.max()

    p00 = px0 * bx + py0 * by
    p10 = px1 * bx + py0 * by
    p11 = px1 * bx + py1 * by
    p01 = px0 * bx + py1 * by

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px = np.dot(pts, bx)
        py = np.dot(pts, by)
        ycoords.append(py.mean() - py0)
        xcoords.append(px - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(basename, small, span_points, corners)

    return corners, np.array(ycoords), xcoords

######################################################################

def visualize_contours(basename, small, cinfo_list):

    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(display, fltp(cinfo.p0), fltp(cinfo.p1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(basename, 1, 'contours', display)

######################################################################

def visualize_spans(basename, small, pagemask, spans):

    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i*3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)
    display[pagemask == 0] /= 4

    debug_show(basename, 2, 'spans', display)

######################################################################

def visualize_span_points(basename, small, span_points, corners):

    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())
        dp0 = dps.min()
        dp1 = dps.max()

        p0 = mean + small_evec * (dp0-dpm)
        p1 = mean + small_evec * (dp1-dpm)

        for pt in points:
            cv2.circle(display, fltp(pt), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(p0), fltp(p1), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    debug_show(basename, 3, 'span points', display)

######################################################################

def imgsize(img):
    h, w = img.shape[:2]
    return '{}x{}'.format(w, h)

######################################################################

def optimize_params(basename, small, dstpoints, kp_lengths, params):

    points_for_params = lambda p: project_keypoints(kp_lengths, p)
    err = lambda ppts: np.sum((dstpoints - ppts)**2)
    objective = lambda p: err(points_for_params(p))

    print '  initial objective is', objective(params)

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(kp_lengths, params)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(basename, 4, 'keypoints before', display)


    print '  optimizing', len(params), 'parameters...'
    start = datetime.datetime.now()
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end = datetime.datetime.now()
    print '  optimization took', round((end-start).total_seconds(), 2), 'seconds'
    print '  final objective is', res.fun
    params = res.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(kp_lengths, params)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(basename, 5, 'keypoints after', display)

    return params

######################################################################

def get_page_dims(corners, rough_dims, params):

    page2img = lambda xy: project_xy(xy, params)

    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)

    err = lambda proj_br: np.sum((dst_br - proj_br.flatten())**2)
    objective = lambda dims: err(page2img(dims))

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print '  got page dims', dims[0], 'x', dims[1]

    return dims

######################################################################

def remap_image(basename, img, page_dims, params):

    dh = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]

    dh = round_nearest_multiple(dh, REMAP_DECIMATE)

    dw = round_nearest_multiple(dh * page_dims[0] / page_dims[1],
                                REMAP_DECIMATE)

    print '  output will be {}x{}'.format(dw, dh)

    dh_small = dh / REMAP_DECIMATE
    dw_small = dw / REMAP_DECIMATE

    oxrng = np.linspace(0, page_dims[0], dw_small)
    oyrng = np.linspace(0, page_dims[1], dh_small)

    ox, oy = np.meshgrid(oxrng, oyrng)

    xy = np.hstack((ox.flatten().reshape((-1, 1)),
                    oy.flatten().reshape((-1, 1)))).astype(np.float32)

    ipts = project_xy(xy, params)

    ipts = norm2pix(img.shape, ipts, False)

    ix = ipts[:, 0, 0].reshape(ox.shape)
    iy = ipts[:, 0, 1].reshape(oy.shape)

    ix = cv2.resize(ix, (dw, dh), interpolation=cv2.INTER_CUBIC)
    iy = cv2.resize(iy, (dw, dh), interpolation=cv2.INTER_CUBIC)

    igray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    remapped = cv2.remap(igray, ix, iy, cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)

    thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   ADAPTIVE_WINSZ,
                                   25)

    im = Image.fromarray(thresh)
    im = im.convert('1')

    threshfile = basename + '_thresh.png'
    im.save(threshfile, dpi=(OUTPUT_DPI, OUTPUT_DPI))

    return threshfile

    if DEBUG_LEVEL >= 1:
        display = resize_to_screen(thresh)
        debug_show(basename, 6, 'output', display)

######################################################################

def main():

    if len(sys.argv) < 2:
        print 'usage:', sys.argv[0], 'IMAGE1 [IMAGE2 ...]'
        sys.exit(0)

    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)

    outfiles = []

    for imgfile in sys.argv[1:]:

        img = cv2.imread(imgfile)
        small = resize_to_screen(img)
        basename = os.path.basename(imgfile)
        head, _ = os.path.splitext(basename)

        print 'loaded', basename, 'with size', imgsize(img),
        print 'and resized to', imgsize(small)

        if DEBUG_LEVEL >= 3:
            debug_show(head, 0.0, 'original', small)

        pagemask, page_outline = get_page_extents(small)

        cinfo_list = get_contours(head, small, pagemask, 'text')
        spans = assemble_spans(head, small, pagemask, cinfo_list)

        if len(spans) < 3:
            print '  detecting lines because only', len(spans), 'text spans'
            cinfo_list = get_contours(head, small, pagemask, 'line')
            spans2 = assemble_spans(head, small, pagemask, cinfo_list)
            if len(spans2) > len(spans):
                spans = spans2


        if len(spans) < 1:
            print 'skipping', head, 'because only', len(spans), 'spans'
            continue

        span_points = sample_spans(small.shape, spans)

        print '  got', len(spans), 'spans',
        print 'with', sum([len(pts) for pts in span_points]), 'points.'

        corners, ycoords, xcoords = keypoints_from_samples(head, small,
                                                           pagemask, page_outline,
                                                           span_points)

        rough_dims, kp_lengths, params = get_default_params(corners,
                                                            ycoords,
                                                            xcoords)

        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                              tuple(span_points))

        params = optimize_params(head, small,
                                 dstpoints,
                                 kp_lengths, params)

        page_dims = get_page_dims(corners, rough_dims, params)

        outfile = remap_image(head, img, page_dims, params)

        outfiles.append(outfile)

        print '  wrote', outfile
        print

    print 'to convert to PDF (requires ImageMagick):'
    print '  convert -compress Group4 ' + ' '.join(outfiles) + ' output.pdf'

######################################################################

if __name__ == '__main__':
    main()

