import cv2
import numpy as np


def check_range(coor, range):
    inside = False
    if range[0] < coor < range[1]:
        inside = True
    return inside


def in_img(pt, shape):
    inside = False
    if 0 < pt[0] < shape[1]:
        if 0 < pt[1] < shape[0]:
            inside = True
    return inside


def expand_line(pt1, pt2, x_range, y_range):
    top_ax = ((0, 0), (100, 0))
    bottom_ax = ((0, y_range[1]), (100, y_range[1]))
    left_ax = ((0, 0), (0, 100))
    right_ax = ((x_range[1], 0), (x_range[1], 100))

    top_int = tuple(line_intersect((pt1, pt2), top_ax).astype(np.int))
    left_int = tuple(line_intersect((pt1, pt2), left_ax).astype(np.int))
    bottom_int = tuple(line_intersect((pt1, pt2), bottom_ax).astype(np.int))
    right_int = tuple(line_intersect((pt1, pt2), right_ax).astype(np.int))

    inters = []
    if check_range(top_int[0], x_range):
        inters.append(top_int)
    if check_range(left_int[1], y_range):
        inters.append(left_int)
    if check_range(right_int[1], y_range):
        inters.append(right_int)
    if check_range(bottom_int[0], x_range):
        inters.append(bottom_int)

    return inters


def line_length(pts):
    length = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
    return length


def get_poly(pts):
    pts = np.array(pts)
    try:
        coeff = np.polyfit(pts[:, 0], pts[:, 1], 1)
    except Warning:  # pokud se u bodu jedne primky shoduji x-ove nebo y-ove souradnice, hlasi to warning a vypocet je nepresny
        if pts[0, 0] == pts[1, 0]:
            pts[0, 0] += 1
        else:
            pts[0, 1] += 1
        coeff = np.polyfit(pts[:, 0], pts[:, 1], 1)

    poly = np.poly1d(coeff)

    return poly


def line_angle(ln1, ln2):
    poly1 = get_poly(ln1)
    poly2 = get_poly(ln2)

    # to ensure that coeffs have always two elements
    coeffs1 = list(poly1.coeffs)
    if len(coeffs1) == 1:
        coeffs1.insert(0, 0)

    coeffs2 = list(poly2.coeffs)
    if len(coeffs2) == 1:
        coeffs2.insert(0, 0)

    n1 = (coeffs1[0], -1, coeffs1[1])
    n2 = (coeffs2[0], -1, coeffs2[1])

    try:
        val = np.dot(n1[:2], n2[:2]) / (np.linalg.norm(n1[:2]) * np.linalg.norm(n2[:2]))
        if val > 1:
            val = 1
        theta = np.arccos(val)
        theta = np.rad2deg(theta)
    except:
        pass

    return theta


def line_position(ln1, ln2):
    angle = line_angle(ln1, ln2)
    pos = 'gen'  # default is general

    if 80 < angle < 100:
        pos = 'perp'  # perpendicular
    elif 170 < angle < 190 or -10 < angle < 10:
        pos = 'para'  # parallel

    return pos


def collinear(ln1, ln2):
    pt1 = ln1[0]
    pt2 = ln1[1]
    pt3 = ln2[0]
    pt4 = ln2[1]
    col1 = pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1])
    col2 = pt1[0] * (pt2[1] - pt4[1]) + pt2[0] * (pt4[1] - pt1[1]) + pt4[0] * (pt1[1] - pt2[1])
    return col1, col2


def merge_lines(lines, show=False, show_now=True):
    col_T = 1000
    types = []

    for l in lines:
        found = False
        for t in types:
            col1, col2 = collinear(l, t)
            # if line l is collinear with line t, got to next one
            if abs(col1 + col2) < col_T:
                found = True
                break
        if not found:
            types.append(l)

    if show:
        im = np.zeros((800, 800, 3))
        im_l = im.copy()
        for l in lines:
            cv2.line(im_l, l[0], l[1], (255, 0, 0), 2)
        im_t = im.copy()
        for l in types:
            cv2.line(im_t, l[0], l[1], (255, 0, 0), 2)

        cv2.imshow('lines', im_l)
        cv2.imshow('types', im_t)
        if show_now:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return types


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def line_intersect(ln1, ln2): # a1, a2, b1, b2
    ln1 = np.array(ln1)
    ln2 = np.array(ln2)
    da = ln1[1] - ln1[0]
    db = ln2[1] - ln2[0]
    dp = ln1[0] - ln2[0]
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + ln2[0]


if __name__ == '__main__':
    im = np.zeros((100, 100, 3))
    line1 = ((20, 10), (40, 30))
    # line1_exp = expand_line(line1[0], line1[1], (0, im.shape[1]), (0, im.shape[0]))
    #
    # im_vis = im.copy()
    # cv2.line(im_vis, line1_exp[0], line1_exp[1], (0, 0, 255), 2)
    # cv2.line(im_vis, line1[0], line1[1], (0, 255, 0), 2)
    #
    # cv2.imshow('line expansion', im_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ll = line_length(((185, 73), (105, 62)))
    print ll