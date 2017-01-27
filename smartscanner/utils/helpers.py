import cv2
import numpy as np

def expand_line(pt1, pt2, x_range, y_range):
    '''

    :param pt1: first point of line
    :param pt2: second point of line
    :param x_range: x coordinates where to expand the line
    :param y_range: y coordinates where to expand the line
    :return: (x1_expanded, y1_expanded), (x2_expanded y2_expanded)
    '''

    # line0 = [float(x) for x in line_in[0]]
    # line1 = [float(x) for x in line_in[1]]
    # line = (line0, line1)

    x1, y1 = pt1
    x2, y2 = pt2
    x_min, x_max = x_range
    y_min, y_max = y_range

    if x1 != x2:
        k = (y2 - y1) / (x2 - x1)
        q = y1 - k * x1

        if (q > 1) and (q < y_max):
            xx1 = 0
            yy1 = q
        elif q < 0:
            xx1 = -q / k
            yy1 = 0
        else:
            xx1 = (y_max - q) / k
            yy1 = y_max

        q2 = k * x_max + q

        if (q2 > 1) and (q2 < y_max):
            xx2 = x_max
            yy2 = q2
        elif q2 < 0:
            xx2 = -q / k
            yy2 = 0
        else:
            xx2 = (y_max - q) / k
            yy2 = y_max
    else:
        xx1 = xx2 = x1
        yy1 = 1
        yy2 = y_max

    return (int(xx1), int(yy1)), (int(xx2), int(yy2))


if __name__ == '__main__':
    im = np.zeros((100, 100, 3))
    line1 = ((20, 10), (40, 30))
    line1_exp = expand_line(line1[0], line1[1], (0, im.shape[1]), (0, im.shape[0]))

    im_vis = im.copy()
    cv2.line(im_vis, line1_exp[0], line1_exp[1], (0, 0, 255), 2)
    cv2.line(im_vis, line1[0], line1[1], (0, 255, 0), 2)

    cv2.imshow('line expansion', im_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()