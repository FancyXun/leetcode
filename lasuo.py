import unittest
import numpy as np
import cv2 as cv


def mazeMinSteps(maze, start, end):
    maze = np.asarray(maze)
    m, n = maze.shape
    # state状态表示是否走过: 0表示走过,1表示未走过
    state = np.zeros(shape=(m, n))
    state[start[0]][start[1]] = 1

    path = [[(start[0], start[1]), 0]]
    paths = []
    # ← ↑ → ↓
    delta = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    while path:
        (x, y), c = path.pop()
        if x == end[0] and y == end[1]:
            reverse_pt = paths[::-1]
            min_pt = [reverse_pt[0], reverse_pt[1]]
            for pt in reverse_pt[2:]:
                if pt[1] == min_pt[-1][1]:
                    # 剔除多余的噪声点
                    continue
                # 只有满足步伐差为1，并且可通过坐标差为1，才为合法点
                if pt[1] + 1 == min_pt[-1][1] and abs(sum(pt[0]) - sum(min_pt[-1][0])) == 1:
                    min_pt.append(pt)
            return c, [start] + list(map(list, zip(*min_pt[::-1])))[0]
        for i in delta:
            nx = x + i[0]
            ny = y + i[1]
            if 0 <= nx < m and 0 <= ny < n:
                if maze[nx][ny] == 0 and state[nx][ny] == 0:
                    path.append([(nx, ny), c + 1])
                    paths.append([(nx, ny), c + 1])
                    state[nx][ny] = 1


def otsu(hist, total):
    no_of_bins = len(hist)

    sum_total = 0
    for x in range(0, no_of_bins):
        sum_total += x * hist[x]

    weight_bg = 0.0
    sum_bg = 0.0
    inter_class_variances = []

    for threshold in range(0, no_of_bins):
        # 背景点w0
        weight_bg += hist[threshold]
        if weight_bg == 0:
            continue
        # 前景点w1
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += threshold * hist[threshold]
        # 背景平均灰度u0
        mean_bg = sum_bg / weight_bg
        # 前景平均灰度u1
        mean_fg = (sum_total - sum_bg) / weight_fg
        # w0*w1*(u0-u1)**2
        inter_class_variances.append(weight_bg * weight_fg * (mean_bg - mean_fg) ** 2)

    return np.argmax(inter_class_variances)


def countRegion(img):
    high, width = img.shape
    mask = np.zeros((high, width))
    mark = 0
    region_map = {}
    for i in range(high):
        for j in range(width):
            if i == 0 and j == 0:
                if img[i][j] == 255:
                    mark = mark + 1
                    mask[i][j] = mark
                    region_map[mark] = mark
            if i == 0 and j != 0:
                if img[i][j] == 255:
                    left = mask[i][j - 1]
                    if left != 0:
                        mask[i][j] = left
                    else:
                        mark = mark + 1
                        mask[i][j] = mark
                        region_map[mark] = mark
            if j == 0 and i != 0:
                if img[i][j] == 255:
                    up = mask[i - 1][j]
                    up_right = mask[i - 1][j + 1]
                    if up == 0 and up_right == 0:
                        mark = mark + 1
                        mask[i][j] = mark
                        region_map[mark] = mark
                    if up == 0 and up_right != 0:
                        mask[i][j] = up_right
                    if up_right == 0 and up != 0:
                        mask[i][j] = up
                    if up != 0 and up_right != 0:
                        if up == up_right:
                            mask[i][j] = up
                        else:
                            field_min = np.min([up, up_right])
                            mask[i][j] = field_min
                            if up < up_right:
                                region_map[up_right] = up
                            else:
                                region_map[up] = up_right
            if i != 0 and j != 0:
                if img[i][j] == 255:
                    up = mask[i - 1][j]
                    up_left = mask[i - 1][j - 1]
                    left = mask[i][j - 1]
                    up_right = 0
                    if j + 1 < width:
                        up_right = mask[i - 1][j + 1]
                    field_max = np.max([up, up_left, up_right, left])
                    if field_max == 0:
                        mark = mark + 1
                        mask[i][j] = mark
                        region_map[mark] = mark
                    else:
                        if up == up_right and up_right == up_left and up == left:
                            mask[i][j] = up
                        else:
                            field_min = np.min([up, up_left, up_right, left])
                            if field_min != 0:
                                mask[i][j] = field_min
                                if up != field_min:
                                    region_map[up] = field_min
                                if up_right != field_min:
                                    region_map[up_right] = field_min
                                if up_left != field_min:
                                    region_map[up_left] = field_min
                                if left != field_min:
                                    region_map[left] = field_min
                            else:
                                n_zero = []
                                if up != 0:
                                    n_zero.append(up)
                                if up_left != 0:
                                    n_zero.append(up_left)
                                if up_right != 0:
                                    n_zero.append(up_right)
                                if left != 0:
                                    n_zero.append(left)
                                tmp_min = np.min(n_zero)
                                mask[i][j] = tmp_min
                                for it in n_zero:
                                    if it != tmp_min:
                                        region_map[it] = tmp_min
    step_map = {}
    step = 0
    # 合并region时，需要参考step
    for k, v in region_map.items():
        if k != v:
            step += 1
            continue
        step_map[k] = step
    for i in range(high):
        for j in range(width):
            key = mask[i][j]
            if key != 0:
                if int(region_map[key]) != int(key):
                    key = region_map[key]
                if key in step_map.keys():
                    mask[i][j] = key - step_map[key]
                else:
                    mask[i][j] = key

    return mask


class MyTestCase(unittest.TestCase):
    def test_mazeMinSteps(self):
        mat1 = [[1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0]]
        start = (1, 1)
        end = (3, 4)
        self.assertEqual(mazeMinSteps(mat1, start, end), (5, [(1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (3, 4)]))

        start = (0, 3)
        end = (3, 4)
        self.assertEqual((mazeMinSteps(mat1, start, end)), None)

        mat1 = [[1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        start = (0, 1)
        end = (8, 9)
        self.assertEqual(mazeMinSteps(mat1, start, end),
                         (16, [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                               (2, 5), (3, 5), (3, 6), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (7, 9), (8, 9)]))
        start = (0, 1)
        end = (3, 0)
        self.assertEqual(mazeMinSteps(mat1, start, end), (4, [(0, 1), (1, 1), (1, 2), (2, 0), (3, 0)]))

    def test_otsu(self):
        img = cv.imread('test.jpg')
        self.assertEqual(len(img.shape), 3)
        rows, cols, _ = img.shape
        # 图像灰度化:权重法
        gray_img = img.dot([0.299, 0.587, 0.144])
        self.assertEqual((rows, cols), (535, 800))

        # 取直方图
        hist = np.histogram(gray_img, 256)[0]
        self.assertEqual(len(hist), 256)
        self.assertEqual(sum(hist), rows * cols)

        thresh = otsu(hist, rows * cols)
        print("The thresh is: " + str(thresh))

    def test_countRegion(self):
        img = [[255, 255, 255, 0, 0, 255, 255, 0, 255, 255],
               [255, 255, 0, 0, 255, 255, 0, 0, 0, 255],
               [255, 255, 0, 0, 255, 255, 255, 0, 255, 255],
               [0, 0, 255, 255, 255, 0, 255, 0, 0, 255],
               [0, 0, 0, 255, 0, 0, 255, 255, 0, 0],
               [255, 0, 0, 255, 255, 0, 0, 255, 0, 255],
               [255, 0, 0, 255, 0, 0, 0, 255, 0, 0],
               [255, 255, 255, 255, 255, 0, 0, 0, 0, 0],
               [255, 0, 0, 0, 0, 255, 0, 0, 0, 255],
               [0, 0, 255, 0, 255, 0, 255, 0, 0, 0]]
        self.assertEqual(countRegion(np.asarray(img)).tolist(),
                         [[1, 1, 1, 0, 0, 1, 1, 0, 2, 2],
                          [1, 1, 0, 0, 1, 1, 0, 0, 0, 2],
                          [1, 1, 0, 0, 1, 1, 1, 0, 2, 2],
                          [0, 0, 1, 1, 1, 0, 1, 0, 0, 2],
                          [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                          [1, 0, 0, 1, 1, 0, 0, 1, 0, 3],
                          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 1, 0, 0, 0, 4],
                          [0, 0, 5, 0, 1, 0, 1, 0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
