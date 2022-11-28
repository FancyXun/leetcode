import unittest

'''
给定一个列表，列表里的数字都是正整数，给定一个数D，找出列表中任意两个数字的差为D，返回下标
'''


def twoDiff(nums, target):
    i = -1
    j = 0
    for j in range(1, len(nums)):
        head = nums[:j]
        if (target + nums[j]) in head:
            i = head.index(target + nums[j])
            break
        elif (-target + nums[j]) in head:
            i = head.index(-target + nums[j])
            break
    if i >= 0:
        return [i, j]


def allTwoDiff(nums, target):
    """
    使用map 降低寻址复杂度
    """
    head = {}
    res = []
    for idx, num in enumerate(nums):
        if (target + num) in head:
            for j in head[target + num]:
                res.append((j, idx))
        if (-target + num) in head:
            for j in head[-target + num]:
                res.append((j, idx))
        # 考虑到数字的重复，所以需要一个List来存取所有的相同value的index
        if nums[idx] in head:
            head[nums[idx]].append(idx)
        else:
            head[nums[idx]] = [idx]
    return res


class StoneTestCase(unittest.TestCase):
    def test_1(self):
        test_nums = [1, 4, 1, 7, 9, 10, 2]
        d = 3
        res = twoDiff(test_nums, d)
        self.assertEqual(res, [0, 1])

    def test_2(self):
        test_nums = [4, 1, 6, 7, 9, 10, 2]
        d = 3
        res = twoDiff(test_nums, d)
        self.assertEqual(res, [0, 1])

    def test_3(self):
        test_nums = []
        d = 3
        res = twoDiff(test_nums, d)
        self.assertEqual(res, None)

    def test_4(self):
        test_nums = [1, 3, 5]
        d = 3
        res = twoDiff(test_nums, d)
        self.assertEqual(res, None)

    def test_5(self):
        test_nums = [1, 3, 5, 8]
        d = 3
        res = twoDiff(test_nums, d)
        self.assertEqual(res, [2, 3])

    def test_6(self):
        test_nums = [1, 4, 7, 9, 12, 15, 12, 9, 6, 3]
        d = 3
        res = allTwoDiff(test_nums, d)
        print(res)
        self.assertEqual(res, [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (3, 6), (4, 7), (6, 7), (3, 8), (7, 8), (8, 9)])

    def test_7(self):
        test_nums = []
        d = 3
        res = allTwoDiff(test_nums, d)
        self.assertEqual(res, [])

    def test_8(self):
        test_nums = [9, 6, 3, 6]
        d = 3
        res = allTwoDiff(test_nums, d)
        print(res)
        self.assertEqual(res, [(0, 1), (1, 2), (0, 3), (2, 3)])


if __name__ == '__main__':
    unittest.main()
