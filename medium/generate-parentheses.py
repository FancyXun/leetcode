'''
数字 n代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。



示例 1：

输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
示例 2：

输入：n = 1
输出：["()"]


提示：

1 <= n <= 8

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/generate-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''

n = 3


def generate(A):
    if len(A) == 2 * n:
        print(A)
    else:
        A.append('(')
        generate(A)
        A.pop()
        A.append(')')
        generate(A)
        A.pop()


print(generate([]))
