class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


order = []


def preOrderRecur(head):
    if not head:
        return

    order.append(head.val)
    preOrderRecur(head.left)
    preOrderRecur(head.right)


def inOrderRecur(head):
    if not head:
        return

    inOrderRecur(head.left)
    order.append(head.val)
    inOrderRecur(head.right)


def postOrderRecur(head):
    if not head:
        return

    postOrderRecur(head.left)
    postOrderRecur(head.right)
    order.append(head.val)


node = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))

print("前序")
preOrderRecur(node)
print(order)
order.clear()
print("中序")
inOrderRecur(node)
print(order)
order.clear()
print("后序")
postOrderRecur(node)
print(order)
order.clear()
