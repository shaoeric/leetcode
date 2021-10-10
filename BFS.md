#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

![image-20211010145944194](figs/image-20211010145944194.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # BFS
        if root is None: return 0
        queue = [root]
        depth = 1

        while queue:
            # 向四周扩散            
            for _ in range(len(queue)):
                node = queue.pop(0)

                # 判断是否到达终点
                if not node.left and not node.right:
                    return depth
                
                # 将node周围节点加入队列
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            # 扩散完一次，说明深入了一层，depth应加1
            depth += 1
        return depth
```

#### [752. 打开转盘锁:star::star:](https://leetcode-cn.com/problems/open-the-lock/)

![image-20211010153532858](figs/image-20211010153532858.png)

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        # 锁向上转
        def lock_up(s, i):
            chars = list(s)
            chars[i] = "0" if chars[i] == "9" else str(int(chars[i]) + 1)
            return ''.join(chars)
        # 锁向下转
        def lock_down(s, i):
            chars = list(s)
            chars[i] = "9" if chars[i] == "0" else str(int(chars[i]) - 1)
            return ''.join(chars)

        queue = ['0000']
        deads = set(deadends)
        visited = set()
        visited.add('0000')
        step = 0

        while queue:
            # 向节点的8个相邻节点遍历
            for _ in range(len(queue)):
                node = queue.pop(0)
                # 判断是否合法，是否达到终点
                if node in deads: continue
                if node == target: return step

                for i in range(4):
                    up = lock_up(node, i)
                    # 避免走回头路
                    if up not in visited:
                        queue.append(up)
                        visited.add(up)
                    down = lock_down(node, i)
                    # 避免走回头路
                    if down not in visited:
                        queue.append(down)
                        visited.add(down)
            step += 1
        return -1
```

