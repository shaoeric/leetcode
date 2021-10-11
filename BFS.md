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
            # 遍历当前层的所有节点
            for _ in range(len(queue)):
                node = queue.pop(0)
                # 处理当前节点，并向四周扩散
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

#### [773. 滑动谜题:star::star:](https://leetcode-cn.com/problems/sliding-puzzle/)

![image-20211010205353892](figs/image-20211010205353892.png)

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        def swap(node: str, i: int, j: int):
            tmp = list(node)
            tmp[i], tmp[j] = tmp[j], tmp[i]
            return ''.join(tmp)
        
        # 2x3的数组转换成字符串
        s = ""
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                s += str(board[i][j])
        
        # 设定好每个位置所对应的邻居索引
        # [0 1 2]
        # [3 4 5]
        neighbor = [(1, 3), (0, 2, 4), (1, 5),
                    (0, 4), (1, 3, 5), (2, 4)]
        
        target = "123450"
        q = [s]
        step = 0
        visited = set()
        visited.add(s)
        while q:
            # 向四周扩散
            for _ in range(len(q)):
                node = q.pop(0)

                if node == target:
                    return step
                
                # 查找0的位置
                pos = node.index('0')
                # 和邻居交换
                for neighbor_idx in neighbor[pos]:
                    tmp = swap(node, pos, neighbor_idx)
                    if tmp not in visited:
                        visited.add(tmp)
                        q.append(tmp)
            step += 1
        return -1
```

#### [429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

![image-20211011101733041](figs/image-20211011101733041.png)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if root is None: return []
        queue = [root]
        res = []
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                # 处理当前节点
                if node:
                    tmp.append(node.val)
                
                    for child in node.children:
                        queue.append(child)
            res.append(tmp)
        return res
```

#### [127. 单词接龙:star::star::star:](https://leetcode-cn.com/problems/word-ladder/)

![image-20211011144313674](figs/image-20211011144313674.png)

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        def diff(s: str, t: str):
            res = 0
            for i in range(len(s)):
                if s[i] != t[i]:
                    res += 1
            return res
        
        def get_available_nodes(begin: str, words: List[str], visited: set):
            availables = []
            for word in words:
                if word not in visited and diff(begin, word) == 1:
                    availables.append(word)
            return availables

        queue = [beginWord]
        visited = set()
        step = 1

        while queue:
            # 遍历当前层的所有节点
            for _ in range(len(queue)):
                # 处理当前节点，如果已经是末尾了，就返回
                begin_node = queue.pop(0)
                if begin_node == endWord:
                    return step
				# 获取当前节点的所有可以访问的子节点
                nodes = get_available_nodes(begin_node, wordList, visited)
                for node in nodes:
                    queue.append(node)
                    visited.add(node)
            step += 1
        return 0
```

