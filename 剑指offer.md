#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

![image-20211111143156998](figs/image-20211111143156998.png)

```python
class CQueue:

    def __init__(self):
        self.a = []
        self.b = []

    def appendTail(self, value: int) -> None:
        self.a.append(value)

    def deleteHead(self) -> int:
        if len(self.a) == 0: return -1
        while self.a:
            self.b.append(self.a.pop())
        res = self.b.pop()
        while self.b:
            self.a.append(self.b.pop())
        return res
# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```

#### [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

![image-20211111144608512](figs/image-20211111144608512.png)

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.a = []
        self.b = []

    def push(self, x: int) -> None:
        self.a.append(x)
        if not self.b or self.b[-1] >= x:
            self.b.append(x)

    def pop(self) -> None:
        if self.a.pop() == self.b[-1]:
            self.b.pop()

    def top(self) -> int:
        return self.a[-1]

    def min(self) -> int:
        return self.b[-1]
```

#### [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

![image-20211111192208129](figs/image-20211111192208129.png)

```python
class Solution:
    def fib(self, n: int) -> int:
        dp = {0: 0, 1: 1}
        def helper(n):
            if n <= 1: return n
            if n in dp: return dp[n]

            a = helper(n-1) % 1000000007
            b = helper(n-2) % 1000000007
            res = (a + b)  %  1000000007
            dp[n-1] = a
            dp[n-2] = b
            dp[n] = res
            return res

        return helper(n)
```

```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(n-1):
            a, b = b, a + b
        return b % 1000000007
```

#### [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

![image-20211111193254632](figs/image-20211111193254632.png)

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        dic = set()
        for i in nums:
            if i not in dic:
                dic.add(i)
            else:
                return i
```

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return nums[i]
```

#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

![image-20211112141130177](figs/image-20211112141130177.png)

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        p = head
        while p:
            res.append(p.val)
            p = p.next
        return res[::-1]
```

```python
# 递归
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if not head:
            return []
        
        res = self.reversePrint(head.next)
        res.append(head.val)
        return res
```

#### [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

![image-20211112141508431](figs/image-20211112141508431.png)

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        cur = head
        while head:
            cur = head
            head = head.next
            cur.next = prev
            prev = cur
        return prev
```

```python
# 递归
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        if not head.next:
            return head

        right = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return right
```

#### [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

![image-20211112144244606](figs/image-20211112144244606.png)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
# 二叉树递归
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        def dfs(head):
            if not head: return None
            if head in visited:
                return visited[head]
			# 当前节点
            cur = Node(head.val)
            visited[head] = cur
            # next
            cur.next = dfs(head.next)
            # randoom
            cur.random = dfs(head.random)
            return cur
        visited = {}
        return dfs(head)
```

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None
        q = [head]

        copy = Node(head.val)
        visited = {head: copy}
        while q:
            cur = q.pop(0)
            if cur.next and cur.next not in visited:
                visited[cur.next] = Node(cur.next.val)
                q.append(cur.next)
            if cur.random and cur.random not in visited:
                visited[cur.random] = Node(cur.random.val)
                q.append(cur.random)
            visited[cur].next = visited.get(cur.next)
            visited[cur].random = visited.get(cur.random)
        return copy
```

#### [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

![image-20211112151120921](figs/image-20211112151120921.png)

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        r, c = len(matrix), len(matrix[0])

        i, j = r-1, 0
        while i >= 0 and j < c:
            if matrix[i][j] == target:
                 return True
            elif matrix[i][j] < target:
                j += 1
            else:
                i -= 1
        return False
```

#### [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

![image-20211112152224234](figs/image-20211112152224234.png)

```python
class Solution:
    def numWays(self, n: int) -> int:
        dic = {}
        def helper(n):
            if n == 0 or n == 1: return 1
            elif n == 2:
                return n
            if n in dic:
                return dic[n]
            
            res = (helper(n-1) % 1000000007 + helper(n-2) % 1000000007) % 1000000007
            dic[n] = res 
            return res
        return helper(n)
```

#### [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

![image-20211112155636595](figs/image-20211112155636595.png)

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        if numbers[0] < numbers[-1]:
            return numbers[0]
        left, right = 0, len(numbers)-1
        while left < right:
            mid = left + (right - left) // 2
            
            # 一定要和right位进行比较，不能与left位比较，left~mid之间的情况较多，并不是二段的。
            if numbers[mid] > numbers[right]:
                left = mid + 1
            elif numbers[mid] < numbers[right]:
                right = mid
            else:
                right -= 1

        return numbers[left]
```

#### [剑指 Offer 12. 矩阵中的路径:star::star:](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

![image-20211112162058265](figs/image-20211112162058265.png)

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        
        self.res = False

        def dfs(i, j, start):
            if len(visited) == len(word):
                self.res = True
                return
            
            for x, y in [(i-1, j), (i + 1, j), (i, j-1), (i, j+1)]:
                if x < 0 or x >= m or y < 0 or y >= n:
                    continue
                if (x, y) not in visited and board[x][y] == word[start]:
                    visited.add((x, y))
                    dfs(x, y, start + 1)
                    if self.res: 
                        return True
                    visited.remove((x, y))
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    visited = {(i, j)}
                    dfs(i, j, 1)
                    if self.res: 
                        return True
           
        return False 
```

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        #矩阵中的路径暴力法，找到路径起始，然后枚举，DFS
        #首先找出来第一个元素，之后从第一个元素起始，进行DFS
        #如何保证不会被重复访问,将已经访问过的节点变为#，保证以后再回到该节点时不会被访问！
        #怎么回退！！
        def DFS(a,b,k):
            if a < 0 or a > n-1 or b < 0 or b > m-1:
                return False
            if board[a][b] != word[k]:
                return False   
            if k == len(word)-1:
                return True
            
            board[a][b] = '#'
            res = DFS(a , b+1 , k+1) or DFS(a , b-1 , k+1) or DFS(a+1 , b , k+1) or DFS(a-1 , b , k+1)
            board[a][b] = word[k]#回退
            return res

        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    if DFS(i,j,0):
                        return True
        return False         
```

#### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

![image-20211113134402019](figs/image-20211113134402019.png)

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ""
        for i in s:
            if i != ' ':
                res += i
            else:
                res += '%20'
        return res
```

#### [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

![image-20211113134744892](figs/image-20211113134744892.png)

#### [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

![image-20211113141240393](figs/image-20211113141240393.png)

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def position_sum(i):
            res = 0
            s = i
            while s > 0:
                res += s % 10
                s = s // 10
            return res
        
        self.res = 0
        
        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return
            if position_sum(i) + position_sum(j) > k:
                return
            if (i, j) in visited:
                return
			# 访问过的就不用再回头看了，所以不用回溯撤销选择
            visited.add((i, j))
            self.res += 1
            for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                dfs(x, y)

        
        visited = set()
        dfs(0, 0)
        return self.res
```

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def position_sum(i):
            res = 0
            s = i
            while s > 0:
                res += s % 10
                s = s // 10
            return res
        
        res = 0
        q = [(0, 0)]
        visited = set()
        # 不要求扩散数，所以while循环中不需要for。弹出节点后，先判断节点有效性，有效则访问
        while q:
            i, j = q.pop(0)
            if i < 0 or i >= m or j < 0 or j >= n:
                continue
            if (i, j) in visited:
                continue
            if position_sum(i) + position_sum(j) > k:
                continue
            
            visited.add((i, j))
            res += 1
            for x, y in [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]:
                q.append((x, y))
                
        return res
```

#### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

![image-20211113144705337](figs/image-20211113144705337.png)

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        
        head_val = preorder[0]
        head = TreeNode(head_val)
        
        index = inorder.index(head_val)
        head.left = self.buildTree(preorder[1: index+1], inorder[:index])
        head.right = self.buildTree(preorder[index+1:], inorder[index+1:])
        return head
```

#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

![image-20211114134140084](figs/image-20211114134140084.png)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0: return 0
        left, right = 0, len(nums) -1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                right = mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        begin = None

        if nums[left] == target:
            begin = left
        else:
            return 0
        
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left + 1) // 2
            if nums[mid] == target:
                left = mid
            elif nums[mid] < target:
                left = mid
            else:
                right = mid - 1
        end = left
        return end - begin + 1
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0: return 0

        left, right = 0, n - 1
        while left + 1 < right:
            mid = left + (right - left) //2
            if nums[mid] == target:
                right = mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if nums[left] == target:
            begin = left
        elif nums[right] == target:
            begin = right
        else:
            return 0
        
        left, right = 0, n - 1
        while left + 1 < right:
            mid = left + (right - left) //2
            if nums[mid] == target:
                left = mid
            elif nums[mid] < target:
                left = mid
            else:
                right = mid -1

        if nums[right] == target:
            end = right
        else:
            end = left
        
        return end - begin + 1
```

#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

![image-20211114135346008](figs/image-20211114135346008.png)

```python
# 二分法
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == mid:
                left = mid + 1
            elif nums[mid] > mid:
                right = mid - 1
        return left
```

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums) + 1
        s = (0 + n) * (n - 1) // 2
        return s - sum(nums)
```

#### [剑指 Offer 14- I. 剪绳子:star:](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

![image-20211114141754700](figs/image-20211114141754700.png)

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0] * (n + 1)
        
        # dp[0] = dp[1] = 0
        dp[2] = 1

        # 从3开始，一直求到dp[n]
        for i in range(3, n + 1):
            # 每次剪j长度的绳子,j:[2,i-1], j可以取1但是没必要，因为剪掉1剩下的是i-1，计算有重复。
            for j in range(2, i):
                # 减去第一段长度为j的绳子后，可以继续剪(dp[i-j]) 也可以不剪 (i-j)
                dp[i] = max(dp[i], j * dp[i-j], j * (i - j))
        return dp[-1]
```

#### [剑指 Offer 14- II. 剪绳子 II:star:](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

![image-20211114142753667](figs/image-20211114142753667.png)

```python
# 由于n太大了，用动态规划速度太慢，所以贪心算法效率高。剪绳子I也可以用贪心
# https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/solution/mian-shi-ti-14-ii-jian-sheng-zi-iitan-xin-er-fen-f/
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n < 4:
            return n - 1
        res = 1
        while n > 4:
            res = res * 3 % 1000000007
            n -= 3
        return res * n % 1000000007
```

#### [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

![image-20211114143511474](figs/image-20211114143511474.png)

```python
# 递归
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        def helper(l1, l2):
            if not l1 and not l2:
                return None
            if not l1:
                return l2
            if not l2:
                return l1
            
            dummy = ListNode(-1)
            if l1.val < l2.val:
                dummy.next = l1
                dummy.next.next = helper(l1.next, l2)
            else:
                dummy.next = l2
                dummy.next.next = helper(l1, l2.next)
            return dummy.next

        return helper(l1, l2)
```

```python
# 迭代
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(-1)
        p = dummy
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        if l1:
            p.next = l1
        if l2:
            p.next = l2
        return dummy.next
```

#### [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

![image-20211115135515322](figs/image-20211115135515322.png)

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        position = {}
        for i, c in enumerate(s):
            if c in position:
                position[c] = -1
            else:
                position[c] = i
        
        first = len(s)
        for pos in position.values():
            if pos != -1 and pos < first:
                first = pos

        return ' ' if first == len(s) else s[first]
```

#### [剑指 Offer 26. 树的子结构:star::star::star:](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

![image-20211115143351961](figs/image-20211115143351961.png)

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
		# 先序遍历A的节点
        # 判断树A中是否包含子树B
        def check(a, b):
            # b越过了叶子节点，说明叶子节点匹配成功，返回True
            if not b:
                return True
            # a越过叶子节点，说明叶子节点没有匹配成功，返回False
            if not a:
                return False
            return a.val == b.val and check(a.left, b.left) and check(a.right, b.right)

        if not A or not B:
            return False
        return check(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
```

#### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

![image-20211115144849235](figs/image-20211115144849235.png)

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        left = self.mirrorTree(root.left)
        right = self.mirrorTree(root.right)
        root.left = right
        root.right = left
        return root
```

#### [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

![image-20211116125232276](figs/image-20211116125232276.png)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root: return []
        q = [root]
        res = []
        while q:
            node = q.pop(0)
            res.append(node.val)

            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        return res
```

#### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

![image-20211116125530698](figs/image-20211116125530698.png)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        q = [root]
        res = []
        while q:
            tmp = []
            for _ in range(len(q)):
                node = q.pop(0)
                tmp.append(node.val)

                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(tmp)
        return res
```

#### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

![image-20211116131237016](figs/image-20211116131237016.png)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        q = [root]
        res = []

        while q:
            tmp = []
  
            for _ in range(len(q)):
                node = q.pop(0)
                tmp.append(node.val)
             
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            if len(res) % 2 == 0:
                res.append(tmp)
            else:
                res.append(tmp[::-1])
        return res
```

#### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

![image-20211116132952455](figs/image-20211116132952455.png)

```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums)-1
        while left < right:
            while left < right and nums[left] % 2 == 1:
                left += 1
            while left < right and nums[right] % 2 == 0:
                right -= 1
            nums[left], nums[right] = nums[right], nums[left]
        return nums
```

#### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

![image-20211116133634879](figs/image-20211116133634879.png)

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        if n == 0: return 0
        
        res = 0
        while n > 0:
            if n % 2 == 1:
                res += 1
            n = n // 2
        return res if res > 0 else 1
```

#### [剑指 Offer 29. 顺时针打印矩阵:star::star:](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

![image-20211116135149115](figs/image-20211116135149115.png)

```python
# https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/solution/shan-chu-di-yi-xing-ni-shi-zhen-xuan-zhuan-python5/
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            # 二维数组转置list(zip(*matrix))
            matrix = list(zip(*matrix))[::-1]
        return res
```

#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

![image-20211116135949076](figs/image-20211116135949076.png)

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        slow, fast = dummy, dummy
        for _ in range(k):
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

#### [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

![image-20211117140354475](figs/image-20211117140354475.png)

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        max_val = 10 ** n
        return [i for i in range(1, max_val)]
```

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        res=[]
        temp=['0']*n
        def helper(index):
            if index==n:
                res.append(int(''.join(temp)))
                return
            for i in range(10):
                temp[index]=chr(ord("0")+i)
                helper(index+1)
        helper(0)
        return res[1:]
```

#### [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

![image-20211118130139383](figs/image-20211118130139383.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n <= 1: return 0
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = -prices[0]
		# 第0列为持有，第一列为卖出
        # 持有要花钱买，所以收益为负
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][1]
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        inf = int(1e9)
        minprice = inf
        maxprofit = 0
        # 记录历史最低点，每天都考虑如果是在历史最低点买入的，今天卖出赚多少钱
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)
            minprice = min(price, minprice)
        return maxprofit
```

#### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

![image-20211118132257923](figs/image-20211118132257923.png)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if n == 0:
                return 1
            if n == 1:
                return x
            
            tmp = helper(x, n // 2)
            if n % 2 == 0:
                return tmp * tmp
            else:
                return tmp * tmp * x
        
        pos = (n > 0)
        res = helper(x, abs(n))
        return res if pos else 1 / res
```

#### [10. 正则表达式匹配:star::star::star::star:](https://leetcode-cn.com/problems/regular-expression-matching/)

![image-20211118134433348](figs/image-20211118134433348.png)

```python
# https://leetcode-cn.com/problems/regular-expression-matching/solution/hen-rong-yi-li-jie-de-zheng-ze-biao-da-s-cpgp/
class Solution:
    def isMatch(self, s: str, p: str):
        if not p: return not s
        if not s and len(p) == 1: return False

        m = len(s) + 1
        n = len(p) + 1

        dp = [[False for _ in range(n)] for _ in range(m)]

        dp[0][0] = True

        # 确定dp数组的第一行，如果遇到了*,只要判断其对应的前面两个元素的dp值
        # 注意：我们无需判断p里面的第一个值是否为"*"，如果为"*",那肯定匹配不到为Fasle,原数组正好是Fasle，所以直接从2开始判断即可
        for j in range(2, n):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j - 2]

        for r in range(1, m):
            i = r - 1  # 对应s中的元素
            for c in range(1, n):
                j = c - 1  # 对应p中的元素
                if s[i] == p[j] or p[j] == '.':
                    dp[r][c] = dp[r - 1][c - 1]
                elif p[j] == '*':
                    if p[j - 1] == s[i] or p[j - 1] == '.':
                        dp[r][c] = dp[r - 1][c] or dp[r][c - 2]
                    else:
                        dp[r][c] = dp[r][c - 2]
                else:
                    dp[r][c] = False

        return dp[m - 1][n - 1]
```

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        # 初始化首行
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        # 状态转移
        for i in range(1, m):
            for j in range(1, n):
                if p[j - 1] == '*':
                    if dp[i][j - 2]: dp[i][j] = True     # p字符为*时，可以把j-2字符重复0次，只看dp[i][j-2]的匹配结果
                    # 如果p字符为*，那j-1字符与s字符相等或者j-1字符为.时，两个条件等价，都相当于p和s可以匹配
                    elif dp[i - 1][j] and s[i - 1] == p[j - 2]: dp[i][j] = True  # 2.
                    elif dp[i - 1][j] and p[j - 2] == '.': dp[i][j] = True       # 3.
                else:
                    if dp[i - 1][j - 1] and s[i - 1] == p[j - 1]: dp[i][j] = True# 1.
                    elif dp[i - 1][j - 1] and p[j - 1] == '.': dp[i][j] = True   # 2.
        return dp[-1][-1]
```



#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

![image-20211121133325202](figs/image-20211121133325202.png)

```python
# 两个状态，选当前的数 和 不选当前的数
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * 2 for _ in range(n)]
        # 初始化需要考虑，如果数组只有一个数组的时候，选不选第一个数最大和都是这个。
        dp[0][0] = dp[0][1] = nums[0]

        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0] + nums[i], nums[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0])
        
        return max(dp[-1])
```

```python
# 状态压缩
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        res = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
            res = max(res, dp[i])
            
        return res
```

#### [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

![image-20211121135635603](figs/image-20211121135635603.png)

```python
# 二维空间
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, n):
            dp[0][i] = dp[0][i-1] + grid[0][i]
        for j in range(1, m):
            dp[j][0] = dp[j-1][0] + grid[j][0]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

```python
# 一维空间
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        dp[0] = grid[0][0]
        for i in range(1, n):
            dp[i] = dp[i-1] + grid[0][i]
        for i in range(1, m):
            for j in range(n):
                if j == 0:
                    dp[j] += grid[i][j]
                else:
                    dp[j] = max(dp[j-1], dp[j]) + grid[i][j]
        return dp[-1]
```

#### [剑指 Offer 46. 把数字翻译成字符串:star:](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

![image-20211122142144703](figs/image-20211122142144703.png)

![image-20211122142332830](figs/image-20211122142332830.png)

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        n = len(s)
        if n < 2:
            return 1
        dp = [0] * n
        dp[0] = 1
        dp[1] = 2 if int(s[0] + s[1]) < 26 else 1
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2] if (int(s[i-1] + s[i]) < 26 and s[i-1] != '0') else dp[i-1]
        return dp[-1]
```

#### [剑指 Offer 48. 最长不含重复字符的子字符串:star::star:](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

![image-20211122145113416](figs/image-20211122145113416.png)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, right = 0, 0
        res = 0
        window = {}

        while right < len(s):
            if s[right] not in window:
                window[s[right]] = 1
            else:
                window[s[right]] += 1

            while window[s[right]] > 1:
                window[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res
```

```python
# 动态规划
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0: return 0
        dic = {}
        
        #dp[j] 代表以字符 s[j]为结尾的 “最长不重复子字符串” 的长度。
        dp = [0] * n
        dp[0] = 1
        dic[s[0]] = 0
        res = 1
        
        for i in range(1, n):
            # 如果不在字典中
            # 如果在字典中，但是上一次出现的位置到现在的位置的距离 超出了要比较的位置，不影响结果
            if s[i] not in dic or (i - dic[s[i]]) > dp[i-1]:
                dp[i] = dp[i-1] + 1
            
            else:
                dp[i] = i - dic[s[i]]
            # 更新位置
            dic[s[i]] = i
            res = max(res, dp[i]) 
        return res
```

#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

![image-20211123124329539](figs/image-20211123124329539.png)

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        p = dummy
        while p.next.val != val:
            p = p.next
        
        p.next = p.next.next
        return dummy.next
```

```python
# 递归
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        def helper(head):
            if not head:
                return head
            if head.val == val:
                return head.next
            
            head.next = helper(head.next)
            return head
        return helper(head)
```

#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

![image-20211123125127875](figs/image-20211123125127875.png)

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        fast, slow = dummy, dummy
        for _ in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

#### [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

![image-20211124194335933](figs/image-20211124194335933.png)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a, b = headA, headB
        while a != b:
            a = a.next if a is not None else headB
            b = b.next if b is not None else headA
        return a
```

#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

![image-20211125143516447](figs/image-20211125143516447.png)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums) 
        i, j = 0, n-1
        while i < j:
            s = nums[i] + nums[j]
            if s == target:
                return [nums[i], nums[j]]
            elif s > target:
                j -= 1
            else:
                i += 1
```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        n = len(nums) 
        
        for i in range(n):
            if nums[i] not in dic:
                dic[target - nums[i]] = i
            else:
                return [nums[dic[nums[i]]], nums[i]]
```

#### [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

![image-20211125145019321](figs/image-20211125145019321.png)

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        stack = []
        tmp = ''
        for i in range(len(s)):
            if tmp == '' and s[i] == ' ':
                continue
            elif s[i] != ' ':
                tmp += s[i]
            elif tmp != '' and s[i] == ' ':
                stack.append(tmp)
                tmp = ''
        if tmp != '':
            stack.append(tmp)

        return ' '.join(stack[::-1])
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        split = s.split(' ')
        split = [c for c in split if c != '']
        return ' '.join(split[::-1])
```

#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

![image-20211126204104397](figs/image-20211126204104397.png)

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        path = []
        res = []

        def backtrack(path, root, targetSum):
            if not root.left and not root.right and targetSum == 0:
                res.append(path[:])
                return
            
            if not root.left and not root.right:
                return

            # 选择左
            if root.left:
                backtrack(path + [root.left.val], root.left, targetSum - root.left.val)
            
            # 选择右
            if root.right:
                backtrack(path + [root.right.val], root.right, targetSum - root.right.val)
            
        if root is None: return []
        backtrack(path + [root.val], root, targetSum - root.val)
        return res
```

#### [剑指 Offer 36. 二叉搜索树与双向链表:star::star::star::star:](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

![image-20211126211440397](figs/image-20211126211440397.png)

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def helper(root):
            if not root:
                return None
            
            helper(root.left)

            if self.pre:
                self.pre.right, root.left = root, self.pre
            else:
                self.head = root
            self.pre = root

            helper(root.right)
            
        self.pre = None
        self.head = None
        helper(root)
        if not self.pre: return None
        self.pre.right, self.head.left = self.head, self.pre
        return self.head
```

#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

![image-20211126212802947](figs/image-20211126212802947.png)

```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        
        def helper(root):
            if not root:
                return None
            
            if helper(root.right):
                return self.pre
            
            self.pre = root
            self.k += 1
            
            if self.k == k:
                return self.pre

            if helper(root.left):
                return self.pre

        self.pre = None
        self.k = 0
        helper(root)
        return self.pre.val
```

```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                if node.left:
                    stack.append(node.left)
                
                stack.append(node)
                stack.append(None)

                if node.right:
                    stack.append(node.right)
            else:
                node = stack.pop()
                k -= 1
                if k == 0:
                    return node.val
```

#### [剑指 Offer 45. 把数组排成最小的数:star::star::star:](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

![image-20211127093019358](figs/image-20211127093019358.png)

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        # 内置排序
        def sort_rule(x, y):
            a, b = x + y, y + x
            if a < b: return -1
            elif a > b: return 1
            else: return 0

        strs = [str(i) for i in nums]
        strs.sort(key=functools.cmp_to_key(sort_rule))
        return ''.join(strs)
```

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        # 快速排序
        def quick_sort(l, r):
            if l >= r: return
            i, j = l, r
            # l为pivot
            while i < j:
                while i < j and strs[j] + strs[l] >= strs[l] + strs[j]: j -= 1
                while i < j and strs[i] + strs[l] <= strs[l] + strs[i]: i += 1
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[l] = strs[l], strs[i]
            quick_sort(l, i-1)
            quick_sort(i+1, r)
        
        strs = [str(i) for i in nums]
        quick_sort(0, len(strs)-1)
        return ''.join(strs)
```

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        # 快速排序
        def quick_sort(l, r):
            if l >= r: return
            i, j = l, r
            # r为pivot
            while i < j:
                while i < j and strs[i] + strs[r] <= strs[r] + strs[i]: i += 1
                while i < j and strs[j] + strs[r] >= strs[r] + strs[j]: j -= 1
                
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[r] = strs[r], strs[i]
            quick_sort(l, i-1)
            quick_sort(i+1, r)
        
        strs = [str(i) for i in nums]
        quick_sort(0, len(strs)-1)
        return ''.join(strs)
```

#### [剑指 Offer 61. 扑克牌中的顺子:star::star:](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

![image-20211127103020814](figs/image-20211127103020814.png)

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        repeat = set()
        mx = -1
        mn = 14
        for num in nums:
            if num in repeat:
                return False
            elif num == 0:
                continue
            repeat.add(num)
            mx = max(mx, num)
            mn = min(mn, num)
        return True if mx - mn < 5 else False
```

#### [剑指 Offer 40. 最小的k个数:star::star::star::star:](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

![image-20211128093230162](figs/image-20211128093230162.png)

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        heapq.heapify(arr)
        res = []
        for _ in range(k):
            res.append(heapq.heappop(arr))
        return res
```

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        def partition(l, r):
            if l >= r: return
            i, j = l, r
            index = random.randint(l, r)
            pivot = arr[index]
            arr[index], arr[i] = arr[i], arr[index]
            while i < j:
                while i < j and arr[j] >= pivot:
                    j -= 1
                arr[i] = arr[j]
                while i < j and arr[i] <= pivot:
                    i += 1
                arr[j] = arr[i]
            arr[i] = pivot            
            return i

        def topk_split(l, r, k):
            if l >= r: return 
            index = partition(l, r)
            topk_split(l, index-1, k)
            topk_split(index+1, r, k)
        
        topk_split(0, len(arr)-1, k)
        return arr[:k]
```

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        def partition(l, r):
            if l >= r: return
            i, j = l, r
            index = random.randint(l, r)
            pivot = arr[index]
            arr[index], arr[i] = arr[i], arr[index]
            while i < j:
                while i < j and arr[j] >= pivot:
                    j -= 1
                while i < j and arr[i] <= pivot:
                    i += 1
                arr[i], arr[j] = arr[j], arr[i]
            arr[i], arr[l] = arr[l], arr[i]
            return i

        def topk_split(l, r, k):
            if l >= r: return 
            index = partition(l, r)
            topk_split(l, index-1, k)
            topk_split(index+1, r, k)
        
        topk_split(0, len(arr)-1, k)
        return arr[:k]
```

#### [剑指 Offer 41. 数据流中的中位数:star::star::star:](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

![image-20211128103536883](figs/image-20211128103536883.png)

```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A = []  # 小顶堆，保存较大的一半数据
        self.B = []  # 大顶堆，保存较小的一半数据
        # 取中位数时两个堆顶 可以找到中间位置
        # heapq是小顶堆

    def addNum(self, num: int) -> None:
        # 偶数时，要给A添加。实现方法是先加到B中，然后将B的堆顶元素加到A中
        if len(self.A) == len(self.B):
            heapq.heappush(self.B, -num)
            heapq.heappush(self.A, -heapq.heappop(self.B))
        # 奇数时，反过来操作
        else:
            heapq.heappush(self.A, num)
            heapq.heappush(self.B, -heapq.heappop(self.A))
    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

![image-20211128103900615](figs/image-20211128103900615.png)

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        def helper(root):
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            return max(left, right) + 1
        return helper(root)
```

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        q = [root]
        res = 0
        while q:
            for _ in range(len(q)):
                node = q.pop(0)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res += 1
        return res
```

#### [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

![image-20211128104724873](figs/image-20211128104724873.png)

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def helper(root):
            if not root:
                return True, 0
            
            left_flag, left_depth = helper(root.left)
            right_flag, right_depth = helper(root.right)
            return left_flag and right_flag and abs(left_depth - right_depth) <= 1, max(left_depth, right_depth) + 1
        return helper(root)[0]
```

#### [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

![image-20211129091033033](figs/image-20211129091033033.png)

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def helper(root, p, q):
            if not root:
                return None
            if root == p or root == q:
                return root
            if root.val > p.val and root.val > q.val:
                return helper(root.left, p, q)
            if root.val < p.val and root.val < q.val:
                return helper(root.right, p, q)
            return root
        return helper(root, p, q)
```

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val: p, q = q, p # 保证 p.val < q.val
        while root:
            if root.val < p.val: # p,q 都在 root 的右子树中
                root = root.right # 遍历至右子节点
            elif root.val > q.val: # p,q 都在 root 的左子树中
                root = root.left # 遍历至左子节点
            else: break
        return root
```

#### [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

![image-20211129092304031](figs/image-20211129092304031.png)

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        def helper(root, p, q):
            if not root:
                return None
            if root == p or root == q:
                return root
            left = helper(root.left, p, q)
            right = helper(root.right, p, q)
            if left and right:
                return root
            if left:
                return left
            if right:
                return right
        return helper(root, p, q)
```

#### [剑指 Offer 64. 求1+2+…+n:star::star::star:](https://leetcode-cn.com/problems/qiu-12n-lcof/)

![image-20211129093158001](figs/image-20211129093158001.png)

```python
class Solution:
    def __init__(self):
        self.res = 0
    def sumNums(self, n: int) -> int:
        n > 1 and self.sumNums(n - 1)
        self.res += n
        return self.res
```

#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

![image-20211130093327043](figs/image-20211130093327043.png)

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if not postorder:
            return True
        
        # 像这样拆分order，结果是错误的。
        """
        for v in range(len(postorder)-1):
            if v < root_val:
                left_list.append(v)
            elif v > root_val:
                right_list.append(v)
        """
        # 应该按照二叉搜索树规则想
        # 根>左 and 根<右
        # 所以列表中的分布是 左 右 根
        # 找到第一个不大于根的位置，说明左子树进行完了
        root_val = postorder[-1]
        for i in range(len(postorder)):
            if postorder[i] > root_val:
                break
        left_list = postorder[:i]
        right_list = postorder[i: -1]
        for right in right_list:
            if right < root_val: return False
            
        left_flag = self.verifyPostorder(left_list)
        right_flag = self.verifyPostorder(right_list)
        return left_flag and right_flag
```



#### [剑指 Offer 65. 不用加减乘除做加法:star::star::star:](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

![image-20211201094809487](figs/image-20211201094809487.png)



![image-20211201094833783](figs/image-20211201094833783.png)

[python解法详细解读（位运算具体过程） - 不用加减乘除做加法 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/solution/pythonjie-fa-xiang-xi-jie-du-wei-yun-sua-jrk8/)

```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        # 获取数的补码，用补码运算
        a = a & x
        b = b & x
        while b != 0:
            c = a & b  # 进位
            a = a ^ b  # 本位
            b = (c << 1) & x
        # 如果结果是负的，需要将补码转换成原码表示
        return a if a <= 0x7fffffff else ~(a ^ x)
```

#### [剑指 Offer 56 - I. 数组中数字出现的次数:star::star::star::star:](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

![image-20211202100312311](figs/image-20211202100312311.png)

[🔥数组中出现1次/2次的数字——垂直方向的位运算💎 - 数组中数字出现的次数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/solution/shu-zu-zhong-chu-xian-1ci-2ci-de-shu-zi-8xrsh/)

![image-20211202100553576](figs/image-20211202100553576.png)

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        # 记录全部数字的异或
        sum = 0
        for num in nums:
            sum ^= num
        # 需要对数组划分，按照某一位特征做划分
        # 找到sum的第一位为1的位置
        mask = 1
        while (mask & sum) == 0:
            mask = mask << 1
        # 此时mask只有一个位置是1，其余位置都是0
        x = 0
        y = 0
        for num in nums:
            if (num & mask) == 0:
                x ^= num
            else:
                y ^= num
        return [x, y]
```

#### [剑指 Offer 56 - II. 数组中数字出现的次数 II:star::star::star:](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

![image-20211202101616635](figs/image-20211202101616635.png)

![image-20211202100613149](figs/image-20211202100613149.png)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        sum = 0
        for i in range(32):
            bit = 0
            for num in nums:
                bit += ((num >> i) & 1)
            sum += (bit % 3) << i
        return sum
```

#### [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

![image-20211203103637597](figs/image-20211203103637597.png)

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dic = {}
        res = None
        max = 0
        for n in nums:
            if n not in dic:
                dic[n] = 1
            else:
                dic[n] += 1
            if dic[n] > max:
                max = dic[n]
                res = n
        return res
```

```python
# 摩尔投票，结果是返回超过一半个数的数字，而不是众数。但题目给出总是存在多数元素，所以可以认为是找众数【只要众数多于总数的一半就一定能解出来  如{7, 7, 5, 5, 5, 2, 2}无法用摩尔投票解 】
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0: x = num
            if num == x:
                votes += 1
            else:
                votes -= 1
        return x
```

#### [剑指 Offer 66. 构建乘积数组:star::star::star:](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

![image-20211203120656848](figs/image-20211203120656848.png)

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        # 前缀积 后缀积
        n = len(a)
        left = [1] * (n + 1)
        right = [1] * (n + 1)
        for i in range(1, n + 1):
            left[i] = left[i-1] * a[i-1]
            right[n - i] = right[n + 1 - i] * a[n - i]
        
        b = []
        for i in range(n):
            b.append(left[i] * right[i+1])
        return b
```

#### [剑指 Offer 57 - II. 和为s的连续正数序列:star:](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

![image-20211204103827436](figs/image-20211204103827436.png)

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res = []
        left, right = 1, 2
        # 滑动窗口
        while right <= target // 2 + 1:
            s = (left + right) * (right - left + 1) // 2
            if s < target:
                right += 1
            elif s > target:
                left += 1
            else:
                res.append(list(range(left, right + 1)))
                right += 1
        return res
```

#### [剑指 Offer 62. 圆圈中最后剩下的数字:star:](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

![image-20211204111259536](figs/image-20211204111259536.png)

[(51条消息) 约瑟夫环——公式法（递推公式）_再难也要坚持-CSDN博客_约瑟夫环公式](https://blog.csdn.net/u011500062/article/details/72855826)

![image-20211204112613191](figs/image-20211204112613191.png)



$f(n, m) = [f(n - 1, m) + m] \% n$

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        index_tmp = 0
        for i in range(2,n+1):
            index_tmp = (index_tmp + m) % i
        return index_tmp
```

```python
sys.setrecursionlimit(100000)

class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        return self.f(n, m)

    def f(self, n, m):
        if n == 0:
            return 0
        x = self.f(n - 1, m)
        return (m + x) % n
```

#### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

![image-20211205100029672](figs/image-20211205100029672.png)

题目指出 pushed 是 popped 的排列 。因此，无需考虑 pushedpushed 和 poppedpopped 长度不同 或 包含元素不同 的情况。如果结果为True，说明最终栈一定是空的

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []  # 借用一个栈 模拟
        i = 0
        for num in pushed:
            stack.append(num)  # 入栈
            while stack and stack[-1] == popped[i]:  # 判断栈顶元素是否等于出栈元素，需要循环出栈，因为可能会有重复元素
                stack.pop()
                i += 1
        # 如果最后栈为空，说明都成功入栈出栈了
        return not stack
```

#### [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

![image-20211206111012952](figs/image-20211206111012952.png)

```python
class Solution:
    def strToInt(self, str: str) -> int:
        str = str.strip()
        if not str: return 0
        res, i, sign = 0, 1, 1
        int_max, int_min, bndry = 2**31-1, -2**31, 2**31//10
        if str[0] == '-': sign = -1  # 保存负号
        elif str[0] != '+': i = 0  # 如果没无符号位，则从i=0开始数字拼接
        for c in str[i:]:
            if not '0' <= c <= '9': break
            if res > bndry or res == bndry and c > '7': return int_max if sign == 1 else int_min  # 数字越界处理
            res = 10 * res + ord(c) - ord('0')
        return sign * res
```

#### [剑指 Offer 59 - II. 队列的最大值:star::star::star:](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

![image-20211207095901413](figs/image-20211207095901413.png)

```python
import queue
class MaxQueue:

    def __init__(self):
        self.queue = queue.Queue()
        self.deque = queue.deque()  # 双向队列，存储递减元素

    def max_value(self) -> int:
        if self.deque:
            return self.deque[0]
        else:
            return -1

    def push_back(self, value: int) -> None:
        # value 入队queue
        # deque队尾将小于value的元素弹出，添加value
        self.queue.put(value)
        while self.deque and self.deque[-1] < value:
            self.deque.pop()
        self.deque.append(value)

    def pop_front(self) -> int:
        # 按照队列的形式，先进先出
        if not self.deque: return -1
        ans = self.queue.get()
        if ans == self.deque[0]:
            self.deque.popleft()
        return ans


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

#### [剑指 Offer 59 - I. 滑动窗口的最大值:star::star::star::star:](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

![image-20211207103311721](figs/image-20211207103311721.png)

```python
import queue
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = queue.deque()  # 维护窗口内的递减元素的索引
        res = []
        for i in range(0, len(nums)):
            # 当要进窗口的元素大于队尾元素时，把队尾元素弹出，只保留最可能成为最大值的元素索引
            while deque and nums[i] > nums[deque[-1]]:  
                deque.pop()
            deque.append(i)

            while i - deque[0] >= k: # 队首元素 超出窗口范围，弹出
                deque.popleft()
            if i >= k -1:  # 当i遍历到第一个窗口k的位置 才添加
                res.append(nums[deque[0]])
        return res
```

#### [剑指 Offer 38. 字符串的排列:star::star:](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

![image-20211208100312891](figs/image-20211208100312891.png)

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        res = []
        s = sorted(list(s))
        path = []
        visited = [0] * len(s)
        def backtrack(path):
            if len(path) == len(s):
                res.append("".join(path))
                return
            
            for i in range(len(s)):
                if visited[i] == 1: continue
                if i > 0 and s[i] == s[i-1] and visited[i-1] == 0: continue
                visited[i] = 1
                backtrack(path + [s[i]])
                visited[i] = 0

        backtrack(path)
        return res
```

#### [剑指 Offer 37. 序列化二叉树:star::star::star::star:](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

![image-20211208104520653](figs/image-20211208104520653.png)

```python
class Codec:
    def serialize(self, root):
        if not root: return "[]"
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else: res.append("null")
        return '[' + ','.join(res) + ']'

    def deserialize(self, data):
        print(data)
        if data == "[]": return
        vals, i = data[1:-1].split(','), 1
        root = TreeNode(int(vals[0]))
        queue = collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        return root
```

#### [剑指 Offer 49. 丑数:star::star::star:](https://leetcode-cn.com/problems/chou-shu-lcof/)

![image-20211209095314594](figs/image-20211209095314594.png)

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * (n)
        a, b, c = 0, 0, 0
        for i in range(1, n):
            na, nb, nc = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(na, nb, nc)
            # n=6时，a=3，b=2，此时a和b都满足条件，则两个索引都要前进一位
            if dp[i] == na: a += 1
            if dp[i] == nb: b += 1
            if dp[i] == nc: c += 1
        return dp[-1]
```

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = [1]
        seen = {1}
        factor = [2, 3, 5]

        for i in range(1, n+1):
            item = heapq.heappop(heap)
            if i == n:
                return item
            else:
                for f in factor:
                    new_item = f * item
                    if new_item not in seen:
                        seen.add(new_item)
                        heapq.heappush(heap, new_item)
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

![image-20211210101111472](figs/image-20211210101111472.png)

![image-20211210101227515](figs/image-20211210101227515.png)

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        self.ans = 0
        def merge_sort(nums):
            n = len(nums)
            if n <= 1: return nums
            # 拆分
            mid = n // 2
            left = merge_sort(nums[: mid])
            right = merge_sort(nums[mid:])
            
            # 合并
            left_point, right_point, res = 0, 0, []
            while left_point < len(left) and right_point < len(right):
                # 不是逆序
                if left[left_point] <= right[right_point]:
                    res.append(left[left_point])
                    left_point += 1
                else:
                    res.append(right[right_point])
                    self.ans += len(left) - left_point
                    right_point += 1
            if left_point < len(left): res += left[left_point:]
            elif right_point < len(right): res += right[right_point:]
            return res
        
        merge_sort(nums)
        return self.ans
```

#### [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

![image-20211211103542476](figs/image-20211211103542476.png)

```python
# https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/1n-zheng-shu-zhong-1-chu-xian-de-ci-shu-umaj8/
class Solution:
    def countDigitOne(self, n: int) -> int:
        # mulk 表示 10^k
        # 在下面的代码中，可以发现 k 并没有被直接使用到（都是使用 10^k）
        # 但为了让代码看起来更加直观，这里保留了 k
        k, mulk = 0, 1
        ans = 0
        while n >= mulk:
            ans += (n // (mulk * 10)) * mulk + min(max(n % (mulk * 10) - mulk + 1, 0), mulk)
            k += 1
            mulk *= 10
        return ans
```

#### [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

![image-20211211110910714](figs/image-20211211110910714.png)

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        digit, start, count = 1, 1, 9
        while n > count: # 1.
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        num = start + (n - 1) // digit # 2.
        return int(str(num)[(n - 1) % digit]) # 3.
```

#### [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

![image-20211211111825858](figs/image-20211211111825858.png)

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        states = [
            { ' ': 0, 's': 1, 'd': 2, '.': 4 }, # 0. start with 'blank'
            { 'd': 2, '.': 4 } ,                # 1. 'sign' before 'e'
            { 'd': 2, '.': 3, 'e': 5, ' ': 8 }, # 2. 'digit' before 'dot'
            { 'd': 3, 'e': 5, ' ': 8 },         # 3. 'digit' after 'dot'
            { 'd': 3 },                         # 4. 'digit' after 'dot' (‘blank’ before 'dot')
            { 's': 6, 'd': 7 },                 # 5. 'e'
            { 'd': 7 },                         # 6. 'sign' after 'e'
            { 'd': 7, ' ': 8 },                 # 7. 'digit' after 'e'
            { ' ': 8 }                          # 8. end with 'blank'
        ]
        p = 0                           # start with state 0
        for c in s:
            if '0' <= c <= '9': t = 'd' # digit
            elif c in "+-": t = 's'     # sign
            elif c in "eE": t = 'e'     # e or E
            elif c in ". ": t = c       # dot, blank
            else: t = '?'               # unknown
            if t not in states[p]: return False
            p = states[p][t]
        return p in (2, 3, 7, 8)
```

#### [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

![image-20211211111936107](figs/image-20211211111936107.png)

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [1 / 6] * 6
        for i in range(2, n + 1):
            tmp = [0] * (5 * i + 1)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j + k] += dp[j] / 6
            dp = tmp
        return dp
```

