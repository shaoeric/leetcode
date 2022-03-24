#### [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)

<img src="figs/image-20220309152100991.png" alt="image-20220309152100991" style="zoom:67%;" />

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        pos = ((dividend >= 0) and (divisor > 0)) or ((dividend <= 0) and (divisor < 0))
        res = 0
        # 逐步递减，求商
        dividend = abs(dividend)
        divisor = abs(divisor)
        while dividend >= divisor:
            # tmp指数级翻倍
            i = 1
            tmp = divisor
            while dividend >= tmp:
                dividend -= tmp
                res += i
                i <<= 1
                tmp <<= 1

        if not pos:
            res = - res
        return res if -2**31 <= res <= 2**31-1 else 2**31-1
```

#### [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

<img src="figs/image-20220312104624147.png" alt="image-20220312104624147" style="zoom:67%;" />

```python
# 超时
class Solution:
    def numDecodings(self, s: str) -> int:
        self.res = 0
        def backtrack(start, cur_len):
            if cur_len == len(s):
                self.res += 1
                return
            if s[start] == '0':
                return
            for i in range(1, 3):
                if int(s[start: start + i]) > 26:
                    break
                if start + i > len(s):
                    break
                backtrack(start + i, cur_len + i)
        backtrack(0, 0)
        return self.res
```

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            # 情况1，只选一个字符，需要保证这个字符不是0
            if s[i-1] !=  '0':
                dp[i] += dp[i-1]
            # 情况2，选两个字符
            if i > 1 and s[i-2] != '0' and int(s[i-2: i]) <= 26:
                dp[i] += dp[i-2]
        return dp[-1]
```

#### [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

<img src="figs/image-20220315190759348.png" alt="image-20220315190759348" style="zoom:67%;" />

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        nodes = {}

        def helper(head):
            if not head: return None
            if head in nodes:
                return nodes[head]
            node = Node(head.val)
            nodes[head] = node
            node.next = helper(head.next)
            node.random = helper(head.random)
            return node
        return helper(head)
```

#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

<img src="figs/image-20220315220317216.png" alt="image-20220315220317216" style="zoom:67%;" />

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        m = len(s)
        n = len(wordDict)
        dp = [False] * (m + 1)
        dp[0] = True
        for i in range(1, m + 1):
            for j in range(n):
                if len(wordDict[j]) > i:
                    continue
                dp[i] = dp[i] or (dp[i-len(wordDict[j])] and s[i-len(wordDict[j]): i] == wordDict[j])
        return dp[-1]
```

#### [202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

<img src="figs/image-20220316185558820.png" alt="image-20220316185558820" style="zoom:67%;" />

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        s = set()
        while True:
            res = 0
            while n > 0:
                res += (n % 10) ** 2
                n //= 10
            if res == 1:
                return True
            if res in s:
                return False
            else:
                n = res
                s.add(res)
```

#### [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

<img src="figs/image-20220317093348064.png" alt="image-20220317093348064" style="zoom:67%;" />

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if mid == 0:
                if nums[mid] > nums[mid + 1]:
                    return mid
                else:
                    left = mid + 1
            if mid == len(nums) - 1:
                if nums[mid] > nums[mid - 1]:
                    return mid
                else:
                    right = mid - 1
            else:
                if nums[mid - 1] < nums[mid] and nums[mid + 1] < nums[mid]:
                    return mid
                elif nums[mid - 1] > nums[mid]:
                    right = mid -1
                else:
                    left = mid + 1
        return left
```

#### [166. 分数到小数](https://leetcode-cn.com/problems/fraction-to-recurring-decimal/)

<img src="figs/image-20220317095240346.png" alt="image-20220317095240346" style="zoom:67%;" />

```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator % denominator == 0:
            return str(numerator // denominator)
        s = []
        if (numerator < 0) != (denominator < 0):
            s.append('-')
        # 整数部分
        numerator = abs(numerator)
        denominator = abs(denominator) 
        integer = numerator // denominator
        s.append(str(integer))
        s.append('.')

        # 小数部分
        index_dic = {}
        reminder = numerator % denominator
        while reminder != 0 and reminder not in index_dic:
            index_dic[reminder] = len(s)
            reminder *= 10
            s.append(str(reminder // denominator))
            reminder %= denominator
        if reminder != 0:
            insert_idx = index_dic[reminder]
            s.insert(insert_idx, '(')
            s.append(')')
        return ''.join(s)
```

#### [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

<img src="figs/image-20220319145635448.png" alt="image-20220319145635448" style="zoom:67%;" />

```python
class Solution:
    def calculate(self, s: str) -> int:
        def calc(num_stack, op_stack):
            op, y, x = op_stack.pop(), num_stack.pop(), num_stack.pop() if num_stack else 0
            ans = 0
            if op == '+': ans = x + y
            elif op == '-': ans = x - y
            elif op == '*': ans = x * y
            elif op == '/': ans = x // y
            elif op == '%': ans = x % y
            elif op == '^': ans = math.pow(x, y)
            num_stack.append(int(ans))

        op_prio = {'+': 0, '-': 0, '*': 1, '/': 1, '%': 1, '^': 2}

        s = "(" + s.replace(' ', '').replace('(-', '(0-') + ')'
        n = len(s)
        op_stack, num_stack = [], []

        i = 0
        while i < n:
            c = s[i]
            i += 1
            if c.isdigit():
                num = int(c)
                while i < n and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i += 1
                num_stack.append(num)
            elif c == '(':
                op_stack.append(c)
            elif c == ')':
                while op_stack and op_stack[-1] != '(':
                    calc(num_stack, op_stack)
                op_stack.pop()
            else:
                while op_stack and op_stack[-1] != '(':
                    prev_op = op_stack[-1]
                    if op_prio[prev_op] < op_prio[c]:
                        break
                    calc(num_stack, op_stack)
                op_stack.append(c)
        return num_stack[0]
```

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

<img src="figs/image-20220320153116935.png" alt="image-20220320153116935" style="zoom:67%;" />

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = collections.deque()
        res = []

        for i, j in enumerate(nums):
            while q and j > nums[q[-1]]:
                q.pop()
            
            if q and q[0] <= i - k:
                q.popleft()
            
            q.append(i)

            if i - k + 1 >= 0:
                res.append(nums[q[0]])
        return res
```

#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

<img src="figs/image-20220321110400175.png" alt="image-20220321110400175" style="zoom:67%;" />

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        fast = slow = 0
        while True:
            fast = nums[nums[fast]]
            slow = nums[slow]
            if fast == slow:
                break
        fast = 0
        while True:
            fast = nums[fast]
            slow = nums[slow]
            if fast == slow:
                return slow
```

#### [350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)

<img src="figs/image-20220322104151679.png" alt="image-20220322104151679" style="zoom:67%;" />

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        left, right = 0, 0
        res = []
        while left < len(nums1) and right < len(nums2):
            if nums1[left] < nums2[right]:
                left += 1
            elif nums1[left] == nums2[right]:
                res.append(nums1[left])
                left += 1
                right += 1
            else:
                right += 1
        return res
```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)
        res = []
        m = collections.Counter()
        for n in nums1:
            m[n] += 1
        for n in nums2:
            if n in m:
                res.append(n)
                m[n] -= 1
                if m[n] == 0:
                    m.pop(n)
        return res
```

#### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

<img src="figs/image-20220323110418238.png" alt="image-20220323110418238" style="zoom:67%;" />

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        res = 0
        def dfs(x, y):
            if mem[x][y] != 0:
                return mem[x][y]
            mem[x][y] += 1
            for i, j in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if i < 0 or i >= m or j < 0 or j >= n:
                    continue
                if matrix[i][j] > matrix[x][y]:
                    mem[x][y] = max(mem[x][y], dfs(i, j) + 1)
            return mem[x][y]
        
        mem = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                res = max(res, dfs(i, j))
        return res
```

#### [371. 两整数之和](https://leetcode-cn.com/problems/sum-of-two-integers/)

<img src="figs/image-20220323111438004.png" alt="image-20220323111438004" style="zoom:67%;" />

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        a &= 0xFFFFFFFF
        b &= 0xFFFFFFFF
        while b:
            carry = (a & b)
            a ^= b
            b = (carry << 1) & 0xFFFFFFFF
        return a if a < 0x80000000 else ~(a^0xFFFFFFFF)
```

#### [380. O(1) 时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

<img src="figs/image-20220324084257772.png" alt="image-20220324084257772" style="zoom:67%;" />

```python
from random import choice
class RandomizedSet:

    def __init__(self):
        self.dic = {}
        self.lst = []

    def insert(self, val: int) -> bool:
        if val in self.dic:
            return False
        self.dic[val] = len(self.lst)
        self.lst.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.dic:
            return False
        idx = self.dic[val]
        last_val = self.lst[-1]
        self.lst[idx], self.lst[-1] = self.lst[-1], self.lst[idx]
        self.dic[last_val] = idx
        self.lst.pop()
        self.dic.pop(val)
        return True

    def getRandom(self) -> int:
        return choice(self.lst)
```

#### [378. 有序矩阵中第 K 小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)

<img src="figs/image-20220324085637117.png" alt="image-20220324085637117" style="zoom:67%;" />

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        def check(mid):
            i, j = n - 1, 0
            num = 0
            while i >= 0 and j < n:
                if matrix[i][j] <= mid:
                    num += i + 1  # 这一列 都小于等于mid
                    j += 1  # 指针指向下一列
                else:
                    i -= 1
            return num
        
        left, right = matrix[0][0], matrix[-1][-1]
        while left <= right:
            mid = left + (right - left) // 2
            cnt = check(mid)
            if cnt >= k:
                right = mid - 1
            else:
                left = mid + 1
        return left
```

