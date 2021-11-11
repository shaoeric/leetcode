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

