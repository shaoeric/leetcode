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

