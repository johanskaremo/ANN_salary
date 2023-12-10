import numpy as np
from collections import deque
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        # The natural logarithm of the value
        threshold = 1e-9
        # Ensure self.data is not less than the threshold
        safe_data = max(self.data, threshold)
        out = Value(np.log(safe_data), (self,), 'log')

        def _backward():
          self.grad += out.grad / self.data if self.data > threshold else 0

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1.0)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        if self.data > 10:
            self.data = 10
        out = Value(np.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def relu(self):

        out = Value(0.0 if self.data < 0.0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0.0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        queue = deque([self])  # Using queue for popping first element
        # Use popleft for FIFO behavior
        while queue:
            v = queue.popleft()  # This pops from the beginning of the queue
            if v not in visited:
                visited.add(v)
                queue.extend(v._prev)  # Add children to the end of the queue
                topo.append(v)

        topo.reverse()
        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0     #set "Loss" gradient to 1 to start the process
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1.0

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1.0

    def __rtruediv__(self, other): # other / self
        return other * self**-1.0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"