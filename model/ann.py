import random
from model.engine import Value
import numpy as np

def softmax(logits):
    exps = [logit.exp() for logit in logits]
    exp_tot = Value(0)
    for exp in exps:
        exp_tot += exp.data
    softmax_probs = [expi.data / exp_tot for expi in exps]
    softmax_values = [Value(prob.data, _children=tuple(logits), _op="softmax") for prob in softmax_probs]
    # exps = [np.exp(logit.data) for logit in logits]
    # exp_tot = sum(exps)
    # softmax_probs = [(expi / exp_tot) for expi in exps]
    # softmax_values = [Value(val, (logit,), "softmax") for val, logit in zip(softmax_probs, logits)]

    def _backward(out, idx, softmax_probs = softmax_probs, logits = logits):
        print(f"Backward called for softmax output index {idx}")
        for i, logit in enumerate(logits):
            if i == idx:
                delta = softmax_probs[i] * (1 - softmax_probs[i])
            else:
                delta = -softmax_probs[i] * softmax_probs[idx]
            logit.grad += out.grad * delta

    for idx, out in enumerate(softmax_values):
            out._backward = lambda out=out, idx=idx: _backward(out, idx)
            print(f"Assigned backward for index {idx}")
    return softmax_values

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = softmax(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"