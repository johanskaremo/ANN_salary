import random
from model.engine import Value
import numpy as np

def softmax(logits):
    # exps = [logit.exp() for logit in logits]
    # exp_tot = sum(exps)
    # softmax_probs = [(expi / exp_tot) for expi in exps]

    exps = [logit.exp() for logit in logits]
    exp_tot = Value(0)  # Start with a Value object instead of a Python scalar

    for exp in exps:
        exp_tot = exp_tot + exp
    softmax_probs = [(expi / exp_tot) for expi in exps]
    # def _backward(out, idx, softmax_probs = softmax_probs, logits = logits):
    #     for i, logit in enumerate(logits):
    #         if i == idx:
    #             delta = softmax_probs[i] * (1 - softmax_probs[i])
    #         else:
    #             delta = -softmax_probs[i] * softmax_probs[idx]
    #         logit.grad += out.grad * delta
    #
    # for idx, out in enumerate(softmax_probs):
    #         out._backward = lambda out=out, idx=idx: _backward(out, idx)
    return softmax_probs

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
            x = layer(x)    #går från höger tillvänster. Den förra outputen x är nu input till nästa lager
            if i == len(self.layers) - 1:
                x = softmax(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"