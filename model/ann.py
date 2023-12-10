import random
from model.engine import Value

def softmax(logits):
    exps = [logit.exp() for logit in logits]
    exp_tot = Value(0)

    #calculate softmax probabilities for all output neurons
    for exp in exps:
        exp_tot = exp_tot + exp
    softmax_probs = [(expi / exp_tot) for expi in exps]

    return softmax_probs

class Module:
    #Reset gradients to zero to not have incorrect updates of parameters
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
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] #creates a list which essensially have nbr of neurons equal to nout

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts     #creates a total list with the initialized MLP eg. [5, 8, 8, 5]
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))] #creating layers with nbr neuron equal to intialization
        self.training = True        #nonlin applies nonlinear activation for all layers except last, training start as true

    def __call__(self, x):
        for i, layer in enumerate(self.layers): #go through all layers
            x = layer(x)    #går från höger till vänster. Den förra outputen x är nu input till nästa lager
            if i == len(self.layers) - 1:
                x = softmax(x)  #check if last layer, then apply softmax
        return x
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"