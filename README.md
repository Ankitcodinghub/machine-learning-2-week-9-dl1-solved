# machine-learning-2-week-9-dl1-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 9-DL1 Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-9-dl1-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98843&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 9-DL1 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
Exercise Sheet 9

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 1: Convolutional Neural Networks (10 P)

Let x = (x(i))i be a multivariate time series represented as a collection of one-dimensional signals. For simplicity of the following derivations, we consider these signals to be of infinite length. We feed x as input to a convolutional layer. The forward computation in this layer is given by:

</div>
</div>
<div class="layoutArea">
<div class="column">
z(j) = 􏰄 􏰇w(ij) ⋆ x(i)􏰈 tt

i

∞

</div>
</div>
<div class="layoutArea">
<div class="column">
=􏰄 􏰄 w(ij)·x(i) s t+s

i s=−∞

</div>
<div class="column">
forall l and t∈Z

</div>
</div>
<div class="layoutArea">
<div class="column">
It results in z = (z(j))j another multivariate time series again composed of a collection of output signals also assumed to be of infinite length. Convolution filters w(ij) have to be learned from the data. After passing the data through the convolutional layer, the neural network output is given as y = f(z) where f is some top-layer function assumed to be differentiable. To learn the model, parameters gradients need to be computed.

(a) Express the gradient ∂y/∂w as a function of the input x and of the gradient ∂y/∂z. Exercise 2: Recursive Neural Networks (10 + 10 P)

Consider a recursive neural network that applies some function φ recursively from the leaves to the root of a binary parse tree. The function φ : Rd × Rd × Rh → Rd takes two incoming nodes ai, aj ∈ Rd and some parameter vector θ ∈ Rh as input, and produces an output ak ∈ Rd. Once we have reached the root node, we apply a function f : Rd → R that produces some real-valued prediction. We assume that both φ and f are differentiable with their inputs.

Consider the sentence ‘the cat sat on the mat’, that is parsed as: ((the, cat) , (sat, (on, (the, mat))))

The leaf nodes of the parsing tree are represented by word embeddings athe, acat, · · · ∈ Rd.

(a) Draw the computational graph associated to the application of the recursive neural network to this sentence,

and write down the set of equations, e.g.

a1 = φ(athe, acat, θ) a2 = φ(athe, amat, θ) a3 = φ(aon, a2, θ)

.

y = f(a5)

(b) Express the total derivative dy/dθ (taking into account all direct and indirect dependencies on the parameter θ) in terms of local derivatives (relating adjacent quantities in the computational graph).

(Hint: you can use for this the chain rule for derivatives that states that for some function h(g1(t), . . . , gN (t))

</div>
</div>
<div class="layoutArea">
<div class="column">
we have:

</div>
<div class="column">
dh= ∂h ·dg1 +···+ ∂h ·dgN dt ∂g1 dt ∂gN dt

</div>
</div>
<div class="layoutArea">
<div class="column">
where d(·) and ∂(·) denote the total and partial derivatives respectively.)

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Exercise 3: Graph Neural Networks (10 + 10 P)

Graph neural networks are a fairly flexible class of neural networks that can sometimes be seen as a gen- eralization of convolutional neural networks and recursive neural networks. We would like to study the equivalence of graph neural network with other neural networks. We will consider in this exercise graph neural network that map two consecutive layers using the equation:

Ht+1 =ρ(ΛHtW)

where ρ denotes the ReLU function. The adjacency matrix Λ is of size N × N with value 1 when there is a connection, and 0 otherwise. The matrix of parameters W is of size d × d, where d is the number of dimensions used to represent each node. We also assume that the initial state H0 is a matrix filled with ones.

<ol>
<li>(a) &nbsp;Consider first the following undirected graph:
Because the graph is unlabeled, the nodes can only be distinguished by their connectivity to other nodes. Consider the case where the graph neural network has depth 2. Depict the multiple trees formed by viewing the graph neural network as a collection of recursive neural networks.
</li>
<li>(b) &nbsp;Consider now the following infinite lattice graph:</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
which is like in the previous example undirected and unlabeled. We consider again the case where the graph neural network has depth 2. Show that the latter can be implemented as a 2D convolutional neural network with four convolution layers and two ReLU layer, i.e. give the sequence of layers and their parameters.

Exercise 4: Programming (50 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
Structured Neural Networks

In this homework, we train a collection of neural networks including a convolutional neural network on the MNIST dataset, and a graph neural network on some graph classification task.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [1]:

</div>
<div class="column">
<pre>import torch
import torch.nn as nn
</pre>
<pre>import torchvision
import torchvision.transforms as transforms
import utils
import numpy
</pre>
import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt

</div>
</div>
<div class="layoutArea">
<div class="column">
We first consider the convolutional neural network, which we apply in the following to the MNIST data.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [2]:

</div>
<div class="column">
<pre>transform = transforms.Compose(
    [transforms.ToTensor(),
</pre>
transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root=’./data’, train=True,

download=True, transform=transform) testset = torchvision.datasets.MNIST(root=’./data’, train=False,

download=True, transform=transform) Xr,Tr = trainset.data.float().view(-1,1,28,28)/127.5-1,trainset.targets

<pre>Xt,Tt = testset.data.float().view(-1,1,28,28)/127.5-1,testset.targets
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We consider for this dataset a convolution network made of four convolutions and two pooling layers.

In [3]: torch.manual_seed(0)

cnn = utils.NNClassifier(nn.Sequential(

<pre>                  nn.Conv2d( 1, 8, 5), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Conv2d( 8, 24, 5), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Conv2d( 24, 72, 4), nn.ReLU(),
                  nn.Conv2d( 72, 10, 1)
</pre>
))

The network is wrapped in the class utils.NNClassifier , which exposes scikit-learn-like functions such as fit() and predict() . To evaluate the convolutional neural network, we also consider two simpler baselines: a one-layer linear network, and

standard fully-connected network composed of two layers.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [4]:

</div>
<div class="column">
torch.manual_seed(0)

lin = utils.NNClassifier(nn.Sequential(nn.Linear(784, 10)),flat=True)

<pre>torch.manual_seed(0)
fc = utils.NNClassifier(nn.Sequential(
</pre>
nn.Linear( 784, 512), nn.ReLU(), nn.Linear( 512, 10) ),flat=True)

</div>
</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
Evaluating the convolutional neural network (15 P)

We now proceed with the comparision of these three classifiers.

Task:

Train each classifier for 5 epochs and print the classification accuracy on the training and test data (i.e. the fraction of the examples that are correctly classified). To avoid running out of memory, predict the training and test accuracy only based on the 2500 first examples of the training and test set respectively.

In [5]: for name,cl in [(‘linear’,lin),(‘full’,fc),(‘conv’,cnn)]:

# ———————————— # TODO: Replace by your code

# ———————————— import solution

<pre>                  errtr,errtt = solution.analyze(cl,Xr,Tr,Xt,Tt)
</pre>
<pre>                  # ------------------------------------
</pre>
print(‘%10s train: %.3f test: %.3f’%(name,errtr,errtt))

<pre>                  linear train: 0.910  test: 0.878
                    full train: 0.966  test: 0.954
                    conv train: 0.990  test: 0.982
</pre>
We observe that the convolutional neural network reaches the higest accuracy with less than 2% of misclassified digits on the test data.

Confidently predicted digits (15 P)

We now ask whether some digits are easier to predict than others for the convolutional neural network. For this, we observe that the neural network produces at its output scores yc for each class c. These scores can be converted to a class probability using the softargmax (also called softmax) function:

pc = exp(yc) ∑1′0 exp(yc′ )

Task:

Find for the convolutional network the data points in the test set that are predicted with the highest probability (the lowest being random guessing). To avoid numerical unstability, your implementation should work in the log-probability domain and make use of numerically stable functions of numpy/scipy such as logsumexp.

</div>
</div>
<div class="layoutArea">
<div class="column">
c =1

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
In [6]:

</div>
<div class="column">
# ———————————— # TODO: Replace by your code

# ———————————— import solution

<pre>highest,lowest = solution.highestlowest(cnn,Xt)
</pre>
<pre># ------------------------------------
</pre>
for digits in [highest,lowest]:

plt.figure(figsize=(8,3))

plt.axis(‘off’) plt.imshow(digits.numpy().reshape(3,8,28,28).transpose(0,2,1,3).reshape(28*3,28*

<pre>8),cmap='gray')
    plt.show()
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the most confident digits are thick and prototypical. Interestingly, the highest confidence digits are all from the class “3”. The low-confidence digits are on the other hand thiner, and are often also more difficult to predict for a human.

Graph Neural Network (20 P)

We consider a graph neural network (GNN) that takes as input graphs of size m given by their adjacency matrix A and which is composed of the following four layers:

H0 = U

H1 = ρ(ΛH0W) H2 = ρ(ΛH1W) H3 = ρ(ΛH2W)

y = 1⊤H3V

U is a matrix of size m×h, W is a matrix of size h×h, V is a matrix of size h×3and Λis the normalized Laplacian associated to the graph adjacency matrix A (i.e. Λ = D−0.5 AD−0.5 where D is a diagonal matrix containing the degree of each node), and ρ(t) = max(0, t) is the rectified linear unit that applies element-wise.

Task:

Implement the forward function of the GNN. It should take as input a minibatch of adjacency matrices A (given as a 3-dimensional tensor of dimensions (minibatch_size × number_nodes × number_nodes)) and return a matrix of size minibatch_size × 3 representing the scores for each example and predicted class.

(Note: in your implementation use array operations instead of looping over all individual examples of the minibatch.)

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
In [7]: class GNN(torch.nn.Module):

</div>
</div>
<div class="layoutArea">
<div class="column">
def __init__(self,nbnodes,nbhid,nbclasses): torch.nn.Module.__init__(self)

self.m = nbnodes

self.h = nbhid

<pre>                      self.c = nbclasses
</pre>
<pre>                      self.U = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0,nbnodes**
              -.5,[nbnodes,nbhid])))
</pre>
<pre>                      self.W = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0,nbhid**-.
              5,[nbhid,nbhid])))
</pre>
<pre>                      self.V = torch.nn.Parameter(torch.zeros([nbhid,nbclasses]))
</pre>
def forward(self,A):

# ———————————— # TODO: Replace by your code

# ———————————— import solution

Y = solution.forward(self,A)

# ————————————

return Y

The graph neural network is now tested on a simple graph classification task where the three classes correspond to star-shaped, chain-shaped and random-shaped graphs. Because the GNN is more difficult to optimize and the dataset is smaller, we train the network for 500 epochs. We compare the GNN with a simple fully-connected network built directly on the adjacency matrix.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [8]:

</div>
<div class="column">
<pre>Ar,Tr,At,Tt = utils.graphdata()
</pre>
torch.manual_seed(0)

dnn = utils.NNClassifier(nn.Sequential(nn.Linear( 225,512), nn.ReLU(),nn.Linear(512, 3)),flat=True)

torch.manual_seed(0)

gnn = utils.NNClassifier(GNN(15,25,3))

for name,net in [(‘DNN’,dnn),(‘GNN’,gnn)]: net.fit(Ar,Tr,lr=0.01,epochs=500)

Yr = net.predict(Ar)

Yt = net.predict(At)

acctr = (Yr.max(dim=1)[1] == Tr).data.numpy().mean()

acctt = (Yt.max(dim=1)[1] == Tt).data.numpy().mean() print(‘name: %10s train: %.3f test: %.3f’%(name,acctr,acctt))

<pre>name:        DNN  train: 1.000  test: 0.829
name:        GNN  train: 1.000  test: 0.965
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that both networks are able to perfectly classify the training data, however, due to its particular structure, the graph neural network generalizes better to new data points.

</div>
</div>
</div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
