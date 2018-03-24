from networks import Exp1, Exp2, Exp3, Exp4, Exp5, Exp6
from utils import save_graph

net1 = Exp1(n_layers=3)
save_graph('results/exp1.dot', (8, 16), net1)

net2 = Exp2(n_layers=3)
save_graph('results/exp2.dot', (8, 16), net2)

net3 = Exp3(n_layers=3)
save_graph('results/exp3.dot', (8, 16), net3)

net4 = Exp4(n_layers=5)
save_graph('results/exp4.dot', (8, 16), net4)

net5 = Exp5(n_layers=3)
save_graph('results/exp5.dot', (8, 3, 64, 64), net5)

net6 = Exp6(n_layers=3)
save_graph('results/exp6.dot', (8, 3, 64, 64), net6)
