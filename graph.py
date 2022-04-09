from typing import List, Optional
import numpy as np
import networkx as nx
from dezero.core import Function, Variable, pow, sub, mul
import matplotlib.pyplot as plt


x = Variable(np.array(2.0), name="x")
a: Variable = pow(x, 4, name="f1")
b: Variable = pow(x, 2, name="f2")
c: Variable = mul(b, Variable(np.array(2.0)), name="f3")
y: Variable = sub(a, c, name="f4")
a.name = "a"
b.name = "b"
c.name = "c"


stack: List[Variable] = [y]
g: nx.DiGraph = nx.DiGraph()
while stack:
    output: Variable = stack.pop()
    f: Optional[Function] = output.creator
    #  木構造のルートのVariableだった場合は、creatorは存在しないため処理をスキップする
    if f is None:
        continue
    # 関数とそのoutputのエッジを追加
    g.add_edge(f.name, output.name)
    inputs: List[Variable] = f.inputs
    for input in inputs:
        # 関数とそのinputのエッジを追加
        g.add_edge(input.name, f.name)

    stack.extend(f.inputs)


nx.draw_networkx(g, arrows=True)
plt.show()
