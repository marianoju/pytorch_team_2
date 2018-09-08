import io
from sklearn.tree import export_graphviz
import pydot


def tree_to_png(dtree, filename):
    dotfile = io.StringIO()
    export_graphviz(dtree, out_file=dotfile,filled=True, rounded=True, special_characters=True)
    pydot.graph_from_dot_data(dotfile.getvalue())[0].write_png(filename)

