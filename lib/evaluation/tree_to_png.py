import io
from sklearn.tree import export_graphviz
import pydot

""" --------------------------------------------------------------------
tree_to_png() takes input: dtree, filename 
'dtree' is an object instantiated by e.g. DecisionTreeRegressor()  
the output is written to a PNG file defined with 'filename'    
-------------------------------------------------------------------- """

def tree_to_png(dtree, filename):
    dotfile = io.StringIO()
    export_graphviz(dtree, out_file=dotfile,filled=True, rounded=True, special_characters=True)
    pydot.graph_from_dot_data(dotfile.getvalue())[0].write_png(filename)

