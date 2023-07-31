import torch
import onnx
import onnx_graphsurgeon as gs

file_name = "./onnxs/controlnet_one_loop_folded_I.onnx"
edit_nodes = []
last_nodes = []

graph = gs.import_onnx(onnx.load(file_name))
cast_node = []
for node in graph.nodes:
    if node.op == "Cast":
        print("iiii")
        node.op = "Identity"
# onnx.save(gs.export_onnx(graph), file_name + "I", save_as_external_data=True)
import os
os._exit(0)

for node in graph.nodes:
    if node.op == "Split" and node.o(0, 1).op == 'Div' and node.o(0, 1).o().op == "Erf":
        edit_nodes.append(node)
        last_nodes.append(node.o())
        # print(node.o().op)

for i, (start, end) in enumerate(zip(edit_nodes, last_nodes)):
    start.o(0, 1).inputs.clear()
    # start.o(1, 1).inputs.clear()
    start.o().inputs.clear()
    gelu = gs.Node(op="CustomGeluPlugin", name=f"GELU_{i}", inputs=[start.outputs[1], start.outputs[0]], outputs=end.outputs)
    gelu.attrs = {"type_id": 1}


    end.outputs.clear()
    graph.nodes.append(gelu)
print(len(edit_nodes))
graph.cleanup(True, True, True).toposort()

onnx.save(gs.export_onnx(graph), file_name, save_as_external_data=True)