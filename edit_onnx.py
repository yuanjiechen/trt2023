import torch
import onnx
import onnx_graphsurgeon as gs
import copy

file_name = "./onnxs/controlnet_one_loop_folded.onnx"
edit_nodes = []
last_nodes = []

graph = gs.import_onnx(onnx.load(file_name))
cast_node = []
for node in graph.nodes:
    if node.op == "Cast":
        while len(node.outputs[0].outputs) != 0:
            outputs_node = node.outputs[0].outputs[0]
            for i, input_node in enumerate(outputs_node.inputs):
                if len(input_node.inputs) > 0 and id(node) == id(input_node.inputs[0]):
                    outputs_node.inputs[i] = node.inputs[0]
        
        node.outputs.clear()
        node.inputs.clear()

    elif node.op == "InstanceNormalization":
    #     i = node.i()
    #     o = node.o()
        node.attrs["eps"] = node.attrs["epsilon"]
        node.attrs["num_groups"] = 32
        node.attrs["bSwish"] = 0
        node.op = "GroupNormalizationPlugin2"
        mul = node.o().o()
        add = node.o().o().o()
        res_1 = node.i()
        res_2 = node.o()
        sig_conv = node.o().o().o().o()
        if sig_conv.op == "Sigmoid":
            sig_mul = node.o().o().o().o(1, 0)

        node.inputs[0] = node.i().i().outputs[0]
        node.inputs[1] = mul.inputs[1]
        node.inputs[2] = add.inputs[1]
        sig_conv.inputs[0] = node.outputs[0]
        if sig_conv.op == "Sigmoid":
            sig_mul.inputs[0] = node.outputs[0]
        mul.inputs.clear()
        add.outputs.clear()
        res_1.inputs.clear()
        res_2.inputs.clear()
        # print(node.o().o().o().o().op)
        # input()
    #     node.inputs[0] = node.i().inputs[0]
    #     node.outputs[0] = node.o().outputs[0]

    #     i.inputs.clear()
    #     i.outputs.clear()

    #     o.inputs.clear()
    #     o.outputs.clear()


graph.cleanup(True, True, True).toposort()
onnx.save(gs.export_onnx(graph), file_name, save_as_external_data=True)
import os
os._exit(0)

# for gelu plugin
# for node in graph.nodes:
#     if node.op == "Split" and node.o(0, 1).op == 'Div' and node.o(0, 1).o().op == "Erf":
#         edit_nodes.append(node)
#         last_nodes.append(node.o())
#         # print(node.o().op)

# for i, (start, end) in enumerate(zip(edit_nodes, last_nodes)):
#     start.o(0, 1).inputs.clear()
#     # start.o(1, 1).inputs.clear()
#     start.o().inputs.clear()
#     gelu = gs.Node(op="CustomGeluPlugin", name=f"GELU_{i}", inputs=[start.outputs[1], start.outputs[0]], outputs=end.outputs)
#     gelu.attrs = {"type_id": 1}


#     end.outputs.clear()
#     graph.nodes.append(gelu)
# print(len(edit_nodes))
# graph.cleanup(True, True, True).toposort()

# onnx.save(gs.export_onnx(graph), file_name, save_as_external_data=True)