# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for circuit cutting"""
import networkx as nx


def get_graph(tape, include_inputs_and_outputs=True, inputs_local=True,
                  measurements_local=False):
    node_counter = 0
    node_data = []

    if inputs_local:
        wire_latest_node = {w: i for i, w in enumerate(tape.wires)}
    else:
        wire_latest_node = {w: 0 for i, w in enumerate(tape.wires)}

    edge_data = []
    edge_labels = {}
    pair_counts = {}

    input_nodes = []
    measure_nodes = []

    if inputs_local:
        for _ in tape.wires:
            node_data.append((node_counter, {"label": "prep"}))
            input_nodes.append(node_counter)
            node_counter += 1
    else:
        node_data.append((node_counter, {"label": "prep"}))
        input_nodes.append(node_counter)
        node_counter += 1

    for op in tape.operations:
        node_data.append((node_counter, {"op": op, "label": op.name}))

        for wire in op.wires:
            e_start = wire_latest_node[wire]
            e_end = node_counter
            e = (e_start, e_end)

            wire_latest_node[wire] = node_counter
            edge_data.append(e)

            try:
                pair_counts[e] += 1
            except KeyError:
                pair_counts[e] = 0

            e_multi = (*e, pair_counts[e])
            edge_labels[e_multi] = wire

        node_counter += 1

    if not measurements_local:
        node_data.append((node_counter, {"label": "meas"}))

        for w, v in wire_latest_node.items():
            e = (v, node_counter)
            edge_data.append(e)

            try:
                pair_counts[e] += 1
            except KeyError:
                pair_counts[e] = 0

            e_multi = (*e, pair_counts[e])
            edge_labels[e_multi] = w

        measure_nodes.append(node_counter)
        node_counter += 1
    else:
        for wire in tape.wires:
            node_data.append((node_counter, {"label": "meas"}))

            e = (wire_latest_node[wire], node_counter)
            edge_data.append(e)

            e_multi = (*e, 0)
            edge_labels[e_multi] = wire

            measure_nodes.append(node_counter)
            node_counter += 1

    g = nx.MultiDiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    nx.set_edge_attributes(g, edge_labels, "label")

    if not include_inputs_and_outputs:
        new_node_data = []
        for d in node_data:
            if d[0] not in input_nodes and d[0] not in measure_nodes:
                new_node_data.append(d)
        node_data = new_node_data

        new_edge_data = []

        for e in edge_data:
            if e[0] not in input_nodes and e[1] not in input_nodes and e[0] not in measure_nodes and \
                    e[1] not in measure_nodes:
                new_edge_data.append(e)
        edge_data = new_edge_data

    g_simple = nx.MultiDiGraph()
    g_simple.add_nodes_from(node_data)
    g_simple.add_edges_from(edge_data)
    nx.set_edge_attributes(g_simple, edge_labels, "label")

    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    g.nodes[0]['pos'] = pos

    first_key_simple = list(g_simple.nodes)[0]
    g_simple.nodes[first_key_simple]["pos"] = pos

    return g if include_inputs_and_outputs else g_simple


def draw(g, edge_colors=None, node_colors=None, font_size=None, node_size=None,
             highlight_gates=False, highlight_edges=None):
    first_key_simple = list(g.nodes)[0]
    pos = g.nodes[first_key_simple]["pos"]

    node_labels = dict(g.nodes.data("label"))

    edge_labels = {}

    for e in g.edges(data=True):
        l = str(e[-1]["label"])
        e = e[:2]

        if e not in edge_labels:
            edge_labels[e] = str(l)
        else:
            current_label = edge_labels[e]
            new_label = current_label + "," + l
            edge_labels[e] = new_label

    if highlight_gates and node_colors is None:
        node_colors = []

        for n, d in g.nodes(data=True):
            if d["label"] in ("prep", "meas"):
                node_colors.append(cols[0])
            else:
                node_colors.append(cols[1])

    if highlight_edges and edge_colors is None:
        edge_colors = []

        for e in g.edges:
            e = e[:2]
            if e in highlight_edges:
                edge_colors.append(cols[2])
            else:
                edge_colors.append("black")

    if node_size:
        nx.draw(g, pos, with_labels=False, edge_color=edge_colors, node_color=node_colors,
                node_size=node_size)
    else:
        nx.draw(g, pos, with_labels=False, edge_color=edge_colors, node_color=node_colors)
    if font_size:
        nx.draw_networkx_labels(g, pos, node_labels, font_size=font_size)
        nx.draw_networkx_edge_labels(g, pos, edge_labels, font_size=font_size)
    else:
        nx.draw_networkx_labels(g, pos, node_labels)
        nx.draw_networkx_edge_labels(g, pos, edge_labels)
