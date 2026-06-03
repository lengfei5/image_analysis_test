import os
import sys
from collections import defaultdict, deque

def read_swc(path):
    comments = []
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue

            parts = s.split()
            if len(parts) < 7:
                continue

            node = {
                "id": int(float(parts[0])),
                "type": int(float(parts[1])),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "r": float(parts[5]),
                "parent": int(float(parts[6])),
                "order": line_no,
            }
            nodes.append(node)

    return comments, nodes

def find_components(nodes):
    id_to_node = {n["id"]: n for n in nodes}
    adj = defaultdict(list)

    for n in nodes:
        nid = n["id"]
        pid = n["parent"]
        if pid != -1 and pid in id_to_node:
            adj[nid].append(pid)
            adj[pid].append(nid)
        else:
            adj[nid] = adj[nid]

    visited = set()
    components = []

    for n in sorted(nodes, key=lambda x: x["order"]):
        start = n["id"]
        if start in visited:
            continue

        comp = []
        q = deque([start])
        visited.add(start)

        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        components.append(comp)

    return components

def write_component(out_path, comments, comp_nodes):
    comp_nodes = sorted(comp_nodes, key=lambda x: x["order"])
    old_to_new = {n["id"]: i + 1 for i, n in enumerate(comp_nodes)}

    with open(out_path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")
        f.write(f"# split component written to {os.path.basename(out_path)}\n")

        for n in comp_nodes:
            new_id = old_to_new[n["id"]]
            old_parent = n["parent"]
            new_parent = old_to_new[old_parent] if old_parent in old_to_new else -1

            f.write(
                f"{new_id} {n['type']} {n['x']} {n['y']} {n['z']} {n['r']} {new_parent}\n"
            )

def split_swc(path):
    comments, nodes = read_swc(path)
    if not nodes:
        print("No SWC nodes found.")
        return

    components = find_components(nodes)
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path) or "."

    comp_node_sets = []
    id_to_node = {n["id"]: n for n in nodes}
    for comp in components:
        comp_node_sets.append([id_to_node[nid] for nid in comp])

    for i, comp_nodes in enumerate(comp_node_sets, start=1):
        out_name = f"{base}_neuron_{i:03d}.swc"
        out_path = os.path.join(out_dir, out_name)
        write_component(out_path, comments, comp_nodes)
        print(f"Wrote {out_path} ({len(comp_nodes)} nodes)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_swc.py input.swc")
        sys.exit(1)

    split_swc(sys.argv[1])
