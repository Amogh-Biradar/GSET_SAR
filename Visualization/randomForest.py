import matplotlib.pyplot as plt
import networkx as nx

def draw_decision_tree(ax, root_pos, depth=2, width=2.2, node_size=400, tree_id=1, leaf_colors=None, bins_per_tree=4):
    G = nx.DiGraph()
    positions = {}
    node_counter = [0]
    leaf_nodes = []

    def add_node(pos, parent=None):
        node_id = f"T{tree_id}_N{node_counter[0]}"
        node_counter[0] += 1
        G.add_node(node_id)
        positions[node_id] = pos
        if parent:
            G.add_edge(parent, node_id)
        return node_id

    def build_tree(x, y, d, w, parent=None):
        node = add_node((x, y), parent=parent)
        if d == 0:
            leaf_nodes.append(node)
            return [node]
        left_x = x - w / 2
        right_x = x + w / 2
        child_y = y - 1.2
        left_leaves = build_tree(left_x, child_y, d - 1, w / 2, node)
        right_leaves = build_tree(right_x, child_y, d - 1, w / 2, node)
        return left_leaves + right_leaves

    leaves = build_tree(*root_pos, depth, width)

    node_colors = []
    for n in G.nodes():
        if n in leaf_nodes and leaf_colors:
            idx = leaf_nodes.index(n)
            node_colors.append(leaf_colors[idx % bins_per_tree])
        else:
            node_colors.append("#4CAF50")

    nx.draw(G, pos=positions, ax=ax, node_size=node_size, node_color=node_colors, arrows=False)
    return G, positions, leaf_nodes

# Create 12 bin colors (30° to 360° in steps of 30°)
azimuth_bins = list(range(30, 361, 30))
azimuth_colors = plt.cm.get_cmap('hsv', len(azimuth_bins))(range(len(azimuth_bins)))

# Create plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1, 13)
ax.set_ylim(-7.5, 5.5)
ax.axis('off')

# Root positions for 3 trees
tree_roots = [(2, 2.5), (6, 2.5), (10, 2.5)]
bins_per_tree = 4

# Draw TDOA input box
ax.text(6, 4.8, 'TDOA Input', fontsize=12,
        bbox=dict(boxstyle="round", fc="#FFC107", ec="black"), ha='center')

# Arrows to each tree
for x, y in tree_roots:
    ax.annotate('', xy=(x, y + 0.2), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Draw trees with color-coded leaves
for i, pos in enumerate(tree_roots):
    start = i * bins_per_tree
    tree_colors = [azimuth_colors[start + j] for j in range(bins_per_tree)]
    leaf_colors = [tree_colors[j // (2 ** (2 - 1))] for j in range(2 ** 2)]
    draw_decision_tree(ax, pos, depth=2, tree_id=i+1, leaf_colors=leaf_colors, bins_per_tree=bins_per_tree)

# Result labels and arrows
for i, (x, y) in enumerate(tree_roots):
    ax.text(x, y - 3.0, f'Result-{i+1}', fontsize=10, ha='center')
    ax.annotate('', xy=(x, y - 2.8), xytext=(x, y - 1.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Voting box
ax.text(6, -4.5, 'Majority Voting / Averaging',
        fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="#90CAF9", ec="black"))

# Arrows from results to voting
for x, _ in tree_roots:
    ax.annotate('', xy=(6, -4.1), xytext=(x, -0.58),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Final output
ax.annotate('', xy=(6, -6.0), xytext=(6, -4.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(6, -6.4, 'Azimuth Output', fontsize=12,
        bbox=dict(boxstyle="round", fc="#E57373", ec="black"), ha='center')

# Title
ax.text(6, 5.2, 'Random Forest Model Diagram', fontsize=16,
        fontweight='bold', ha='center', color='#2C3E50')

# Legend
# Full azimuth bin color legend


plt.tight_layout()
plt.savefig("random_forest_color_bins.png", dpi=300)
plt.show()
