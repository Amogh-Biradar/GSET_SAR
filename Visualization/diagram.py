import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(-8, 21)
ax.axis('off')

colors = {
    'conv': '#5A9BD4',
    'deconv': '#8FD14F',
    'activation': '#FFA726',
    'pool': '#FFF176',
    'linear': '#81C784',
    'input': '#BA68C8',
    'output': '#E57373'
}

def draw_layers(layers):
    for i, layer in enumerate(layers):
        x, y = layer['pos']
        w, h = layer['size']
        color = colors[layer['type']]
        text_color = 'white' if color in ['#5A9BD4', '#BA68C8', '#E57373'] else 'black'
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, layer['name'], ha='center', va='center',
                fontsize=9, fontweight='bold', color=text_color)

    # Arrows
    for i in range(len(layers) - 1):
        x1, y1 = layers[i]['pos']
        x2, y2 = layers[i + 1]['pos']
        ax.annotate('', xy=(x2, y2 + layers[i + 1]['size'][1]/2),
                    xytext=(x1, y1 - layers[i]['size'][1]/2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Denoiser CNN
denoiser_layers = [
    {'name': 'Input\nMixed Wave', 'type': 'input', 'pos': (5, 19), 'size': (2, 1.5)},
    {'name': 'Conv1D\n1→16', 'type': 'conv', 'pos': (5, 17.2), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 16.1), 'size': (1.2, 0.6)},
    {'name': 'Conv1D\n16→64', 'type': 'conv', 'pos': (5, 14.5), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 13.4), 'size': (1.2, 0.6)},
    {'name': 'Conv1D\n64→256', 'type': 'conv', 'pos': (5, 11.8), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 10.7), 'size': (1.2, 0.6)},
    {'name': 'Conv1D\n256→512', 'type': 'conv', 'pos': (5, 9.1), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 8), 'size': (1.2, 0.6)},
    {'name': 'ConvT1D\n512→256', 'type': 'deconv', 'pos': (5, 6.4), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 5.3), 'size': (1.2, 0.6)},
    {'name': 'ConvT1D\n256→64', 'type': 'deconv', 'pos': (5, 3.7), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, 2.6), 'size': (1.2, 0.6)},
    {'name': 'ConvT1D\n64→16', 'type': 'deconv', 'pos': (5, 1.0), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (5, -0.1), 'size': (1.2, 0.6)},
    {'name': 'Conv1D\n16→1', 'type': 'conv', 'pos': (5, -1.7), 'size': (2, 1.2)},
    {'name': 'Tanh', 'type': 'activation', 'pos': (5, -2.8), 'size': (1.2, 0.6)},
    {'name': 'Output\nDenoised Wave', 'type': 'output', 'pos': (5, -4.5), 'size': (2, 1.5)},
]

# Classifier CNN
classifier_layers = [
    {'name': 'Input\nSound Wave', 'type': 'input', 'pos': (13, 19), 'size': (2, 1.5)},
    {'name': 'Conv1D\n1→16', 'type': 'conv', 'pos': (13, 17.2), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (13, 16.1), 'size': (1.2, 0.6)},
    {'name': 'MaxPool /4', 'type': 'pool', 'pos': (13, 15.0), 'size': (1.6, 0.6)},
    {'name': 'Conv1D\n16→64', 'type': 'conv', 'pos': (13, 13.2), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (13, 12.1), 'size': (1.2, 0.6)},
    {'name': 'MaxPool /4', 'type': 'pool', 'pos': (13, 11.0), 'size': (1.6, 0.6)},
    {'name': 'Conv1D\n64→128', 'type': 'conv', 'pos': (13, 9.2), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (13, 8.1), 'size': (1.2, 0.6)},
    {'name': 'MaxPool /4', 'type': 'pool', 'pos': (13, 7.0), 'size': (1.6, 0.6)},
    {'name': 'Conv1D\n128→256', 'type': 'conv', 'pos': (13, 5.2), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (13, 4.1), 'size': (1.2, 0.6)},
    {'name': 'MaxPool /4', 'type': 'pool', 'pos': (13, 3.0), 'size': (1.6, 0.6)},
    {'name': 'AdaptiveAvgPool\n→128', 'type': 'pool', 'pos': (13, 1.2), 'size': (2, 1.2)},
    {'name': 'Flatten', 'type': 'pool', 'pos': (13, 0.0), 'size': (1.6, 0.8)},
    {'name': 'Linear\n32768→64', 'type': 'linear', 'pos': (13, -1.7), 'size': (2, 1.2)},
    {'name': 'ReLU', 'type': 'activation', 'pos': (13, -2.8), 'size': (1.2, 0.6)},
    {'name': 'Linear\n64→1', 'type': 'linear', 'pos': (13, -4.3), 'size': (2, 1.2)},
    {'name': 'Sigmoid', 'type': 'activation', 'pos': (13, -5.4), 'size': (1.2, 0.6)},
    {'name': 'Output\nScream?', 'type': 'output', 'pos': (13, -7.0), 'size': (2, 1.5)},
]

draw_layers(denoiser_layers)
draw_layers(classifier_layers)

# Section titles
ax.text(5, 20.5, 'Denoiser CNN Architecture', ha='center', fontsize=14, fontweight='bold')
ax.text(13, 20.5, 'Scream Classifier CNN Architecture', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("cnn_architectures_combined_vertical.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
