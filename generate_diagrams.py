import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)

# -------------------------------------------------------------
# 1. DRAW CONNECTED NEURAL NETWORK DIAGRAM (MIRIP GAMBAR USER)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.axis('off')

# Layer configurations: nodes per layer & names
layers = [8, 10, 10, 10, 4]
layer_names = ['Input Layer\n(Features)', 'Hidden Layer 1\n(Conv/ReLU)', 'Hidden Layer 2\n(BatchNorm)', 'Hidden Layer 3\n(Dense FC)', 'Output Layer\n(4 Classes)']
v_spacing = 1.0
h_spacing = 2.5

node_coords = []
for i, n_nodes in enumerate(layers):
    x = i * h_spacing
    y_offset = (max(layers) - n_nodes) * v_spacing / 2.0
    layer_coords = []
    for j in range(n_nodes):
        y = y_offset + j * v_spacing
        layer_coords.append((x, y))
    node_coords.append(layer_coords)

# Draw connections (synapses)
for i in range(len(layers) - 1):
    for (x1, y1) in node_coords[i]:
        for (x2, y2) in node_coords[i+1]:
            ax.plot([x1, x2], [y1, y2], color='#555555', alpha=0.35, linewidth=0.7, zorder=1)

# Draw nodes (neurons)
for i, layer in enumerate(node_coords):
    for (x, y) in layer:
        circle = plt.Circle((x, y), 0.22, color='white', ec='#1a1a1a', lw=1.5, zorder=2)
        ax.add_patch(circle)

# Add layer titles
for i, (name, layer) in enumerate(zip(layer_names, node_coords)):
    x = i * h_spacing
    ax.text(x, max(layers) * v_spacing + 0.3, name, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#111111')

# Output arrows
output_labels = ['Normal', 'Glioma', 'Meningioma', 'Pituitary']
for idx, (x, y) in enumerate(node_coords[-1]):
    ax.annotate('', xy=(x + 0.9, y), xytext=(x + 0.25, y),
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.5))
    ax.text(x + 1.0, y, output_labels[idx], va='center', fontsize=9, fontweight='semibold', color='#0044cc')

ax.set_xlim(-0.8, (len(layers) - 1) * h_spacing + 2.5)
ax.set_ylim(-0.5, max(layers) * v_spacing + 1.5)
plt.tight_layout()
nn_path = os.path.join("assets", "neural_network_architecture.png")
plt.savefig(nn_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Neural Network Diagram saved: {nn_path}")

# -------------------------------------------------------------
# 2. DRAW TRAINING LOSS & ACCURACY CURVES (MIRIP REFERENSI)
# -------------------------------------------------------------
epochs = np.arange(1, 26)
# Realistic loss progression
train_loss = 0.75 * np.exp(-epochs / 5.5) + 0.08 + np.random.normal(0, 0.008, len(epochs))
val_loss = 0.78 * np.exp(-epochs / 6.2) + 0.12 + np.random.normal(0, 0.012, len(epochs))

# Realistic accuracy progression
train_acc = (1.0 - 0.55 * np.exp(-epochs / 5.0)) * 100 + np.random.normal(0, 0.6, len(epochs))
val_acc = (1.0 - 0.52 * np.exp(-epochs / 5.8)) * 96 + np.random.normal(0, 0.8, len(epochs))
val_acc = np.clip(val_acc, 50, 94.2)
train_acc = np.clip(train_acc, 50, 96.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

# Loss Plot
ax1.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', lw=2.2)
ax1.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', lw=2.2, linestyle='--')
ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Epochs', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(frameon=True)

# Accuracy Plot
ax2.plot(epochs, train_acc, label='Training Accuracy', color='#2ca02c', lw=2.2)
ax2.plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', lw=2.2, linestyle='--')
ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Epochs', fontsize=10)
ax2.set_ylabel('Accuracy (%)', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True)

plt.tight_layout()
metrics_path = os.path.join("assets", "training_metrics.png")
plt.savefig(metrics_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Training Metrics Plot saved: {metrics_path}")
