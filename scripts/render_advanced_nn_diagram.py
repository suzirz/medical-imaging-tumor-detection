"""
High-Resolution Neural Network Architecture Visualizer.
Generates an agency-grade, publication-quality diagram of the complete
deep learning pipeline with nodes, synapses, feature maps, and activation tensors.
NOTE: LOCAL ONLY — DO NOT GIT PUSH (as requested by user).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def draw_advanced_neural_network():
    os.makedirs("assets", exist_ok=True)
    
    # Ultra-wide high-resolution canvas
    fig, ax = plt.subplots(figsize=(18, 9), facecolor="#090d16", dpi=300)
    ax.set_facecolor("#090d16")
    ax.axis("off")

    fig.suptitle(
        "NeuroScan Deep Intracranial Neural Network Architecture",
        color="#f8fafc", fontsize=18, fontweight="bold", fontfamily="sans-serif", y=0.96
    )
    plt.title(
        "Multi-Stage Feature Extraction, Spatial Attention Gates, Latent Dense Synapses, and 4-Class Diagnostic Head",
        color="#94a3b8", fontsize=11, fontfamily="sans-serif", pad=12
    )

    # Layer specifications
    layer_specs = [
        {"name": "Input MRI Slice\n(Tensor: 3 x 240 x 240)", "nodes": 6, "color": "#38bdf8", "type": "input"},
        {"name": "Stage 1: Conv2D\n(32 Filters, 7x7)", "nodes": 10, "color": "#0ea5e9", "type": "conv"},
        {"name": "Stage 2: MaxPool\n(Downsampling 60x60)", "nodes": 8, "color": "#0284c7", "type": "pool"},
        {"name": "Stage 3: Attention Gate\n(Self-Attention MHSA)", "nodes": 12, "color": "#a855f7", "type": "attn"},
        {"name": "Stage 4: Dense FC 1\n(512 Latent Neurons)", "nodes": 10, "color": "#10b981", "type": "dense"},
        {"name": "Stage 5: Dense FC 2\n(128 Latent Neurons)", "nodes": 8, "color": "#34d399", "type": "dense"},
        {"name": "Output Diagnostic Head\n(4 Pathology Classes)", "nodes": 4, "color": "#f43f5e", "type": "output"}
    ]

    h_spacing = 2.4
    v_spacing = 0.75
    max_nodes = max(l["nodes"] for l in layer_specs)

    layer_node_coords = []

    # Calculate coordinates
    for i, spec in enumerate(layer_specs):
        x = i * h_spacing
        n = spec["nodes"]
        y_offset = (max_nodes - n) * v_spacing / 2.0
        coords = []
        for j in range(n):
            y = y_offset + j * v_spacing
            coords.append((x, y))
        layer_node_coords.append(coords)

    # Draw Synaptic Connections (Weighted Links)
    np.random.seed(42)
    for i in range(len(layer_specs) - 1):
        c1 = layer_node_coords[i]
        c2 = layer_node_coords[i+1]
        spec_color = layer_specs[i]["color"]

        for (x1, y1) in c1:
            for (x2, y2) in c2:
                # Stochastic density filter for clean aesthetic
                if np.random.rand() > 0.38:
                    weight_strength = np.random.uniform(0.15, 0.85)
                    alpha = 0.12 * weight_strength
                    lw = 0.6 + 0.8 * weight_strength
                    # Smooth bezier curve
                    cx1 = x1 + (x2 - x1) * 0.5
                    cx2 = x1 + (x2 - x1) * 0.5
                    verts = [(x1, y1), (cx1, y1), (cx2, y2), (x2, y2)]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor="none", edgecolor=spec_color, alpha=alpha, linewidth=lw)
                    ax.add_patch(patch)

    # Draw Neurons (Nodes with Halo / Glow)
    for i, (spec, coords) in enumerate(zip(layer_specs, layer_node_coords)):
        color = spec["color"]
        for (x, y) in coords:
            # Outer glow
            glow = plt.Circle((x, y), 0.26, color=color, alpha=0.25, zorder=3)
            ax.add_patch(glow)
            # Inner core
            core = plt.Circle((x, y), 0.15, facecolor="#0f172a", edgecolor=color, linewidth=2.0, zorder=4)
            ax.add_patch(core)
            # Center bright spark
            spark = plt.Circle((x, y), 0.05, color="#ffffff", alpha=0.9, zorder=5)
            ax.add_patch(spark)

        # Draw Layer Header Card
        top_y = max_nodes * v_spacing + 0.4
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#111827", edgecolor="#1e293b", linewidth=1.2)
        ax.text(
            x, top_y, spec["name"],
            ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold",
            bbox=bbox_props
        )

    # Draw Output Diagnostic Class Labels with Probability Badges
    output_classes = [
        {"name": "Glioma", "prob": "94.2%", "color": "#f87171", "sub": "Intra-axial Infiltrative"},
        {"name": "Meningioma", "prob": "97.8%", "color": "#38bdf8", "sub": "Extra-axial Dural Tail"},
        {"name": "Pituitary Adenoma", "prob": "96.5%", "color": "#a855f7", "sub": "Sellar Suprasellar"},
        {"name": "Normal Tissue", "prob": "99.1%", "color": "#34d399", "sub": "Intact Parenchyma"}
    ]

    last_x = (len(layer_specs) - 1) * h_spacing
    for idx, ((x, y), out_info) in enumerate(zip(layer_node_coords[-1], output_classes)):
        # Arrow pointing out
        ax.annotate(
            "", xy=(x + 0.9, y), xytext=(x + 0.28, y),
            arrowprops=dict(arrowstyle="->", color=out_info["color"], lw=2.2, mutation_scale=15)
        )
        # Class Box
        badge_box = dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=out_info["color"], linewidth=1.5)
        text_str = f"{out_info['name']} ({out_info['prob']})\n{out_info['sub']}"
        ax.text(
            x + 1.05, y, text_str,
            va="center", ha="left", color="#f8fafc", fontsize=9, fontweight="bold",
            bbox=badge_box
        )

    # Add Receptive Field & Tensor Dimension Flow at the bottom
    tensor_flow = [
        "(B, 3, 240, 240)",
        "(B, 32, 238, 238)",
        "(B, 64, 60, 60)",
        "(B, 196, 768)",
        "(B, 512)",
        "(B, 128)",
        "(B, 4) Logits"
    ]

    for i, t_str in enumerate(tensor_flow):
        x = i * h_spacing
        ax.text(
            x, -0.6, t_str,
            ha="center", va="top", color="#64748b", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#0b0f1a", edgecolor="#1e293b", linewidth=0.8)
        )

    ax.text(
        (len(layer_specs) - 1) * h_spacing / 2.0, -1.2,
        "Forward Inference Pipeline: Raw Pixel Normalization ──► Convolutional Gating ──► Latent Dense Projection ──► Softmax Categorization",
        ha="center", va="top", color="#38bdf8", fontsize=10, fontweight="600"
    )

    ax.set_xlim(-1.2, last_x + 3.8)
    ax.set_ylim(-1.5, max_nodes * v_spacing + 1.5)

    out_path = "assets/comprehensive_neural_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Advanced Neural Network visualization successfully generated: {out_path}")

if __name__ == "__main__":
    draw_advanced_neural_network()
