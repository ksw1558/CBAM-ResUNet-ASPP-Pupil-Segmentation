from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("teacher_report_assets/model_architecture")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = OUT_DIR / "cbam_resunet_aspp_architecture"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 0.9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


COLORS = {
    "input": "#F7F7F5",
    "encoder": "#E1EDF7",
    "bottleneck": "#F0EAF8",
    "aspp": "#E5F3EC",
    "cbam": "#F7E9CF",
    "decoder": "#F7E5E1",
    "output": "#E8ECEF",
    "border": "#475569",
    "encoder_border": "#2F6690",
    "aspp_border": "#3D7A61",
    "cbam_border": "#A96A16",
    "decoder_border": "#A34F48",
    "skip": "#5D7185",
    "arrow": "#24303F",
    "text": "#111827",
    "muted": "#536273",
}


def rounded_box(ax, xy, w, h, text, fc, ec, fs=7.4, lw=1.0, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=COLORS["text"],
        weight=weight,
        linespacing=1.16,
    )


def arrow(ax, p1, p2, *, color=None, ls="-", lw=1.15, ms=9, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=color or COLORS["arrow"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_aspp_detail(ax):
    x0, y0, w, h = 3.18, 0.38, 4.16, 1.10
    rounded_box(ax, (x0, y0), w, h, "", "#FFFFFF", "#AAB4BF", fs=7.0, lw=0.9)
    ax.text(x0 + 0.16, y0 + h - 0.19, "ASPP detail", fontsize=7.7, weight="bold", ha="left", va="center")

    branches = ["1x1\nConv", "3x3\nr=6", "3x3\nr=12", "3x3\nr=18", "Image\nPooling"]
    for i, label in enumerate(branches):
        bx = x0 + 0.20 + i * 0.76
        rounded_box(ax, (bx, y0 + 0.40), 0.58, 0.38, label, COLORS["aspp"], COLORS["aspp_border"], fs=5.8)

    ax.text(
        x0 + w / 2,
        y0 + 0.15,
        "Concatenation -> 1x1 projection (2048 -> 512)",
        fontsize=6.1,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )


def draw_cbam_detail(ax):
    x0, y0, w, h = 7.68, 0.38, 3.40, 1.10
    rounded_box(ax, (x0, y0), w, h, "", "#FFFFFF", "#AAB4BF", fs=7.0, lw=0.9)
    ax.text(x0 + 0.16, y0 + h - 0.19, "CBAM detail", fontsize=7.7, weight="bold", ha="left", va="center")
    rounded_box(
        ax,
        (x0 + 0.24, y0 + 0.39),
        1.05,
        0.38,
        "Channel\nAttention",
        COLORS["cbam"],
        COLORS["cbam_border"],
        fs=6.0,
    )
    rounded_box(
        ax,
        (x0 + 2.08, y0 + 0.39),
        1.05,
        0.38,
        "Spatial\nAttention",
        COLORS["cbam"],
        COLORS["cbam_border"],
        fs=6.0,
    )
    arrow(ax, (x0 + 1.31, y0 + 0.58), (x0 + 2.06, y0 + 0.58), lw=0.9, ms=7)
    ax.text(
        x0 + w / 2,
        y0 + 0.15,
        "Channel reweighting -> spatial refinement",
        fontsize=5.9,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )


def main():
    fig, ax = plt.subplots(figsize=(14.4, 8.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0.0, 8.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        6.6,
        8.48,
        "Architecture of the Proposed CBAM-ResUNet-ASPP Model",
        fontsize=15,
        weight="bold",
        ha="center",
        va="center",
    )

    enc_x, enc_w, block_h = 0.58, 2.28, 0.63
    enc_y = [7.30, 6.42, 5.54, 4.66, 3.78, 2.90, 2.02]
    encoder = [
        ("Input image\n3 x 320 x 320", COLORS["input"], COLORS["border"]),
        ("Layer0\nConv 7x7 + BN + ReLU\n64 ch, 160 x 160", COLORS["encoder"], COLORS["encoder_border"]),
        ("MaxPool\n3x3, stride=2\n64 ch, 80 x 80", COLORS["encoder"], COLORS["encoder_border"]),
        ("Layer1\nResNet Bottleneck x3, stride=1\n256 ch, 80 x 80", COLORS["encoder"], COLORS["encoder_border"]),
        ("Layer2\nResNet Bottleneck x4, stride=2\n512 ch, 40 x 40", COLORS["encoder"], COLORS["encoder_border"]),
        ("Layer3\nResNet Bottleneck x6, stride=2\n1024 ch, 20 x 20", COLORS["encoder"], COLORS["encoder_border"]),
        ("Layer4 (standard)\nResNet Bottleneck x3, stride=2\n2048 ch, 10 x 10", COLORS["encoder"], COLORS["encoder_border"]),
    ]
    for y, (label, fc, ec) in zip(enc_y, encoder):
        rounded_box(ax, (enc_x, y), enc_w, block_h, label, fc, ec, fs=7.25, weight="bold" if "Input" in label else "normal")
    for y1, y2 in zip(enc_y, enc_y[1:]):
        arrow(ax, (enc_x + enc_w / 2, y1), (enc_x + enc_w / 2, y2 + block_h), lw=1.1)

    ax.text(
        enc_x + enc_w / 2,
        8.02,
        "Encoder (ResNet50 Backbone)",
        fontsize=8.8,
        weight="bold",
        ha="center",
        va="center",
        color=COLORS["encoder_border"],
    )
    bottleneck_y = 2.02
    rounded_box(
        ax,
        (3.58, bottleneck_y),
        1.84,
        block_h,
        "ASPP\n2048 -> 512",
        COLORS["aspp"],
        COLORS["aspp_border"],
        fs=7.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (5.72, bottleneck_y),
        1.64,
        block_h,
        "CBAM\n512 ch",
        COLORS["cbam"],
        COLORS["cbam_border"],
        fs=7.5,
        weight="bold",
    )
    arrow(ax, (enc_x + enc_w, bottleneck_y + block_h / 2), (3.58, bottleneck_y + block_h / 2), lw=1.2)
    arrow(ax, (5.42, bottleneck_y + block_h / 2), (5.72, bottleneck_y + block_h / 2), lw=1.2)
    ax.text(
        5.46,
        1.80,
        "Bottleneck (ASPP + CBAM)",
        fontsize=8.8,
        weight="bold",
        ha="center",
        va="center",
        color="#6A4B91",
    )

    dec_x, dec_w = 8.46, 3.30
    dec_y = [2.02, 3.18, 4.34, 5.50, 6.66]
    decoder = [
        (
            "Up4*\nUpsample x2 + Concatenation\nfeat3[1024] + bottleneck[512] = 1536\nResCBAMBlock -> 512 ch, 20 x 20",
            COLORS["decoder"],
            COLORS["decoder_border"],
        ),
        (
            "Up3*\nUpsample x2 + Concatenation\nfeat2[512] + up4[512] = 1024\nResCBAMBlock -> 256 ch, 40 x 40",
            COLORS["decoder"],
            COLORS["decoder_border"],
        ),
        (
            "Up2\nUpsample x2 + Concatenation\nfeat1[256] + up3[256] = 512\nResBlock -> 128 ch, 80 x 80",
            COLORS["decoder"],
            COLORS["decoder_border"],
        ),
        (
            "Up1\nUpsample x2 + Conv 3x3 + ReLU\n64 ch, 160 x 160\nNo residual / no skip connection",
            COLORS["decoder"],
            COLORS["decoder_border"],
        ),
        (
            "Prediction Head\n1x1 Conv -> 2 x 160 x 160 logits\nBinary segmentation output",
            COLORS["output"],
            COLORS["border"],
        ),
    ]
    for y, (label, fc, ec) in zip(dec_y, decoder):
        rounded_box(ax, (dec_x, y), dec_w, 0.83, label, fc, ec, fs=6.95, weight="bold" if "Prediction" in label else "normal")
    arrow(ax, (7.36, bottleneck_y + block_h / 2), (dec_x, dec_y[0] + 0.42), lw=1.2)
    for y1, y2 in zip(dec_y, dec_y[1:]):
        arrow(ax, (dec_x + dec_w / 2, y1 + 0.83), (dec_x + dec_w / 2, y2), lw=1.12)

    skip_specs = [
        ((enc_x + enc_w, enc_y[5] + block_h / 2), (dec_x, dec_y[0] + 0.42)),
        ((enc_x + enc_w, enc_y[4] + block_h / 2), (dec_x, dec_y[1] + 0.42)),
        ((enc_x + enc_w, enc_y[3] + block_h / 2), (dec_x, dec_y[2] + 0.42)),
    ]
    for p1, p2 in skip_specs:
        arrow(ax, p1, p2, color=COLORS["skip"], ls="--", lw=1.0, ms=8, rad=-0.05)

    ax.text(
        dec_x + dec_w / 2,
        8.02,
        "Decoder",
        fontsize=8.8,
        weight="bold",
        ha="center",
        va="center",
        color=COLORS["decoder_border"],
    )
    ax.text(
        dec_x + dec_w / 2,
        7.76,
        "Up4* -> Up3* -> Up2 -> Up1",
        fontsize=6.7,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    ax.plot([0.58, 12.34], [1.62, 1.62], color="#D0D7DE", lw=0.8)
    draw_aspp_detail(ax)
    draw_cbam_detail(ax)
    ax.text(
        0.62,
        0.92,
        "Data Flow",
        fontsize=6.7,
        ha="left",
        va="center",
        color=COLORS["muted"],
    )
    arrow(ax, (1.45, 0.92), (2.10, 0.92), lw=1.0, ms=8)
    ax.text(
        0.62,
        0.55,
        "Skip Connection",
        fontsize=6.7,
        ha="left",
        va="center",
        color=COLORS["muted"],
    )
    arrow(ax, (1.72, 0.55), (2.37, 0.55), color=COLORS["skip"], ls="--", lw=1.0, ms=8)

    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.png", dpi=600, bbox_inches="tight", metadata={"dpi": "600"})
    fig.savefig(f"{OUT_BASE}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(OUT_BASE.resolve())


if __name__ == "__main__":
    main()
