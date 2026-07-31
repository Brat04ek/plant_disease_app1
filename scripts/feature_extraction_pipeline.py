import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse

plt.rcParams["font.family"] = "DejaVu Sans"


INPUT_IMAGE_BASENAME = "0ade14b6-8937-43ea-93eb-98343af6bae7___JR_HL 8026"


def find_input_image():
    extensions = ["png", "jpg", "jpeg", "JPG", "JPEG", "PNG"]
    for ext in extensions:
        path = f"{INPUT_IMAGE_BASENAME}.{ext}"
        if os.path.exists(path):
            return path

    matches = glob.glob(INPUT_IMAGE_BASENAME + "*")
    if matches:
        return matches[0]

    return None


def add_box(ax, x, y, w, h, face, edge, lw=1.5, radius=12, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, s, size=12, weight="normal", color="black",
             ha="center", va="center", z=5, linespacing=1.05):
    ax.text(
        x, y, s,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing
    )


def add_arrow(ax, x1, y1, x2, y2, color="black", lw=2.0, scale=16, z=4):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z
    )
    ax.add_patch(arr)
    return arr


def draw_histogram(ax, x, y, w, h, color, seed=1, n=16):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.25, 1.0, n)

    gap = w * 0.012
    bar_w = (w - gap * (n + 1)) / n

    for i, v in enumerate(values):
        bx = x + gap + i * (bar_w + gap)
        bh = (h - 8) * v
        by = y + h - bh - 4

        ax.add_patch(
            Rectangle(
                (bx, by),
                bar_w,
                bh,
                facecolor=color,
                edgecolor=color,
                alpha=0.75,
                zorder=5
            )
        )

    ax.plot([x, x + w], [y + h - 3, y + h - 3], color="black", lw=0.8, alpha=0.6, zorder=6)


def draw_input_photo(ax, x, y, w, h):
    image_path = find_input_image()

    if image_path is not None:
        img = plt.imread(image_path)

        # Якщо зображення 256x256, воно буде вставлене без обрізання,
        # просто масштабоване в межах прямокутника.
        ax.imshow(img, extent=(x, x + w, y + h, y), zorder=3)
    else:
        # Резервний варіант, якщо файл не знайдено
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#2d8f2d", edgecolor="none", zorder=3))
        add_text(ax, x + w / 2, y + h / 2, "image\nnot found", size=10, weight="bold", color="white")

    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", lw=1.0, zorder=4))


def draw_orb_icon(ax, x, y, w, h, edge):
    gx, gy, gs = x + 10, y + 10, 46

    for i in range(5):
        for j in range(5):
            shade = 0.74 - 0.04 * ((i + j) % 3)
            ax.add_patch(
                Rectangle(
                    (gx + i * gs / 5, gy + j * gs / 5),
                    gs / 5,
                    gs / 5,
                    facecolor=str(shade),
                    edgecolor="black",
                    lw=0.35,
                    zorder=5
                )
            )

    ax.add_patch(
        Circle(
            (gx + gs * 0.48, gy + gs * 0.50),
            3.2,
            facecolor="#2e7d32",
            edgecolor="black",
            lw=0.35,
            zorder=6
        )
    )

    add_arrow(ax, x + 72, y + h / 2, x + 102, y + h / 2, lw=1.5, scale=12)

    bx, by, bs = x + 112, y + 8, 54
    ax.add_patch(
        Rectangle(
            (bx, by), bs, bs,
            facecolor="#d9d9d9",
            edgecolor="black",
            lw=0.7,
            zorder=5
        )
    )

    rng = np.random.default_rng(3)
    for _ in range(8):
        px = bx + rng.uniform(7, bs - 7)
        py = by + rng.uniform(7, bs - 7)

        ax.add_patch(
            Circle(
                (px, py),
                3.2,
                facecolor="#43a047",
                edgecolor="black",
                lw=0.35,
                zorder=6
            )
        )
        ax.plot([bx + bs / 2, px], [by + bs / 2, py], color=edge, lw=0.7, alpha=0.8, zorder=5)


def draw_bovw_icon(ax, x, y, w, h):
    centers = [(x + 62, y + 36), (x + 122, y + 36), (x + 182, y + 36)]
    colors = ["#2f78bd", "#f28e2b", "#8e44ad"]
    rng = np.random.default_rng(5)

    for (cx, cy), c in zip(centers, colors):
        ax.add_patch(
            Ellipse(
                (cx, cy),
                48,
                58,
                fill=False,
                edgecolor="black",
                linestyle="--",
                lw=0.9,
                alpha=0.8,
                zorder=5
            )
        )

        for _ in range(8):
            px = cx + rng.normal(0, 10)
            py = cy + rng.normal(0, 12)

            ax.add_patch(
                Circle(
                    (px, py),
                    3.2,
                    facecolor=c,
                    edgecolor="white",
                    lw=0.45,
                    zorder=6
                )
            )


def draw_lbp_matrix(ax, x, y, cell=31):
    vals = [
        [1, 1, 1],
        [1, "", 1],
        [0, 1, 0]
    ]

    for r in range(3):
        for c in range(3):
            fc = "#f7f7f7" if vals[r][c] != "" else "#d0d0d0"

            ax.add_patch(
                Rectangle(
                    (x + c * cell, y + r * cell),
                    cell,
                    cell,
                    facecolor=fc,
                    edgecolor="black",
                    lw=0.9,
                    zorder=5
                )
            )

            add_text(ax, x + c * cell + cell / 2, y + r * cell + cell / 2,
                     str(vals[r][c]), size=11, weight="bold")


def draw_segmentation_mask(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="black", edgecolor="black", lw=0.8, zorder=5))

    blobs = [
        (x + 40, y + 45, 42, 25, -15),
        (x + 100, y + 55, 38, 32, 10),
        (x + 145, y + 28, 48, 40, 5),
        (x + 175, y + 60, 28, 16, 15),
        (x + 190, y + 25, 20, 13, -20),
    ]

    for bx, by, bw, bh, angle in blobs:
        ax.add_patch(Ellipse((bx, by), bw, bh, angle=angle, facecolor="white", edgecolor="white", zorder=6))


def draw_feature_vector(ax, x, y, w, h, colors):
    n = 26
    gap = 1.8
    cell_w = (w - 60 - gap * (n - 1)) / n
    cy = y + 48

    for i in range(n):
        if i < 5:
            fc = colors[0]
        elif i < 9:
            fc = colors[1]
        elif i < 13:
            fc = colors[2]
        elif i < 20:
            fc = colors[3]
        else:
            fc = colors[4]

        ax.add_patch(
            Rectangle(
                (x + 30 + i * (cell_w + gap), cy),
                cell_w,
                18,
                facecolor=fc,
                edgecolor="black",
                lw=0.5,
                alpha=0.45,
                zorder=5
            )
        )

    add_text(ax, x + w / 2, cy + 9, "...", size=10, weight="bold")


def draw_diagram(output_path="feature_extraction_pipeline_fixed.png"):
    fig, ax = plt.subplots(figsize=(22, 13))

    ax.set_xlim(0, 2300)
    ax.set_ylim(1300, 0)

    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 1900, 1180, facecolor="#fbfbfb", edgecolor="none", zorder=0))

    schemes = {
        "blue":   ("#eef6ff", "#2c5c9c", "#5aa1e3"),
        "yellow": ("#fff9e8", "#b8860b", "#f0c94d"),
        "green":  ("#f1faed", "#2e7d32", "#7bc96f"),
        "purple": ("#f7f1fb", "#6a4c93", "#b79ad6"),
        "orange": ("#fff2e8", "#d2691e", "#df7f32"),
        "red":    ("#fff1f1", "#d32f2f", "#ffb3b3"),
    }
    PAGE_CENTER_X = 1150
    # -------------------------
    # Верхній блок
    # -------------------------
    top_w = 500
    top_h = 180
    top_x = PAGE_CENTER_X - top_w / 2

    add_box(ax, top_x, 25, top_w, top_h, "#f3fbf0", "#1b5e20", lw=1.6, radius=12)

    add_text(ax, PAGE_CENTER_X, 58, "ВХІДНЕ ЗОБРАЖЕННЯ", size=20, weight="bold")
    add_text(ax, PAGE_CENTER_X, 90, "(листок рослини)", size=13, weight="bold")

    draw_input_photo(ax, PAGE_CENTER_X - 70, 105, 140, 80)
    # -------------------------
    # Розгалуження
    # -------------------------
    
    input_center_x = PAGE_CENTER_X
    branch_y = 235

    xs = [80, 520, 960, 1400, 1840]
    col_w, col_y, col_h = 340, 275, 530
    vec_y, vec_h = 830, 55

    col_names = [
        ("ORB + BoVW", "(локальні ознаки)", "blue"),
        ("HSV-ознаки", "(колірний простір)", "yellow"),
        ("Статистичні колірні\nознаки", "", "green"),
        ("Площа ураження", "(disease area)", "purple"),
        ("LBP-ознаки", "(локальний бінарний\nшаблон)", "orange"),
    ]


    centers = [x + col_w / 2 for x in xs]

    add_arrow(ax, input_center_x, 205, input_center_x, branch_y, lw=2.0, scale=15)
    ax.plot([centers[0], centers[-1]], [branch_y, branch_y], color="black", lw=2.0, zorder=3)

    for cx in centers:
        add_arrow(ax, cx, branch_y, cx, col_y, lw=1.8, scale=12)

    # -------------------------
    # Основні колонки
    # -------------------------
    for i, (x, (main_title, sub_title, key)) in enumerate(zip(xs, col_names), start=1):
        face, edge, accent = schemes[key]

        add_box(ax, x, col_y, col_w, col_h, face, edge, lw=1.4, radius=13)

        add_text(ax, x + col_w / 2, col_y + 32, main_title, size=15, weight="bold", va="top", linespacing=1.0)

        if sub_title != "":
            add_text(ax, x + col_w / 2, col_y + 58, sub_title, size=11.5, weight="bold", va="top", linespacing=1.0)

        add_box(ax, x + 25, vec_y, col_w - 50, vec_h, face, edge, lw=1.3, radius=7)
        add_text(ax, x + col_w / 2, vec_y + vec_h / 2, f"Вектор ознак {i}", size=15, weight="bold")

        add_arrow(ax, x + col_w / 2, col_y + col_h, x + col_w / 2, vec_y, lw=1.8, scale=14)

    # -------------------------
    # Колонка 1: ORB + BoVW
    # -------------------------
    x = xs[0]
    face, edge, accent = schemes["blue"]

    sub_y = [360, 505, 650]
    sub_h = 125

    labels = [
        "ORB (FAST +\nдескриптори)",
        "Bag of Visual Words\n(візуальний словник)",
        "Гістограма частот\n(BoVW-вектор)"
    ]

    for y, lab in zip(sub_y, labels):
        add_box(ax, x + 15, y, col_w - 30, sub_h, "#ffffff", edge, lw=0.9, radius=8)
        add_text(ax, x + col_w / 2, y + 25, lab, size=12, weight="bold", linespacing=1.0)

    draw_orb_icon(ax, x + 40, sub_y[0] + 52, col_w - 80, 56, edge)
    draw_bovw_icon(ax, x + 34, sub_y[1] + 52, col_w - 68, 56)
    draw_histogram(ax, x + 32, sub_y[2] + 62, col_w - 64, 50, "#2f78bd", seed=8, n=18)

    add_arrow(ax, x + col_w / 2, sub_y[0] + sub_h, x + col_w / 2, sub_y[1], lw=1.5, scale=12)
    add_arrow(ax, x + col_w / 2, sub_y[1] + sub_h, x + col_w / 2, sub_y[2], lw=1.5, scale=12)

    # -------------------------
    # Колонка 2: HSV
    # -------------------------
    x = xs[1]
    face, edge, accent = schemes["yellow"]

    hsv_titles = ["H (Hue)", "S (Saturation)", "V (Value)"]
    hsv_colors = ["#ff7f2a", "#8bc34a", "#6d9dc5"]

    for k, (title, c) in enumerate(zip(hsv_titles, hsv_colors)):
        y = 370 + k * 135
        add_box(ax, x + 18, y, col_w - 36, 105, "#ffffff", edge, lw=0.9, radius=8)
        add_text(ax, x + col_w / 2, y + 22, title, size=12, weight="bold")

        if k == 0:
            bar_colors = [
                "#ff6d2d", "#ff9f1c", "#ffd166", "#e8ed4f",
                "#9bd770", "#43aa8b", "#4cc9f0", "#4895ef",
                "#5e60ce", "#9d4edd", "#ff7f2a"
            ]

            vals = np.array([0.9, 0.68, 0.62, 0.55, 0.47, 0.66, 0.82, 0.78, 0.68, 0.60, 0.55, 0.70])
            n = len(vals)

            bx0, by0, bw, bh = x + 34, y + 45, col_w - 68, 50
            gap = 2
            bar_w = (bw - gap * (n - 1)) / n

            for j, v in enumerate(vals):
                color = bar_colors[j % len(bar_colors)]
                ax.add_patch(
                    Rectangle(
                        (bx0 + j * (bar_w + gap), by0 + bh * (1 - v)),
                        bar_w,
                        bh * v,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.8,
                        zorder=5
                    )
                )

            ax.plot([bx0, bx0 + bw], [by0 + bh, by0 + bh], color="black", lw=0.8, alpha=0.5)

        else:
            draw_histogram(ax, x + 34, y + 45, col_w - 68, 50, c, seed=15 + k, n=14)

    # -------------------------
    # Колонка 3: статистичні ознаки
    # -------------------------
    x = xs[2]

    stats = [
        "Середнє значення\n(mean)",
        "Стандартне\nвідхилення",
        "Дисперсія",
        "Ентропія",
        "Асиметрія",
        "Ексцес",
        "Контраст",
        "Енергія",
    ]

    ty = col_y + 120

    for j, item in enumerate(stats):
        yy = ty + j * 48
        add_text(ax, x + 25, yy, "•", size=17, weight="bold", ha="left")
        add_text(ax, x + 65, yy, item, size=13, weight="bold", ha="left", linespacing=0.95)

    # -------------------------
    # Колонка 4: площа ураження
    # -------------------------
    x = xs[3]
    face, edge, accent = schemes["purple"]

    add_box(ax, x + 18, 365, col_w - 36, 205, "#ffffff", edge, lw=0.9, radius=8)
    add_text(ax, x + col_w / 2, 410, "Сегментація\nуражених ділянок", size=13, weight="bold", linespacing=1.0)

    draw_segmentation_mask(ax, x + 42, 455, 200, 90)

    add_box(ax, x + 18, 610, col_w - 36, 130, "#ffffff", edge, lw=0.9, radius=8)

    add_text(
        ax,
        x + col_w / 2,
        675,
        "Обчислення площі\nуражених ділянок\n(у % від площі листка)",
        size=12,
        weight="bold",
        linespacing=1.0
    )

    add_arrow(ax, x + col_w / 2, 570, x + col_w / 2, 610, lw=1.5, scale=12)

    # -------------------------
    # Колонка 5: LBP
    # -------------------------
    x = xs[4]
    face, edge, accent = schemes["orange"]

    add_box(ax, x + 18, 380, col_w - 36, 175, "#ffffff", edge, lw=0.9, radius=8)
    add_text(ax, x + col_w / 2, 420, "Обчислення LBP\nдля зображення", size=13, weight="bold", linespacing=1.0)

    draw_lbp_matrix(ax, x + 96, 455, cell=31)

    add_box(ax, x + 18, 595, col_w - 36, 145, "#ffffff", edge, lw=0.9, radius=8)
    add_text(ax, x + col_w / 2, 635, "Гістограма\nLBP-кодів", size=13, weight="bold", linespacing=1.0)

    draw_histogram(ax, x + 40, 675, col_w - 80, 50, "#b9652b", seed=22, n=20)

    add_arrow(ax, x + col_w / 2, 555, x + col_w / 2, 595, lw=1.5, scale=12)

    # -------------------------
    # Нижнє об'єднання векторів
    # -------------------------
    bus_y = 925

    for cx in centers:
        add_arrow(ax, cx, vec_y + vec_h, cx, bus_y, lw=1.8, scale=12)

    ax.plot([centers[0], centers[-1]], [bus_y, bus_y], color="black", lw=2.0, zorder=3)
    
    comb_w, comb_h = 1050, 95
    comb_x, comb_y = PAGE_CENTER_X - comb_w / 2, 965

    add_arrow(ax, PAGE_CENTER_X, bus_y, PAGE_CENTER_X, comb_y, lw=2.0, scale=15)

    add_box(ax, comb_x, comb_y, comb_w, comb_h, "#fff7df", "#b8860b", lw=1.3, radius=8)
    add_text(ax, PAGE_CENTER_X, comb_y + 30, "ОБ’ЄДНАНИЙ ВЕКТОР ОЗНАК", size=17, weight="bold")

    draw_feature_vector(
        ax,
        comb_x + 10,
        comb_y + 15,
        comb_w - 20,
        comb_h - 20,
        [
            schemes["blue"][2],
            schemes["yellow"][2],
            schemes["green"][2],
            schemes["purple"][2],
            schemes["orange"][2],
        ]
    )

    # -------------------------
    # Класифікатор
    # -------------------------
    clf_w, clf_h = 520, 120
    clf_x, clf_y = PAGE_CENTER_X - clf_w / 2, 1090

    add_arrow(ax, PAGE_CENTER_X, comb_y + comb_h, PAGE_CENTER_X, clf_y, lw=2.0, scale=15)

    add_box(ax, clf_x, clf_y, clf_w, clf_h, "#fff1f1", "#d32f2f", lw=1.3, radius=8)

    add_text(ax, PAGE_CENTER_X, clf_y + 22, "КЛАСИФІКАТОР", size=17, weight="bold", va="top")
    add_text(ax, PAGE_CENTER_X, clf_y + 52, "(SVM / Random Forest / інший)", size=11.5, weight="bold", va="top")
    add_text(ax, PAGE_CENTER_X, clf_y + 82, "→ Клас хвороби рослини", size=13, weight="bold", va="top")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    draw_diagram("feature_extraction_pipeline_fixed.png")
    print("Saved: feature_extraction_pipeline_fixed.png")