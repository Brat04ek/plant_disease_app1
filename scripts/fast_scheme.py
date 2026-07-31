import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, ConnectionPatch


def create_synthetic_left_image(size=120):
    """
    Створює умовне сіре зображення для лівої частини рисунка.
    За потреби цю функцію можна замінити читанням реального зображення.
    """
    y, x = np.mgrid[0:size, 0:size]

    img = 0.55 + 0.25 * (x / size) - 0.15 * (y / size)

    curve = 0.55 * size + 0.18 * size * np.sin((y / size) * 2.5 * np.pi)
    mask = np.abs(x - curve) < 7
    img[mask] -= 0.35

    mask2 = np.abs(x - curve - 12) < 4
    img[mask2] += 0.15

    rng = np.random.default_rng(4)
    img += rng.normal(0, 0.035, img.shape)

    return np.clip(img, 0, 1)


def label_position_for_fast_pixel(idx):
    """
    Позиції номерів навколо 16 пікселів FAST.
    """
    if idx in [16, 1, 2]:
        return 0, -0.82, "center", "bottom"       # зверху
    elif idx == 3:
        return 0.70, -0.70, "left", "bottom"     # зверху справа
    elif idx in [4, 5, 6]:
        return 0.82, 0, "left", "center"         # справа
    elif idx == 7:
        return 0.70, 0.70, "left", "top"         # знизу справа
    elif idx in [8, 9, 10]:
        return 0, 0.82, "center", "top"          # знизу
    elif idx == 11:
        return -0.70, 0.70, "right", "top"       # знизу зліва
    elif idx in [12, 13, 14]:
        return -0.82, 0, "right", "center"       # зліва
    elif idx == 15:
        return -0.70, -0.70, "right", "bottom"   # зверху зліва

    return 0, 0, "center", "center"


def draw_fast_like_scheme(output_path="fast_like_scheme_improved.png"):
    fig = plt.figure(figsize=(10, 4.8))

    # -------------------------
    # Ліва частина
    # -------------------------
    ax_left = fig.add_axes([0.05, 0.20, 0.32, 0.60])
    ax_left.set_aspect("equal")
    ax_left.axis("off")

    img = create_synthetic_left_image()
    ax_left.imshow(img, cmap="gray", vmin=0, vmax=1)

    # Важливо: розмір ROI = 15, щоб він напряму відповідав сітці 15x15 справа
    roi_x, roi_y = 56, 48
    roi_size = 15

    roi_rect = Rectangle(
        (roi_x - 0.5, roi_y - 0.5),
        roi_size,
        roi_size,
        fill=False,
        edgecolor="white",
        linewidth=1.8,
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")]
    )
    ax_left.add_patch(roi_rect)

    # -------------------------
    # Права частина: zoom
    # -------------------------
    ax_grid = fig.add_axes([0.47, 0.08, 0.48, 0.84])
    ax_grid.set_aspect("equal")
    ax_grid.axis("off")

    n = 15
    center = n // 2

    # Тепер zoom справа реально береться з області зліва,
    # тому центральна частина відповідає фрагменту зображення
    grid_img = img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size].copy()

    # Легке підсилення контрасту без штучного затемнення центру
    grid_img = (grid_img - grid_img.min()) / (grid_img.max() - grid_img.min() + 1e-9)
    grid_img = 0.15 + 0.75 * grid_img

    ax_grid.imshow(grid_img, cmap="gray", vmin=0, vmax=1, origin="upper")

    # Сітка
    for i in range(n + 1):
        ax_grid.plot(
            [i - 0.5, i - 0.5],
            [-0.5, n - 0.5],
            color="white",
            linewidth=0.9,
            alpha=0.75,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="black", alpha=0.35)]
        )
        ax_grid.plot(
            [-0.5, n - 0.5],
            [i - 0.5, i - 0.5],
            color="white",
            linewidth=0.9,
            alpha=0.75,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="black", alpha=0.35)]
        )

    # Центральний піксель p
    ax_grid.text(
        center,
        center,
        "p",
        color="white",
        fontsize=16,
        ha="center",
        va="center",
        fontstyle="italic",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")]
    )

    # Пікселі кола FAST-16
    circle_points = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]

    fast_pixels = [(center + dx, center + dy) for dx, dy in circle_points]

    # Контур навколо FAST-пікселів:
    # чорний товстий контур + білий пунктир поверх нього
    polyline = fast_pixels + [fast_pixels[0]]

    ax_grid.plot(
        [p[0] for p in polyline],
        [p[1] for p in polyline],
        color="black",
        linewidth=4.2,
        linestyle="--",
        alpha=0.95
    )

    ax_grid.plot(
        [p[0] for p in polyline],
        [p[1] for p in polyline],
        color="white",
        linewidth=2.2,
        linestyle="--"
    )

    # Виділення кожного з 16 пікселів
    for idx, (px, py) in enumerate(fast_pixels, start=1):
        # Зовнішній чорний контур
        outer_rect = Rectangle(
            (px - 0.5, py - 0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=3.2
        )
        ax_grid.add_patch(outer_rect)

        # Внутрішній білий контур
        inner_rect = Rectangle(
            (px - 0.5, py - 0.5),
            1,
            1,
            fill=False,
            edgecolor="white",
            linewidth=1.6
        )
        ax_grid.add_patch(inner_rect)

        # Номер пікселя
        dx, dy, ha, va = label_position_for_fast_pixel(idx)

        ax_grid.text(
            px + dx,
            py + dy,
            str(idx),
            color="white",
            fontsize=9,
            ha=ha,
            va=va,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="black")]
        )

    ax_grid.set_xlim(-0.5, n - 0.5)
    ax_grid.set_ylim(n - 0.5, -0.5)

    # -------------------------
    # Лінії збільшення
    # -------------------------
    con1 = ConnectionPatch(
        xyA=(roi_x + roi_size - 0.5, roi_y - 0.5),
        coordsA=ax_left.transData,
        xyB=(-0.5, -0.5),
        coordsB=ax_grid.transData,
        color="gray",
        linewidth=1.2
    )

    con2 = ConnectionPatch(
        xyA=(roi_x + roi_size - 0.5, roi_y + roi_size - 0.5),
        coordsA=ax_left.transData,
        xyB=(-0.5, n - 0.5),
        coordsB=ax_grid.transData,
        color="gray",
        linewidth=1.2
    )

    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()


draw_fast_like_scheme("fast_like_scheme_improved.png")