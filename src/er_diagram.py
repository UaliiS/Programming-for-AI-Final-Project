"""
Draws the ER diagram of the SQLite schema (outputs/plots/er_diagram.png) from the
schema description below, so the picture always matches src/database.py.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import ER_DIAGRAM_PNG

# table -> list of (column, type, key)  key in {"PK", "FK", "PK,FK", ""}
TABLES: Dict[str, List[Tuple[str, str, str]]] = {
    "users": [
        ("user_id", "TEXT", "PK"), ("name", "TEXT", ""), ("review_count", "INTEGER", ""),
        ("yelping_since", "TEXT", ""), ("useful / funny / cool", "INTEGER", ""),
        ("is_elite", "INTEGER", ""), ("n_elite_years", "INTEGER", ""), ("elite_years", "TEXT", ""),
        ("fans", "INTEGER", ""), ("average_stars", "REAL", ""),
    ],
    "reviews": [
        ("review_id", "TEXT", "PK"), ("user_id", "TEXT", "FK"), ("business_id", "TEXT", "FK"),
        ("stars", "INTEGER", ""), ("useful / funny / cool", "INTEGER", ""), ("text_length", "INTEGER", ""),
        ("date", "TEXT", ""), ("year", "INTEGER", ""), ("month", "INTEGER", ""),
    ],
    "businesses": [
        ("business_id", "TEXT", "PK"), ("name", "TEXT", ""), ("city", "TEXT", ""), ("state", "TEXT", ""),
        ("latitude / longitude", "REAL", ""), ("stars", "REAL", ""), ("is_open", "INTEGER", ""),
        ("price_range", "INTEGER", ""), ("primary_category", "TEXT", ""),
    ],
    "business_categories": [("business_id", "TEXT", "PK,FK"), ("category_id", "INTEGER", "PK,FK")],
    "categories": [("category_id", "INTEGER", "PK"), ("category_name", "TEXT", "")],
}

POSITIONS = {  # x, y of the top-left corner (axes units)
    "users": (0.2, 9.2), "reviews": (4.6, 9.2), "businesses": (9.0, 9.2),
    "business_categories": (9.0, 3.6), "categories": (4.6, 3.6),
}
# (from_table, from_col, to_table, to_col, label)
RELATIONS = [
    ("users", "user_id", "reviews", "user_id", "1 : N"),
    ("businesses", "business_id", "reviews", "business_id", "1 : N"),
    ("businesses", "business_id", "business_categories", "business_id", "1 : N"),
    ("categories", "category_id", "business_categories", "category_id", "1 : N"),
]

BOX_W = 3.6
ROW_H = 0.42
HEAD_H = 0.55


def _row_y(table: str, col: str) -> float:
    x, y = POSITIONS[table]
    names = [c[0] for c in TABLES[table]]
    i = names.index(col)
    return y - HEAD_H - ROW_H * (i + 0.5)


def draw_er_diagram(path: Path = ER_DIAGRAM_PNG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 10)
    ax.axis("off")

    for table, cols in TABLES.items():
        x, y = POSITIONS[table]
        h = HEAD_H + ROW_H * len(cols)
        ax.add_patch(FancyBboxPatch((x, y - h), BOX_W, h, boxstyle="round,pad=0.02",
                                    fc="white", ec="#2b3a55", lw=1.4))
        ax.add_patch(FancyBboxPatch((x, y - HEAD_H), BOX_W, HEAD_H, boxstyle="round,pad=0.02",
                                    fc="#2b3a55", ec="#2b3a55", lw=1.4))
        ax.text(x + BOX_W / 2, y - HEAD_H / 2, table, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        for i, (name, typ, key) in enumerate(cols):
            ry = y - HEAD_H - ROW_H * (i + 0.5)
            weight = "bold" if "PK" in key else "normal"
            ax.text(x + 0.12, ry, key, ha="left", va="center", fontsize=8.5,
                    color="#b5462f" if "FK" in key else "#2b3a55", fontweight="bold")
            ax.text(x + 0.75, ry, name, ha="left", va="center", fontsize=9.5, fontweight=weight)
            ax.text(x + BOX_W - 0.12, ry, typ, ha="right", va="center", fontsize=8.5, color="#666666")

    for t1, c1, t2, c2, label in RELATIONS:
        x1, _ = POSITIONS[t1]
        x2, _ = POSITIONS[t2]
        y1, y2 = _row_y(t1, c1), _row_y(t2, c2)
        if x1 < x2:
            sx, ex = x1 + BOX_W, x2
        elif x1 > x2:
            sx, ex = x1, x2 + BOX_W
        else:  # same column (businesses -> business_categories): route along the right side
            sx, ex = x1 + BOX_W, x2 + BOX_W
        if x1 == x2:
            ax.annotate("", xy=(ex, y2), xytext=(sx, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3,
                                        connectionstyle="arc3,rad=-0.35"))
            ax.text(sx + 0.75, (y1 + y2) / 2, label, fontsize=9, color="#555555")
        else:
            ax.annotate("", xy=(ex, y2), xytext=(sx, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3,
                                        connectionstyle="arc3,rad=0.0"))
            ax.text((sx + ex) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=9, color="#555555")

    ax.text(0.2, 1.2, "Arrow = one-to-many (points to the 'many' side). PK = primary key, FK = foreign key.\n"
            "business_categories resolves the many-to-many relation between businesses and categories.\n"
            "Extra table agg_city_category_year (no keys) stores pre-computed aggregates for the dashboard.",
            fontsize=9.5, color="#333333", va="bottom")
    ax.set_title("ER diagram — SQLite database data/db/yelp_analytics.db", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw_er_diagram())
