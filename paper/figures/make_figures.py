import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
# nDCG@10 on full CUREv1 (English, Spanish, French). The learned gate is not
# shown: it is not comparable with the ordinary ranking pipeline (see text).
NDCG = {
    "BM25": (0.3098, 0.0158, 0.0283),
    "Dense\n(BGE-M3)": (0.5390, 0.4911, 0.4889),
    "Late\n(Jina)": (0.5766, 0.5442, 0.5158),
    "Fusion-\nonly": (0.5811, 0.5422, 0.5319),
    "Late +\nattribute": (0.5760, 0.5441, 0.5156),
    "ClinAttr-\nRule": (0.5805, 0.5419, 0.5317),
}

# Pair consistency: (A_o, A_c, balanced, PC, PC_low, PC_high) per system and domain.
# The interval is the exact (Clopper-Pearson) 95% interval, scipy.stats.beta.ppf(.025, k, n-k+1) and
# beta.ppf(.975, k+1, n-k) with k = round(PC * n); it replaces the earlier bootstrap interval.
SYM = {
    "Orthopedic Surgery": {"n": 29, "rows": {
        "Dense": (0.414, 0.724, 0.569, 0.172, 0.058, 0.358),
        "Late": (0.379, 0.483, 0.431, 0.103, 0.022, 0.274),
        "Rule": (0.414, 0.621, 0.517, 0.138, 0.039, 0.317),
        "Gate": (0.414, 0.621, 0.517, 0.138, 0.039, 0.317)}},
    "Pulmonology": {"n": 122, "rows": {
        "Dense": (0.516, 0.566, 0.541, 0.246, 0.172, 0.332),
        "Late": (0.434, 0.443, 0.439, 0.189, 0.123, 0.269),
        "Rule": (0.582, 0.516, 0.549, 0.279, 0.201, 0.367),
        "Gate": (0.582, 0.525, 0.553, 0.279, 0.201, 0.367)}},
    "Psychiatry and Psychology": {"n": 113, "rows": {
        "Dense": (0.504, 0.584, 0.544, 0.204, 0.134, 0.290),
        "Late": (0.363, 0.513, 0.438, 0.142, 0.083, 0.220),
        "Rule": (0.478, 0.646, 0.562, 0.239, 0.164, 0.328),
        "Gate": (0.478, 0.637, 0.558, 0.230, 0.156, 0.319)}},
}


# ----------------------------------------------------------------------------
# Figures are drawn at the final print width of the Springer LNCS text block
# (4.8 in), so fonts stay legible when included with width=\textwidth.
# ----------------------------------------------------------------------------
W = 4.8


# ----------------------------------------------------------------------------
# Figure 1 (revised): ordinary retrieval effectiveness
# ----------------------------------------------------------------------------
def fig_ndcg(path="fig_ndcg_revised.pdf"):
    langs = ["English", "Spanish", "French"]
    shades = ["#2F6FA3", "#E0922F", "#4E9A51"]
    hatches = ["", "//", ".."]
    methods = list(NDCG)
    width = 0.26
    fig, ax = plt.subplots(figsize=(W, 2.7))
    for j, lang in enumerate(langs):
        xs = [i + (j - 1) * width for i in range(len(methods))]
        vals = [NDCG[m][j] for m in methods]
        ax.bar(xs, vals, width, label=lang, color=shades[j], edgecolor="black",
               linewidth=0.5, hatch=hatches[j])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=6.5)
    ax.set_ylabel("nDCG@10", fontsize=7)
    ax.set_ylim(0, 0.72)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Colour scheme shared by all figures (colour-blind-safe hues)
# ----------------------------------------------------------------------------
C_DATA = ("#DCEBF7", "#2F6FA3")    # data and resources
C_LLM = ("#FCE5CC", "#C77418")     # language-model stages
C_DET = ("#DDEFD9", "#3E8E41")     # deterministic steps
C_EVAL = ("#E8DDF2", "#7A4EA3")    # evaluation
C_BAD = "#B83227"
SYS_COL = {"Dense": "#2F6FA3", "Late": "#C77418", "Rule": "#3E8E41", "Gate": "#7A4EA3"}


# ----------------------------------------------------------------------------
# Pair consistency figure
# ----------------------------------------------------------------------------
def fig_pair_consistency(path="fig_pair_consistency.pdf"):
    methods = [("Dense", "o"), ("Late", "s"), ("Rule", "^"), ("Gate", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), sharey=True)
    for ax, (dom, v) in zip(axes, SYM.items()):
        ax.axhspan(0, 0.25, color="#F3F3F3", zorder=0)
        for i, (key, mk) in enumerate(methods):
            r = v["rows"][key]
            pc, lo, hi = r[3], r[4], r[5]
            col = SYS_COL[key]
            ax.plot([i - 0.28, i + 0.28], [r[0] * r[1]] * 2, color="black", linewidth=1.6,
                    solid_capstyle="butt", zorder=2)
            ax.errorbar(i, pc, yerr=[[pc - lo], [hi - pc]], fmt=mk, color=col, ecolor=col,
                        capsize=3, markersize=5.2, linewidth=1.3, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3)
        ax.axhline(0.25, linestyle="--", color=C_BAD, linewidth=1.0, zorder=1)
        ax.set_title(f"{dom.replace('Psychiatry and Psychology', 'Psychiatry and Psych.')}\n(n = {v['n']})",
                     fontsize=6.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([m[0] for m in methods], fontsize=6.3)
        for lab, (key, _) in zip(ax.get_xticklabels(), methods):
            lab.set_color(SYS_COL[key])
            lab.set_fontweight("bold")
        ax.tick_params(axis="y", labelsize=6.3)
        ax.set_ylim(0, 0.45)
        ax.set_xlim(-0.6, 3.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pair consistency", fontsize=7)
    handles = [
        Line2D([0], [0], marker="o", color="#555555", linestyle="-", markersize=4,
               label="Observed, exact 95% interval"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Product of directional accuracies"),
        Line2D([0], [0], linestyle="--", color=C_BAD, linewidth=1.0, label="Coin flip (0.25)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.0, frameon=False,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pipeline figure (drawn in inches on a 4.8 x 4.0 canvas)
# ----------------------------------------------------------------------------
def module(ax, x, y, w, h, col, title, body, num=None, fs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=1.0, edgecolor=col[1], facecolor=col[0], zorder=2))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=6.5,
            fontweight="bold", color=col[1], zorder=3)
    ax.text(x + w / 2, y + h - 0.23, body, ha="center", va="top", fontsize=fs,
            linespacing=1.3, zorder=3)
    if num is not None:
        ax.add_patch(Circle((x + 0.02, y + h - 0.02), 0.085, facecolor=col[1], edgecolor="white",
                            linewidth=0.8, zorder=4))
        ax.text(x + 0.02, y + h - 0.02, str(num), ha="center", va="center", fontsize=6.2,
                color="white", fontweight="bold", zorder=5)


def cylinder(ax, x, y, w, h, col, title, body, fs=5.9):
    e = 0.09
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, facecolor=col[0], edgecolor="none", zorder=2))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=col[1], linewidth=1.0, zorder=3)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=col[1], linewidth=1.0, zorder=3)
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, facecolor=col[0], edgecolor=col[1],
                         linewidth=1.0, zorder=2))
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, facecolor=col[0], edgecolor=col[1],
                         linewidth=1.0, zorder=4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=6.5, fontweight="bold",
            color=col[1], zorder=5)
    ax.text(x + w / 2, y + h - 0.36, body, ha="center", va="top", fontsize=fs, linespacing=1.3, zorder=5)


def arr(ax, pts, color="#333333", lw=1.1):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))


def panel(ax, x, y, w, h, label, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=0.8, edgecolor="#9a9a9a", facecolor="#FAFAFA",
                                linestyle=(0, (4, 2)), zorder=0))
    ax.text(x + 0.1, y + h - 0.07, label, fontsize=8, fontweight="bold", va="top", zorder=1)
    ax.text(x + 0.3, y + h - 0.085, title, fontsize=6.8, va="top", style="italic", zorder=1)


def fig_pipeline_flowchart(path="fig_pipeline_flowchart.pdf"):
    H = 4.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---------------- Panel A: construction ----------------
    panel(ax, 0.04, 1.62, 4.72, 2.4, "A", "Counterfactual construction")
    w, h = 1.0, 0.82
    xs = [0.12, 1.33, 2.54, 3.75]
    r1 = 2.98
    cylinder(ax, xs[0], r1, w, h, C_DATA, "CUREv1", "2,000 queries\nen, fr, es\nEnglish corpus")
    module(ax, xs[1], r1, w, h, C_LLM, "Generator", "Qwen2.5-7B proposes\nsource span,\nreplacement, q$'$", 1)
    module(ax, xs[2], r1, w, h, C_DET, "Substitution", "find span in d;\nreplace the unique\nmatch to give d$'$", 2)
    module(ax, xs[3], r1, w, h, C_LLM, "Validator", "Llama-3.1-8B checks\nplausibility, logic,\nleakage, safety", 3)
    my = r1 + h / 2
    for i in range(3):
        arr(ax, [(xs[i] + w + 0.01, my), (xs[i + 1] - 0.01, my)])
    # discard branch
    ax.text(xs[2] + w / 2, r1 - 0.1, "no unique match:\ndiscard", ha="center", va="top",
            fontsize=5.5, color=C_BAD, style="italic")
    arr(ax, [(xs[2] + w / 2, r1 - 0.005), (xs[2] + w / 2, r1 - 0.075)], color=C_BAD, lw=0.9)

    # row 2: example (left) and triage / release (right), flowing right to left
    r2 = 1.74
    h2 = 0.92
    module(ax, xs[3], r2, w, h2, C_DET, "Triage", "one changed region;\nno answer value\nleaked into d$'$", 4)
    arr(ax, [(xs[3] + w / 2, r1 - 0.005), (xs[3] + w / 2, r2 + h2 + 0.01)])
    ax.plot([xs[3] + w / 2, xs[3] + w / 2], [r1 - 0.005, r2 + h2 + 0.01], color="#333333", linewidth=1.1, zorder=1)
    cylinder(ax, xs[2], r2, w, h2, C_DATA, "Released pairs", "Ortho. 111\nPulm. 122\nPsych. 113")
    arr(ax, [(xs[3] - 0.01, r2 + h2 / 2), (xs[2] + w + 0.01, r2 + h2 / 2)])

    # example box
    ex_x, ex_w = 0.12, 2.3
    ax.add_patch(FancyBboxPatch((ex_x, r2), ex_w, h2, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=0.9, edgecolor="#777777", facecolor="white", zorder=2))
    ax.text(ex_x + ex_w / 2, r2 + h2 - 0.07, "Example pair (illustrative, not from the data)", ha="center",
            va="top", fontsize=6.0, fontweight="bold", color="#444444", zorder=3)
    cx0, cx1 = ex_x + 0.1, ex_x + 1.22
    ax.text(cx0, r2 + h2 - 0.25, "Original", fontsize=5.8, color=C_DATA[1], fontweight="bold", va="top", zorder=3)
    ax.text(cx1, r2 + h2 - 0.25, "Edited", fontsize=5.8, color=C_BAD, fontweight="bold", va="top", zorder=3)
    ax.text(cx0, r2 + h2 - 0.38, "q:  therapy for\n     stage $\\mathbf{II}$ sarcoidosis", fontsize=5.7, va="top", linespacing=1.3, zorder=3)
    ax.text(cx1, r2 + h2 - 0.38, "q$'$: therapy for\n      stage $\\mathbf{IV}$ sarcoidosis", fontsize=5.7, va="top", linespacing=1.3, zorder=3)
    ax.text(cx0, r2 + h2 - 0.62, "d:  ... advised for\n     stage $\\mathbf{II}$ ...", fontsize=5.7, va="top", linespacing=1.3, zorder=3)
    ax.text(cx1, r2 + h2 - 0.62, "d$'$: ... advised for\n      stage $\\mathbf{IV}$ ...", fontsize=5.7, va="top", linespacing=1.3, zorder=3)
    arr(ax, [(xs[2] - 0.0, r2 + h2 / 2), (ex_x + ex_w + 0.01, r2 + h2 / 2)], color="#999999", lw=0.8)

    # ---------------- Panel B: evaluation ----------------
    panel(ax, 0.04, 0.38, 4.72, 1.14, "B", "Bidirectional evaluation")
    by, bh = 0.46, 0.7
    module(ax, xs[0], by, w, bh, C_DATA, "Pair set", "q, d, q$'$, d$'$\nper approved\ncandidate", fs=5.8)
    module(ax, xs[1], by, w, bh, C_EVAL, "Systems", "dense, late,\nfixed fusion,\nlearned gate", fs=5.8)
    module(ax, xs[2], by, w, bh, C_EVAL, "Score", "S(q,d) > S(q,d$'$)\nS(q$'$,d$'$) > S(q$'$,d)\nties are errors", fs=5.5)
    module(ax, xs[3], by, w, bh, C_EVAL, "Metrics", "A$_o$, A$_c$, balanced,\npair consistency;\nbootstrap intervals", fs=5.5)
    my2 = by + bh / 2
    for i in range(3):
        arr(ax, [(xs[i] + w + 0.01, my2), (xs[i + 1] - 0.01, my2)])
    # released pairs -> pair set
    arr(ax, [(xs[2] + w / 2, r2 - 0.005), (xs[2] + w / 2, 1.575), (xs[0] + w / 2, 1.575),
             (xs[0] + w / 2, by + bh + 0.01)])

    # ---------------- legend ----------------
    items = [(C_DATA, "Data"), (C_LLM, "Language model"), (C_DET, "Deterministic step"), (C_EVAL, "Evaluation")]
    lx = 0.35
    for col, lab in items:
        ax.add_patch(FancyBboxPatch((lx, 0.1), 0.2, 0.13, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=col[0], edgecolor=col[1], linewidth=0.9))
        ax.text(lx + 0.27, 0.165, lab, fontsize=6.0, va="center")
        lx += 0.45 + 0.052 * len(lab)
    fig.savefig(path)
    plt.close(fig)


def chip(ax, x, y, w, h, col, text, fs=5.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.9, edgecolor=col[1], facecolor=col[0], zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25,
            fontweight="bold" if bold else "normal", zorder=3)


def fig_pipeline(path="fig_pipeline.pdf"):
    """Framework view: inputs, construction, evaluation. Main methods only."""
    H = 3.5
    o = 0.7
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = o + 2.6
    bot = o
    C_NEUTRAL = ("#F3F3F3", "#666666")

    panel(ax, 0.04, bot, 0.88, top - bot, "", "Inputs")
    panel(ax, 0.98, bot, 1.68, top - bot, "", "Construction")
    panel(ax, 2.82, bot, 1.94, top - bot, "", "Bidirectional evaluation")

    # ---- inputs ----
    cylinder(ax, 0.12, top - 1.0, 0.72, 0.72, C_DATA, "CUREv1", "queries and\npassages\nen, fr, es", fs=5.4)
    module(ax, 0.12, bot + 0.1, 0.72, 1.08, C_DATA, "Attributes",
           "negation\ndosage\nfrequency\nsubtype\nstage\ntemporal\nseverity", fs=5.3)

    # ---- construction ----
    bx, bw, bh = 1.12, 1.4, 0.54
    y1, y2 = top - 0.84, top - 1.52
    y3, h3 = bot + 0.1, 0.8
    module(ax, bx, y1, bw, bh, C_LLM, "Generator", "Qwen2.5-7B proposes span,\nreplacement, edited query", 1, fs=5.2)
    module(ax, bx, y2, bw, bh, C_DET, "Substitution", "unique match in passage\ngives the edited passage", 2, fs=5.2)
    module(ax, bx, y3, bw, h3, C_NEUTRAL, "Screening", "", 3)
    chip(ax, bx + 0.08, y3 + 0.34, bw - 0.16, 0.25, C_LLM, "Validator: Llama-3.1-8B", fs=5.2)
    chip(ax, bx + 0.08, y3 + 0.06, bw - 0.16, 0.25, C_DET, "Two rule-based filters", fs=5.2)
    arr(ax, [(bx + bw / 2, y1 - 0.005), (bx + bw / 2, y2 + bh + 0.01)])
    arr(ax, [(bx + bw / 2, y2 - 0.005), (bx + bw / 2, y3 + h3 + 0.01)])
    arr(ax, [(0.12 + 0.72 + 0.01, y1 + 0.38), (bx - 0.01, y1 + 0.38)])
    ay = bot + 0.1 + 0.54
    arr(ax, [(0.12 + 0.72 + 0.01, ay), (0.95, ay), (0.95, y1 + 0.14), (bx - 0.01, y1 + 0.14)])

    # ---- evaluation ----
    ex, ew = 2.92, 1.74
    ps_h = 0.5
    ps_y = top - 0.3 - ps_h
    module(ax, ex, ps_y, ew, ps_h, C_DATA, "Pair set", "346 screened pairs\n82 training, 264 evaluation", fs=5.2)
    gy1 = ps_y - 0.1
    gy0 = gy1 - 1.02
    ax.add_patch(FancyBboxPatch((ex - 0.04, gy0), ew + 0.08, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.8, edgecolor="#8a8a8a", facecolor="white", linestyle=(0, (3, 2)), zorder=1))
    ax.text(ex + 0.02, gy1 - 0.04, "Both query directions", fontsize=5.8, style="italic", va="top", color="#444444", zorder=3)
    cg, cy, ch = 0.06, gy1 - 0.52, 0.36
    widths = [0.44, 0.5, ew - 0.44 - 0.5 - 2 * cg]
    chips = [("Dense", C_EVAL), ("Late\ninteraction", C_EVAL), ("Attribute features\nfull or masked", C_DET)]
    cx = ex
    cxs = []
    for (t, col), cw in zip(chips, widths):
        chip(ax, cx, cy, cw, ch, col, t, fs=5.1)
        cxs.append(cx + cw / 2)
        cx += cw + cg
    fy = gy0 + 0.07
    chip(ax, ex, fy, ew, 0.28, C_EVAL, "Fixed rule or learned gate", fs=5.4, bold=True)
    for c in cxs:
        arr(ax, [(c, cy - 0.005), (c, fy + 0.29)])
    arr(ax, [(ex + ew / 2, ps_y - 0.005), (ex + ew / 2, gy1 + 0.01)])
    my = bot + 0.1
    module(ax, ex, my, ew, 0.46, C_EVAL, "Bidirectional metrics", "A$_o$, A$_c$, balanced, pair\nconsistency, exact intervals", fs=5.1)
    arr(ax, [(ex + ew / 2, gy0 - 0.005), (ex + ew / 2, my + 0.47)])
    arr(ax, [(bx + bw + 0.01, y3 + h3 / 2), (2.74, y3 + h3 / 2), (2.74, ps_y + ps_h / 2), (ex - 0.01, ps_y + ps_h / 2)])

    # ---- example strip ----
    ey, eh = 0.27, 0.34
    ax.add_patch(FancyBboxPatch((0.04, ey), 4.72, eh, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.8, edgecolor="#777777", facecolor="white", zorder=2))
    ax.text(0.12, ey + eh - 0.05, "Example (illustrative)", fontsize=5.6, fontweight="bold", color="#444444", va="top", zorder=3)
    ax.text(1.38, ey + eh - 0.05, "Passage edit: ... advised for stage $\\mathbf{II}$ ...   $\\rightarrow$   ... advised for stage $\\mathbf{IV}$ ...",
            fontsize=5.6, va="top", zorder=3)
    ax.text(1.38, ey + eh - 0.18, "Edited query: rewritten to fit the edited passage, new value not stated.",
            fontsize=5.6, va="top", zorder=3)

    items = [(C_DATA, "Data"), (C_LLM, "Language model"), (C_DET, "Deterministic step"), (C_EVAL, "Evaluation")]
    lx = 0.35
    for col, lab in items:
        ax.add_patch(FancyBboxPatch((lx, 0.06), 0.2, 0.12, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=col[0], edgecolor=col[1], linewidth=0.9))
        ax.text(lx + 0.27, 0.12, lab, fontsize=6.0, va="center")
        lx += 0.45 + 0.052 * len(lab)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_ndcg()
    fig_pair_consistency()
    print("saved fig_pipeline.pdf, fig_ndcg_revised.pdf, fig_pair_consistency.pdf")
