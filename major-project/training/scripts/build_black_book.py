"""
Build the SPIT black book (LaTeX) from paper/Thesis_draft.docx.

Output: paper/black_book/ (report.tex and one .tex per template section, figs/).
Chapter text, tables and captions come from the .docx, so numbers are never retyped.
Front matter (title, certificates, ...) reads names from details.tex, which is only
created if missing, so re-running never overwrites filled-in names.

Run from major-project/: python training/scripts/build_black_book.py
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper" / "Thesis_draft.docx"
OUT = ROOT / "paper" / "black_book"
TEXT_W_MM = 150

EQUATIONS = {  # the .docx holds these as images; transcribed from those images
    "equation1": r"E_i = C(t_{\mathrm{end}}) - C(t_{\mathrm{start}})",
    "equation2": r"T_i = \frac{n_i}{t_i} \qquad\qquad P_i = \frac{E_i}{t_i}",
    "equation3": r"\text{Stack A: } e = \operatorname{median}_{i:\, n_i \geq 16} \frac{E_i}{n_i}"
                 r"\qquad\qquad \text{Stack B: } e = \frac{\sum_i E_i}{\sum_i n_i}",
    "equation4": r"E_i = E_b \cdot \frac{n_i}{\sum_{j \in b} n_j}",
    "equation5": r"\text{saving} = 1 - \frac{e_{\mathrm{config}}}{e_{\mathrm{float16}}}",
    "equation6": r"\hat{x} = \min\!\left(1,\ \max\!\left(0,\ "
                 r"\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right)\right)",
    "equation7": r"\mathrm{FK} = 0.39\,\frac{w}{s} + 11.8\,\frac{y}{w} - 15.59"
                 r"\qquad\qquad H = -\sum_c p_c \log_2 p_c",
    "equation8": r"y^{*} = \frac{\int y\,\mu(y)\,dy}{\int \mu(y)\,dy}",
}
EQUATION8_TIER = (r"\text{tier} = \text{low if } y^{*} < 33;\quad \text{middle if } "
                  r"33 \leq y^{*} \leq 66;\quad \text{high if } y^{*} > 66")
FIG_NAMES = {"image1": "fig_architecture", "image2": "fig_workflow", "image3": "fig_router",
             "image4": "fig_relative_energy", "image5": "fig_why", "image6": "fig_tradeoff",
             "image7": "fig_disruption", "image8": "fig_routing_run5"}
# Figure 5.1's title is drawn inside the image; its .docx caption is only a source line.
CAPTION_OVERRIDE = {"Figure 5.1": "Energy per token relative to the same model's float16 at "
                                  "batch 1: vLLM tiers use 36-64% less, bitsandbytes tiers save "
                                  "none. Source: paper/results.md and "
                                  "paper/analysis/session5_summary.csv; ratios computed from the "
                                  "medians in Tables 5.1 and 5.4."}
UNICODE = {"×": r"$\times$", "·": r"\textperiodcentered{}", "–": "--", "—": "---", "−": "$-$",
           "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "’": "'", "‘": "`", "“": "``",
           "”": "''", "→": r"$\rightarrow$", "\u00a0": "~", "±": r"$\pm$", "…": r"\ldots{}",
           "μ": r"$\mu$", "\u200b": ""}


def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#{}":
            out.append("\\" + ch)
        elif ch == "_":
            out.append(r"\_\allowbreak{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch in "<>":
            out.append(r"\textless{}" if ch == "<" else r"\textgreater{}")
        elif ch == '"':
            out.append("''")
        else:
            out.append(UNICODE.get(ch, ch))
    return "".join(out)


def fix_quotes(s):
    # straight double quotes were turned into '' above; make opening ones ``
    return re.sub(r"(^|[\s(\[])''", r"\1``", s)


def cites(s):
    def rep(m):
        return r"~\cite{" + ",".join("b" + n for n in re.findall(r"\d+", m.group(0))) + "}"
    return re.sub(r"\s?(?:\[\d{1,2}\])+", rep, s)


class Conv:
    def __init__(self):
        self.d = docx.Document(SRC)
        self.numfmt = self._numbering()

    def _numbering(self):
        fmt, part = {}, self.d.part.numbering_part
        abstract = {}
        for a in part.element.findall(qn("w:abstractNum")):
            lvl = a.find(qn("w:lvl"))
            f = lvl.find(qn("w:numFmt")).get(qn("w:val"))
            abstract[a.get(qn("w:abstractNumId"))] = f
        for n in part.element.findall(qn("w:num")):
            fmt[n.get(qn("w:numId"))] = abstract[n.find(qn("w:abstractNumId")).get(qn("w:val"))]
        return fmt

    def list_kind(self, el):
        num = el.find(".//" + qn("w:numId"))
        if num is None:
            return None
        return "itemize" if self.numfmt.get(num.get(qn("w:val"))) == "bullet" else "enumerate"

    def inline(self, el, with_cites=True, plain_bold=False):
        """Paragraph element to LaTeX, keeping bold/italic/mono and hyperlinks."""
        chunks = []  # (text, b, i, m, url)

        def run(r, url):
            t = "".join((c.text or "") if c.tag == qn("w:t") else
                        (" " if c.tag in (qn("w:tab"), qn("w:br")) else "")
                        for c in r)
            if not t:
                return
            rp = r.find(qn("w:rPr"))
            on = lambda tag: rp is not None and rp.find(qn(tag)) is not None and \
                rp.find(qn(tag)).get(qn("w:val")) not in ("0", "false")
            font = rp.find(qn("w:rFonts")) if rp is not None else None
            mono = font is not None and any(k in (font.get(qn("w:ascii")) or "")
                                            for k in ("Consol", "Courier", "Mono"))
            chunks.append([t, on("w:b"), on("w:i"), mono, url])

        for c in el:
            if c.tag == qn("w:r"):
                run(c, None)
            elif c.tag == qn("w:hyperlink"):
                rid = c.get(qn("r:id"))
                url = self.d.part.rels[rid].target_ref if rid else None
                for r in c.findall(qn("w:r")):
                    run(r, url)
        merged = []
        for ch in chunks:
            if merged and merged[-1][1:] == ch[1:]:
                merged[-1][0] += ch[0]
            else:
                merged.append(ch)
        out = ""
        for t, b, i, m, url in merged:
            s = esc(t)
            if m:
                s = r"\texttt{" + s + "}"
            if i:
                s = r"\emph{" + s + "}"
            if b and not plain_bold:
                lead = len(s) - len(s.lstrip())
                trail = len(s) - len(s.rstrip())
                s = s[:lead] + r"\textbf{" + s.strip() + "}" + (s[-trail:] if trail else "")
            if url:
                s = r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + s + "}"
            out += s
        out = fix_quotes(out.strip())
        return cites(out) if with_cites else out

    def widths(self, rows):
        n = len(rows[0])
        avail = TEXT_W_MM - n * 2.8
        floors, weights = [], []
        for j in range(n):
            texts = [re.sub(r"\\[a-zA-Z]+\*?|[{}]", " ", r[j][0]) for r in rows if j < len(r)]
            words = [w for t in texts for w in re.split(r"[\s/]+", t) if w]
            longest = min(max((len(w) for w in words), default=4), 16)
            floors.append(longest * 1.75)
            weights.append(max(sum(len(t) for t in texts), 1) ** 0.5)
        total = sum(floors)
        if total >= avail:
            return [f * avail / total for f in floors]
        rest = avail - total
        return [f + rest * w / sum(weights) for f, w in zip(floors, weights)]

    def table(self, el, caption=None):
        t = Table(el, self.d)
        rows = []
        for tr in t._tbl.tr_lst:
            cells = []
            for tc in tr.tc_lst:
                span = tc.grid_span
                paras = [self.inline(p) for p in tc.findall(qn("w:p"))]
                cells.append((r" \newline ".join(p for p in paras if p), span))
            rows.append(cells)
        n = max(sum(c[1] for c in r) for r in rows)
        flat = [[c for c in r for _ in range(c[1])] for r in rows]
        flat = [r + [("", 1)] * (n - len(r)) for r in flat]
        w = self.widths(flat)
        spec = "".join(r">{\raggedright\arraybackslash}p{%.1fmm}" % x for x in w)

        def row(r, head=False):
            cells, j = [], 0
            for text, span in r:
                if head:
                    text = r"\textbf{" + text.replace(r"\textbf{", "{") + "}" if text else ""
                if span > 1:
                    wid = sum(w[j:j + span]) + (span - 1) * 2.8
                    text = (r"\multicolumn{%d}{>{\raggedright\arraybackslash}p{%.1fmm}}{%s}"
                            % (span, wid, text))
                cells.append(text)
                j += span
            return " & ".join(cells) + r" \\"

        out = ["", r"\begingroup\footnotesize\setlength{\tabcolsep}{4pt}",
               r"\begin{longtable}{" + spec + "}", r"\toprule", row(rows[0], True), r"\midrule",
               r"\endhead"]
        out += [row(r) for r in rows[1:]]
        out.append(r"\bottomrule")
        if caption:
            out.append(caption + r"\\")
        out.append(r"\end{longtable}")
        out += [r"\endgroup", ""]
        return "\n".join(out)

    def caption(self, el, prefix):
        raw = "".join(x.text or "" for x in el.iter(qn("w:t")))
        m = re.match(prefix + r"\s+(\d+\.\d+)\s*[:\u00b7·]?\s*(.*)", raw, re.S)
        key = f"{prefix} {m.group(1)}"
        if key in CAPTION_OVERRIDE:
            body = fix_quotes(esc(CAPTION_OVERRIDE[key]))
        else:
            # keep formatting-free text, then capitalise the first letter
            body = fix_quotes(esc(m.group(2).strip()))
            body = cites(body[0].upper() + body[1:])
        short = re.split(r"\.\s|\s\\textperiodcentered\{\}\s|\s+Source:", body)[0].rstrip(".")
        short = re.sub(r"~?\\cite\{[^}]*\}", "", short)
        return key, r"\caption[" + short.replace("]", "") + "]{" + body + "}"


def build():
    c = Conv()
    body = list(c.d.element.body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    z = zipfile.ZipFile(SRC)
    sections, cur = {}, None
    seq = {"Table": [], "Figure": []}
    open_list = None
    skip = set()

    def emit(s):
        nonlocal open_list
        if open_list:
            sections[cur].append(r"\end{%s}" % open_list)
            open_list = None
        sections[cur].append(s)

    for i, el in enumerate(body):
        if i in skip:
            continue
        if el.tag == qn("w:tbl"):
            cap = None
            nxt = body[i + 1] if i + 1 < len(body) else None
            if nxt is not None and nxt.tag == qn("w:p"):
                t = "".join(x.text or "" for x in nxt.iter(qn("w:t")))
                if re.match(r"Table \d+\.\d+", t):
                    key, cap = c.caption(nxt, "Table")
                    seq["Table"].append(key)
                    skip.add(i + 1)
            emit(c.table(el, cap))
            continue
        if el.tag != qn("w:p"):
            continue
        p = Paragraph(el, c.d)
        style, text = p.style.name, p.text.strip()
        blip = el.find(".//" + qn("a:blip"))
        if style == "Heading 1":
            cur = "_title"
            sections[cur] = []
            continue
        if style == "Heading 2":
            if open_list:
                sections[cur].append(r"\end{%s}" % open_list)
                open_list = None
            m = re.match(r"Chapter (\d): (.*)", text)
            if m:
                cur = {"1": "introduction", "2": "literature_survey", "3": "design",
                       "4": "implementation", "5": "results", "6": "conclusion"}[m.group(1)]
                sections[cur] = [r"\chapter{%s}" % esc(m.group(2))]
            elif text == "Abstract":
                cur = "abstract"
                sections[cur] = []
            elif text == "References":
                cur = "bibliography"
                sections[cur] = []
            elif text.startswith("Appendix A"):
                cur = "appendix"
                sections[cur] = [r"\appendix", r"\chapter{%s}" % esc(text.split(": ", 1)[1])]
            continue
        if cur == "_title":
            continue
        if style == "Heading 3":
            title = re.sub(r"^\d+\.\d+\s+", "", text)
            emit(r"\section{%s}" % esc(title))
            continue
        if blip is not None:
            part = c.d.part.related_parts[blip.get(qn("r:embed"))].partname
            stem = Path(str(part)).stem
            if stem.startswith("equation"):
                emit(r"\begin{equation}" + "\n" + EQUATIONS[stem] + "\n" + r"\end{equation}")
                if stem == "equation8":
                    emit(r"\begin{equation}" + "\n" + EQUATION8_TIER + "\n" + r"\end{equation}")
                continue
            name = FIG_NAMES[stem] + ".png"
            (OUT / "figs" / name).write_bytes(z.read(f"word/media/{stem}.png"))
            nxt = body[i + 1]
            key, cap = c.caption(nxt, "Figure")
            seq["Figure"].append(key)
            skip.add(i + 1)
            emit("\n".join([r"\begin{figure}[htbp]", r"\centering",
                            r"\includegraphics[width=\linewidth]{figs/%s}" % name, cap,
                            r"\end{figure}"]))
            continue
        if not text:
            continue
        if cur == "bibliography":
            m = re.match(r"\[(\d+)\]\s*", text)
            if m:
                ltx = c.inline(el, with_cites=False)
                ltx = re.sub(r"^\[\d+\]\s*", "", ltx)
                sections[cur].append(r"\bibitem{b%s} %s" % (m.group(1), ltx))
            continue
        kind = c.list_kind(el)
        if kind:
            if open_list != kind:
                if open_list:
                    sections[cur].append(r"\end{%s}" % open_list)
                sections[cur].append(r"\begin{%s}" % kind)
                open_list = kind
            sections[cur].append(r"\item " + c.inline(el))
            continue
        emit(c.inline(el) + "\n")
    if open_list:
        sections[cur].append(r"\end{%s}" % open_list)

    exp_t = [f"Table 5.{k}" for k in range(1, 15)]
    exp_f = ["Figure 3.1", "Figure 3.2", "Figure 3.3"] + [f"Figure 5.{k}" for k in range(1, 6)]
    assert seq["Table"] == exp_t, seq["Table"]
    assert seq["Figure"] == exp_f, seq["Figure"]

    w = lambda name, s: (OUT / f"{name}.tex").write_text(s, encoding="utf-8")
    for name in ("introduction", "literature_survey", "design", "implementation", "results",
                 "conclusion", "appendix"):
        w(name, "\n".join(sections[name]) + "\n")
    w("abstract", r"\clearpage" + "\n" + r"\begin{abstract}" + "\n"
      + "\n".join(sections["abstract"]) + "\n" + r"\end{abstract}" + "\n")
    w("bibliography", r"\clearpage\phantomsection" + "\n"
      + r"\addcontentsline{toc}{chapter}{Bibliography}" + "\n"
      + r"\begin{thebibliography}{99}" + "\n" + "\n".join(sections["bibliography"]) + "\n"
      + r"\end{thebibliography}" + "\n")
    print("chapters written;", len(seq["Table"]), "captioned tables,", len(seq["Figure"]), "figures")


if __name__ == "__main__":
    sys.exit(build())
