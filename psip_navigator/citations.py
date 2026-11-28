"""Turn the answer's raw source IDs ([WEL-010-03], [AET-POL-005, CUR-Q004]) into numbered,
clickable citations [1] [2] for the chat page.

Clicking a number opens a small box under that line with the source's title, the passage it
cites and a link to the original. The same sources are listed, numbered, under the answer.
"""
import html
import re

ID = r"[A-Z]{2,4}-[A-Z0-9-]+"
GROUP = re.compile(rf"\[\s*({ID}(?:\s*[,;，、]\s*{ID})*)\s*\]")

CSS = """
<style>
.psip-cite { cursor: pointer; outline: none; }
.psip-cite > sup { color: #011F5B; font-weight: 600; padding: 0 1px; }
.psip-cite:hover > sup, .psip-cite:focus-within > sup { text-decoration: underline; }
.psip-pop { display: none; }
.psip-cite:focus-within .psip-pop {
  display: block; margin: 6px 0 10px; padding: 10px 12px; border-radius: 8px;
  background: rgba(1, 31, 91, 0.05); border-left: 3px solid #011F5B;
  font-size: 0.86rem; line-height: 1.45; font-weight: 400; cursor: auto;
}
.psip-pop b, .psip-src b { display: block; margin-bottom: 3px; }
.psip-quote { display: block; color: rgba(49, 51, 63, 0.8); }
.psip-src { margin: 0 0 12px; font-size: 0.88rem; line-height: 1.45; }
.psip-src .psip-num { font-weight: 700; color: #011F5B; }
</style>
"""


def _clean(text: str, limit: int = 420) -> str:
    t = re.sub(r"\[([^\]]+)\]\((?:https?|tel|mailto):[^)]*\)", r"\1", text)  # markdown links -> text
    t = re.sub(r"\*\*|__|^#+\s*|\s#+\s", " ", t, flags=re.M)                 # bold / heading marks
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + " …"
    return t


def _esc(text: str) -> str:
    # escape HTML, and "$" so Streamlit does not read dollar amounts as math
    return html.escape(text).replace("$", "&#36;")


SITES = {"wellness.upenn.edu": "Wellness at Penn", "global.upenn.edu": "Penn Global",
         "pennmedicine.org": "Penn Medicine", "lghealth.org": "Penn Medicine",
         "healthcare.gov": "HealthCare.gov", "cms.gov": "CMS.gov"}


def _site(url: str) -> str:
    host = re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0]
    return next((v for k, v in SITES.items() if host.endswith(k)), "")


def title_of(c: dict) -> str:
    if c.get("source_type") == "curated":
        based = ", ".join(c.get("cites") or [])
        return "Team-verified answer" + (f" (based on {based})" if based else "")
    page = f", p. {c['page_start']}" if c.get("page_start") else ""
    title = c.get("source_title") or c["id"]
    site = "Aetna" if c["id"].startswith("AET-") else _site(c.get("url"))
    if site and site.lower() not in title.lower():
        title = f"{site}: {title}"
    return f"{title}{page}"


def _link(c: dict) -> str:
    url = c.get("url") or ""
    if not url.startswith("http"):
        return ""
    return f' <a href="{html.escape(url, quote=True)}" target="_blank">Open source ↗</a>'


def render(text: str, chunks, partial: bool = False):
    """Return (answer_markdown, sources_html, n_sources).

    answer_markdown: the answer with [ID] groups replaced by clickable [1] [2] markers.
    sources_html: the numbered source list (empty string when nothing was cited).
    partial: True while streaming; hides a half-written "[WEL-0" at the end.
    """
    by_id = {c["id"]: c for c in chunks or []}
    order = []

    def num(i):
        if i not in order:
            order.append(i)
        return order.index(i) + 1

    def repl(m):
        ids = [i for i in re.split(r"\s*[,;，、]\s*", m.group(1)) if i in by_id]
        out = []
        for i in dict.fromkeys(ids):
            n, c = num(i), by_id[i]
            out.append(
                f'<span class="psip-cite" tabindex="0"><sup>[{n}]</sup>'
                f'<span class="psip-pop"><b>[{n}] {_esc(title_of(c))}</b>'
                f'<span class="psip-quote">“{_esc(_clean(c["text"]))}”</span>{_link(c)}</span></span>')
        return "".join(out)

    body = text
    if partial:
        body = re.sub(r"\[[A-Z0-9 ,;，、-]*$", "", body)
    body = body.replace("$", r"\$")
    body = GROUP.sub(repl, body)

    items = []
    for n, i in enumerate(order, 1):
        c = by_id[i]
        items.append(f'<div class="psip-src"><b><span class="psip-num">[{n}]</span> {_esc(title_of(c))}'
                     f'{_link(c)}</b><span class="psip-quote">“{_esc(_clean(c["text"], 600))}”</span></div>')
    return body, "".join(items), len(order)
