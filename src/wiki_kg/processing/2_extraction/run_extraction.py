# the parser in particular is quite messy as a large part of it was vibe coded (there were simply too many exceptions and edge cases to handle)
# sorry for the low legibility
# - Guilherme

from datatrove.pipeline.base import PipelineStep
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Configuration constants ----
# Paths/buckets
HERE = Path(__file__).resolve().parent
GCP_RAW_PREFIX = "gs://wikipedia-graph/wikipedia/raw_html_dumps"
GCP_PARSED_PREFIX = "gs://wikipedia-graph/wikipedia/parsed_html"
DISAMBIG_IDS_PATH = HERE / "disambiguation_sia_ids.json"
COMPILED_REF_WORDS_PATH = HERE / "compiled_ref_words.json"
LOGGING_DIR = HERE / "logs"

# Slurm defaults
SLURM_TIME = "10:00:00"
SLURM_PARTITION = "hopper-cpu"
SLURM_CPUS_PER_TASK = 3
SLURM_QOS = "high"
SLURM_MEM_PER_CPU = "1950M"


class WikipediaReader(PipelineStep):
    name = "WikipediaReader"

    def __init__(self, wiki: str):
        super().__init__()
        self.wiki = wiki

    def run(self, data, rank=0, world_size=1):
        from datatrove.io import get_datafolder
        from collections import Counter  # noqa: F401  # kept for parity with original
        from tqdm import tqdm  # noqa: F401  # kept for parity with original

        import tarfile
        import io
        import json
        import re
        from datatrove.data import Document

        wiki_df = get_datafolder(GCP_RAW_PREFIX + "/" + self.wiki)

        with open(DISAMBIG_IDS_PATH, "r") as f:
            disambiguation_ids = set(
                [
                    x["item"].removeprefix("http://www.wikidata.org/entity/")
                    for x in json.load(f)
                ]
            )

        def is_disambiguation(rec):
            wikidata_id = rec.get("main_entity", {}).get("identifier")
            if wikidata_id in disambiguation_ids:
                return True
            # fallback
            html = rec.get("article_body", {}).get("html") or ""
            return "mw:PageProp/disambiguation" in html

        _redir_re = re.compile(r"^\s*#redirect\b", re.I)

        def is_redirect(rec):
            wt = rec.get("article_body", {}).get("wikitext") or ""
            if _redir_re.match(wt):
                return True
            html = rec.get("article_body", {}).get("html") or ""
            # In Parsoid/HTML, real redirect pages are indicated in the head with
            # a link tag having rel="mw:PageProp/redirect". The class "mw-redirect"
            # marks links that point to a redirect target and appears on many
            # normal pages, so it must NOT be used to classify the page itself.
            if not html:
                return False
            return ('rel="mw:PageProp/redirect"' in html) or (
                "rel='mw:PageProp/redirect'" in html
            )

        def iter_jsonl(path):
            with (
                wiki_df.open(path, "rb") as fh,
                tarfile.open(fileobj=fh, mode="r:gz") as tf,
            ):
                for m in tf:
                    with tf.extractfile(m) as f:
                        for _i, line in enumerate(
                            io.TextIOWrapper(f, encoding="utf-8")
                        ):
                            yield json.loads(line)

        def iter_docs(jsonl_iter):
            for rec in jsonl_iter:
                self.stat_update("total")
                # filter: ns0 + non-redirects
                ns = (rec.get("namespace") or {}).get("identifier")
                if ns != 0:
                    self.stat_update("dropped_ns")
                    continue
                if is_redirect(rec):
                    self.stat_update("dropped_redirect")
                    continue

                if is_disambiguation(rec):
                    self.stat_update("dropped_disamb")
                    continue

                page_id = rec.get("identifier")
                title = rec.get("name")
                url = rec.get("url")
                ts_modified = rec.get("date_modified")
                html = rec.get("article_body", {}).get("html") or ""
                wikitext = rec.get("article_body", {}).get("wikitext") or ""
                version = (rec.get("version") or {}).get("identifier")
                wikidata_id = rec.get("main_entity", {}).get("identifier")
                if not html:
                    self.stat_update("dropped_no_html")
                    continue
                wikiname = rec.get("is_part_of", {}).get("identifier")

                meta = {
                    "wikiname": wikiname,
                    "page_id": page_id,
                    "title": title,
                    "url": url,
                    "date_modified": ts_modified,
                    "in_language": rec.get("in_language", {}).get("identifier"),
                    "wikidata_id": wikidata_id,
                    "bytes_html": len(html.encode("utf-8")),
                    "wikitext": wikitext,
                    "version": version,
                    # 'html': html,
                    # "ALL_STUFF": rec
                }

                yield Document(
                    text=html,  # KEEP ALL HTML
                    id=f"{wikiname}/{page_id}",
                    metadata=meta,
                )

        files_shard = wiki_df.get_shard(rank, world_size)
        for filepath in files_shard:
            for doc in iter_docs(iter_jsonl(filepath)):
                yield doc


import json
import re
from typing import List, Dict, Any

from datatrove.pipeline.base import PipelineStep


class WikipediaParser(PipelineStep):
    name = "WikipediaParser"

    def __init__(self, wiki: str, timeout: float = 5):
        super().__init__()
        self.wiki = wiki
        self.timeout = timeout

        import json

        with open(COMPILED_REF_WORDS_PATH) as f:
            self.ref_words = set(
                json.load(f).get(self.wiki.removesuffix("_namespace_0"), [])
            )

    # --- Infobox extraction (moved from app.py) ---
    # Keep the implementation identical to the working version in app.py
    def extract_infoboxes(self, stew) -> List[Dict[str, Any]]:
        import typing
        from bs4.element import Tag
        import re

        def _classes(tag: Tag) -> typing.List[str]:
            cls = tag.get("class") or []
            return cls if isinstance(cls, list) else str(cls).split()

        def _is_infobox_like_table(tag: Tag) -> bool:
            if tag.name != "table":
                return False

            def _has_infobox_class(t: Tag) -> bool:
                cls = [c.lower() for c in _classes(t)]
                return any(
                    ("infobox" in c)
                    or ("sinottico" in c)
                    or ("vcard" in c)
                    or ("infocaseta" in c)
                    for c in cls
                )

            if _has_infobox_class(tag):
                return True
            p = tag.parent
            while p is not None and hasattr(p, "name"):
                if _has_infobox_class(p):
                    return True
                p = p.parent
            # Heuristics: legacy fact boxes with right float/align or narrow widths, excluding wikitables
            try:
                cls_self = [c.lower() for c in (_classes(tag) or [])]
                if any("wikitable" in c for c in cls_self):
                    return False
                style_val = (tag.get("style") or "").lower()
                align_attr = (tag.get("align") or "").lower()
                width_attr = (tag.get("width") or "").lower()
                floated_like = ("float:right" in "".join(style_val.split())) or (
                    "float: right" in style_val
                )
                right_aligned = align_attr == "right"
                narrow_like = any(
                    tok in width_attr
                    for tok in ["160", "180", "200", "220", "240", "260", "280", "300"]
                ) or any(
                    tok in style_val
                    for tok in [
                        "width:160",
                        "width:180",
                        "width:200",
                        "width:220",
                        "width:240",
                        "width:260",
                        "width:280",
                        "width:300",
                    ]
                )
                if floated_like or right_aligned or narrow_like:
                    return True
            except Exception:
                pass
            return False

        def _is_infobox(tag: Tag) -> bool:
            return _is_infobox_like_table(tag)

        def _clean_key(s: str) -> str:
            s = (s or "").strip()
            if s.endswith(":"):
                s = s[:-1].strip()
            return s

        def _clean_value_text(s: str) -> str:
            import re

            s = (s or "").replace("\xa0", " ")
            s = " ".join(s.split())
            s = s.strip()
            # Trim dangling list/separator punctuation at the end (e.g., " ·", "/", ",")
            s = re.sub(r"(?:\s*(?:[·•/,:;]))+\s*$", "", s)
            return s

        def _assign_kv(store, key: str, value: str) -> None:
            if not key or not value:
                return
            if key in store:
                existing = store[key]
                if isinstance(existing, list):
                    if value not in existing:
                        existing.append(value)
                else:
                    if value != existing:
                        store[key] = [existing, value]
            else:
                store[key] = value

        def _extract_text(cell: Tag) -> str:
            # Prefer extraction on the original node (preserves table context for <br> → " · ")
            primary = self._extract_cell_text_with_filter(
                cell,
                exclude_elements={
                    "Citation",
                    "Reference",
                    "noprint",
                    "Navigation",
                    "Category",
                    "Media-audio",
                    "Media-img",
                    "Media-video",
                },
            ).strip()
            # Build an alternative text with citations/hidden nodes removed and explicit separator for line breaks
            alt = ""
            try:
                from bs4 import BeautifulSoup

                cell_soup = BeautifulSoup(str(cell), "lxml").find()
                for sup in cell_soup.find_all(
                    "sup",
                    class_=lambda v: v
                    and ("reference" in (v if isinstance(v, list) else str(v).split())),
                ):
                    sup.decompose()
                for t in cell_soup.find_all(True, attrs={"style": True}):
                    style_compact = "".join(str(t.get("style", "")).split()).lower()
                    if "display:none" in style_compact:
                        t.decompose()
                # Replace <br> with explicit separator but keep inline elements joined by a space
                for br in cell_soup.find_all("br"):
                    br.replace_with(" · ")
                alt = _clean_value_text(cell_soup.get_text(separator=" ", strip=True))
            except Exception:
                pass
            # Choose the variant only when it drops citations or primary is empty
            if ("[" in primary and "]" in primary and "[" not in alt) or (
                not primary and alt
            ):
                return alt
            return primary

        def _is_header_th(th: Tag) -> bool:
            cls = [c.lower() for c in _classes(th)]
            try:
                if th.get("scope", "").lower() == "row":
                    return False
                colspan = int(th.get("colspan", 1))
                if colspan >= 2:
                    parent_tr = th.find_parent("tr")
                    parent_table = th.find_parent("table")
                    if parent_tr is not None and parent_table is not None:
                        first_tr = None
                        for ch in parent_table.find_all(
                            ["tbody", "tr"], recursive=False
                        ):
                            if ch.name == "tr":
                                first_tr = ch
                                break
                            if ch.name == "tbody":
                                tr_in = ch.find("tr", recursive=False)
                                if tr_in is not None:
                                    first_tr = tr_in
                                    break
                        if first_tr is not None and first_tr is parent_tr:
                            return False
                    return True
                return False
            except Exception:
                return False

        def _extract_header_title(tr: Tag) -> str:
            th = tr.find("th")
            if th is not None:
                return _extract_text(th)
            return _extract_text(tr)

        def _iter_rows(table_like: Tag):
            for child in table_like.find_all(["tbody", "tr"], recursive=False):
                if child.name == "tr":
                    yield child
                elif child.name == "tbody":
                    for tr in child.find_all("tr", recursive=False):
                        yield tr

        def _has_any_class(tag: Tag, class_names: set) -> bool:
            try:
                cls = tag.get("class") or []
                cls_list = cls if isinstance(cls, list) else str(cls).split()
                cls_list = [c.lower() for c in cls_list]
                return any(c in class_names for c in cls_list)
            except Exception:
                return False

        def _fallback_table_title(table_like: Tag) -> str:
            cap = table_like.find("caption")
            if cap:
                t = _extract_text(cap)
                if t:
                    return t
            first_tr = None
            for ch in table_like.find_all(["tbody", "tr"], recursive=False):
                if ch.name == "tr":
                    first_tr = ch
                    break
                if ch.name == "tbody":
                    tr_in = ch.find("tr", recursive=False)
                    if tr_in is not None:
                        first_tr = tr_in
                        break
            if first_tr is not None:
                th = first_tr.find("th")
                if th is not None:
                    return _extract_text(th)
            anc_table = table_like.find_parent("table")
            if anc_table is not None:
                outer_tr = None
                for tr in table_like.find_parents("tr"):
                    if tr.find_parent("table") is anc_table:
                        outer_tr = tr
                        break
                if outer_tr is not None:
                    for prev_tr in outer_tr.find_previous_siblings("tr"):
                        ths_prev = prev_tr.find_all("th", recursive=False)
                        # Accept any th as a header for nested block titles
                        if ths_prev:
                            ttxt = _extract_header_title(prev_tr)
                            if ttxt:
                                return ttxt
            return ""

        results: List[Dict[str, Any]] = []
        ingested_tables: set[int] = set()
        for table in stew.find_all("table"):
            if id(table) in ingested_tables:
                continue
            if not _is_infobox(table):
                continue

            _results_len_before = len(results)

            if not table.find_all("th"):
                title_fallback = _fallback_table_title(table)
                data_no_th: Dict[str, Any] = {}
                for tr in _iter_rows(table):
                    cells = tr.find_all("td", recursive=False)
                    if len(cells) < 2:
                        continue
                    if _has_any_class(tr, {"infobox-below", "noprint"}) or any(
                        _has_any_class(c, {"infobox-below", "noprint"}) for c in cells
                    ):
                        continue
                    key = _clean_key(_extract_text(cells[0]))
                    vals: List[str] = []
                    for c in cells[1:]:
                        txt = _extract_text(c)
                        if txt:
                            vals.append(txt)
                    value = _clean_value_text(" · ".join(vals))
                    _assign_kv(data_no_th, key, value)
                if data_no_th:
                    results.append(
                        {
                            "title": title_fallback,
                            "data": data_no_th,
                        }
                    )
                continue

            current_title: str = ""
            current_data: Dict[str, Any] = {}
            encountered_header: bool = False

            for tr in _iter_rows(table):
                ths = tr.find_all("th", recursive=False)
                tds = tr.find_all("td", recursive=False)
                if _has_any_class(tr, {"infobox-below", "noprint"}) or any(
                    _has_any_class(c, {"infobox-below", "noprint"}) for c in (ths + tds)
                ):
                    continue
                if any(_is_header_th(th) for th in ths):
                    encountered_header = True
                    if current_data:
                        title_to_use = (
                            current_title
                            if current_title
                            else _fallback_table_title(table)
                        )
                        if title_to_use:
                            results.append(
                                {"title": title_to_use, "data": current_data}
                            )
                    current_title = _extract_header_title(tr)
                    current_data = {}
                    continue

                cells = tr.find_all(["th", "td"], recursive=False)
                if len(cells) == 0:
                    continue

                if len(cells) == 1:
                    only = cells[0]
                    inner_tables = only.find_all("table")
                    if inner_tables:
                        # Extract key/value pairs from the first inner table and mark it to skip later
                        inner = inner_tables[0]
                        ingested_tables.add(id(inner))
                        for inner_tr in _iter_rows(inner):
                            inner_cells = inner_tr.find_all(
                                ["td", "th"], recursive=False
                            )
                            if len(inner_cells) < 2:
                                continue
                            k = _clean_key(_extract_text(inner_cells[0]))
                            vals_inner: List[str] = []
                            for c in inner_cells[1:]:
                                tv = _extract_text(c)
                                if tv:
                                    vals_inner.append(tv)
                            v = _clean_value_text(" · ".join(vals_inner))
                            _assign_kv(current_data, k, v)
                        continue
                    single_val = _extract_text(only)
                    if not single_val:
                        continue
                    _assign_kv(current_data, current_title, single_val)
                    continue

                label_cell = None
                # Prefer explicit label cells: th/td with scope=row or class contains infobox-label
                for th in ths:
                    if th.get("scope", "").lower() == "row" or "infobox-label" in [
                        c.lower() for c in _classes(th)
                    ]:
                        label_cell = th
                        break
                if label_cell is None:
                    for td in tds:
                        if "infobox-label" in [c.lower() for c in _classes(td)]:
                            label_cell = td
                            break
                if label_cell is None:
                    label_cell = cells[0]
                value_cells = cells[1:]

                key = _clean_key(_extract_text(label_cell))
                values: List[str] = []
                for td in value_cells:
                    txt = _extract_text(td)
                    if txt:
                        values.append(txt)
                value = _clean_value_text(" · ".join(values))
                _assign_kv(current_data, key, value)

            if current_title and current_data:
                results.append({"title": current_title, "data": current_data})

            if not encountered_header and current_data:
                title_fallback = _fallback_table_title(table)
                results.append({"title": title_fallback, "data": current_data})

            # Fallback: if nothing was appended for this infobox table, try a simple 2-column extraction
            if len(results) == _results_len_before:
                try:
                    simple_data: Dict[str, Any] = {}
                    for tr in table.find_all("tr", recursive=False):
                        cells = tr.find_all(["td", "th"], recursive=False)
                        if len(cells) < 2:
                            continue
                        # Skip header rows that span both columns
                        try:
                            colspan_sum = sum(
                                max(1, int(c.get("colspan", 1))) for c in cells
                            )
                            if colspan_sum <= 1:
                                continue
                            if len(cells) == 1:
                                continue
                        except Exception:
                            pass
                        key_cell = cells[0]
                        val_cell = cells[1]
                        k = _clean_key(_extract_text(key_cell))
                        v = _clean_value_text(_extract_text(val_cell))
                        if k and v:
                            _assign_kv(simple_data, k, v)
                    if simple_data:
                        tt = _fallback_table_title(table)
                        results.append({"title": tt, "data": simple_data})
                except Exception:
                    pass

        return results

    # --- Existing helpers (unchanged) ---
    def _extract_cell_text_with_filter(self, cell, exclude_elements=None) -> str:
        import typing
        from bs4.element import Tag

        try:

            def _style_has_display_none(tag: Tag) -> bool:
                try:
                    style = tag.get("style") or ""
                    style_compact = "".join(str(style).split()).lower()
                    return "display:none" in style_compact
                except Exception:
                    return False

            def _is_hidden(node) -> bool:
                t = node if hasattr(node, "attrs") else getattr(node, "parent", None)
                while t is not None and t is not cell:
                    if hasattr(t, "attrs") and _style_has_display_none(t):
                        return True
                    t = getattr(t, "parent", None)
                return hasattr(node, "attrs") and _style_has_display_none(node)

            visible_imgs = [im for im in cell.find_all("img") if not _is_hidden(im)]
            if visible_imgs:
                visible_text_nodes = []
                for s in cell.find_all(string=True):
                    if not _is_hidden(s.parent):
                        visible_text_nodes.append(str(s))
                normalized = " ".join(
                    "".join(visible_text_nodes)
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .replace("\xa0", " ")
                    .split()
                )
                if not normalized:
                    alts = [
                        im.get("alt")
                        for im in visible_imgs
                        if (im.get("alt") or "").strip()
                    ]
                    if alts:
                        return " · ".join(alt.strip() for alt in alts)
            parts: typing.List[str] = []
            last_token: str = ""
            for (
                node_text,
                _is_transcluded,
                element_types,
                _para_context,
            ) in self.custom_html_to_plaintext(cell):
                if exclude_elements and exclude_elements.intersection(element_types):
                    continue
                token = " " if node_text == "\n" else node_text
                if token == last_token and token.strip():
                    continue
                parts.append(token)
                if token.strip():
                    last_token = token
            text = " ".join(
                "".join(parts)
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\xa0", " ")
                .split()
            )
            return text if text else cell.get_text(separator=" ", strip=True)
        except Exception:
            return cell.get_text(separator=" ", strip=True)

    def table_to_markdown(
        self, table, include_header: bool = True, exclude_elements=None
    ) -> str:
        import typing
        from bs4.element import Tag

        assert isinstance(table, Tag) and table.name == "table", (
            "Expected a <table> Tag"
        )
        inner_tables_in_cells: typing.List[Tag] = []
        direct_rows: typing.List[Tag] = []
        for child in table.find_all(["tbody", "tr"], recursive=False):
            if child.name == "tr":
                direct_rows.append(child)
            elif child.name == "tbody":
                direct_rows.extend(child.find_all("tr", recursive=False))
        for tr in direct_rows:
            for td in tr.find_all(["td", "th"], recursive=False):
                inner_tables_in_cells.extend(td.find_all("table"))
        # If the table is merely a wrapper around a single inner table (common for centering), unwrap it
        if inner_tables_in_cells and len(inner_tables_in_cells) == 1:
            try:
                only_trs = [tr for tr in direct_rows]
                if len(only_trs) == 1:
                    only_cells = [
                        c for c in only_trs[0].find_all(["td", "th"], recursive=False)
                    ]
                    if len(only_cells) == 1:
                        cell = only_cells[0]
                        inner = inner_tables_in_cells[0]
                        # Confirm the inner table is inside this cell and the cell has no other meaningful content
                        if inner in cell.find_all("table"):
                            has_non_table_content = False
                            for node in cell.contents:
                                if getattr(node, "name", None) == "table":
                                    continue
                                text = getattr(
                                    node, "get_text", lambda **_: str(node)
                                )()
                                if (text or "").strip():
                                    has_non_table_content = True
                                    break
                            if not has_non_table_content:
                                # Unwrap: render the inner table directly
                                return self.table_to_markdown(
                                    inner,
                                    include_header=include_header,
                                    exclude_elements=exclude_elements,
                                )
            except Exception:
                pass
        if inner_tables_in_cells and len(inner_tables_in_cells) >= 2:
            rendered: typing.List[str] = []
            for t in inner_tables_in_cells:
                md = self.table_to_markdown(
                    t, include_header=include_header, exclude_elements=exclude_elements
                )
                if md.strip():
                    rendered.append(md)
            return "\n\n".join(rendered)
        header_rows: typing.List[typing.List[Tag]] = []
        body_rows: typing.List[typing.List[Tag]] = []
        thead = table.find("thead")
        if thead:
            for tr in thead.find_all("tr", recursive=False):
                cells = [c for c in tr.find_all(["th", "td"], recursive=False)]
                if cells:
                    header_rows.append(cells)
        tbodies = table.find_all("tbody")
        if tbodies:
            for tb in tbodies:
                for tr in tb.find_all("tr", recursive=False):
                    cells = [c for c in tr.find_all(["th", "td"], recursive=False)]
                    if cells:
                        body_rows.append(cells)
        else:
            for tr in table.find_all("tr", recursive=False):
                cells = [c for c in tr.find_all(["th", "td"], recursive=False)]
                if cells:
                    body_rows.append(cells)
        if include_header and not header_rows and body_rows:
            while body_rows and all(c.name == "th" for c in body_rows[0]):
                header_rows.append(body_rows.pop(0))
        if include_header and header_rows:

            def _not_empty_header(row: typing.List[Tag]) -> bool:
                for c in row:
                    txt = self._extract_cell_text_with_filter(
                        c, exclude_elements=exclude_elements
                    )
                    if (txt or "").strip():
                        return True
                return False

            header_rows = [r for r in header_rows if _not_empty_header(r)]
            if header_rows and body_rows:

                def _cols(rows: typing.List[typing.List[Tag]]) -> int:
                    max_cols = 0
                    for r in rows:
                        total = 0
                        for c in r:
                            try:
                                total += max(1, int(c.get("colspan", 1)))
                            except Exception:
                                total += 1
                        if total > max_cols:
                            max_cols = total
                    return max_cols

                header_cols = _cols(header_rows)
                body_cols = _cols(body_rows)
                if abs(header_cols - body_cols) >= 2:
                    header_rows = []

        def _span_count(tag: Tag, attr: str) -> int:
            try:
                return max(1, int(tag.get(attr, 1)))
            except Exception:
                return 1

        pending_rowspans: typing.Dict[int, typing.Tuple[str, int]] = {}

        def _place_cell(
            row: typing.List[str],
            start_idx: int,
            value: str,
            colspan: int,
            rowspan: int,
        ):
            idx = start_idx
            while idx < len(row) and row[idx] != "":
                idx += 1
            for j in range(colspan):
                col = idx + j
                if col >= len(row):
                    row.extend([""] * (col - len(row) + 1))
                row[col] = value
                if rowspan > 1:
                    pending_rowspans[col] = (value, rowspan - 1)
            return idx + colspan

        def _start_row_with_rowspans() -> typing.List[str]:
            row: typing.List[str] = []
            for col in sorted(list(pending_rowspans.keys())):
                val, remain = pending_rowspans[col]
                if col >= len(row):
                    row.extend([""] * (col - len(row) + 1))
                row[col] = val
                if remain <= 1:
                    pending_rowspans.pop(col, None)
                else:
                    pending_rowspans[col] = (val, remain - 1)
            return row

        # Helper to extract text from a table cell while ignoring images/figures (prevent image ALT leakage)
        def _extract_cell_text_for_table(cell: Tag) -> str:
            try:
                from bs4 import BeautifulSoup  # type: ignore

                cell_copy = BeautifulSoup(str(cell), "lxml").find()
                if cell_copy is None:
                    return ""
                # Collect ALT texts before dropping images/wrappers
                alts: typing.List[str] = []
                for im in cell_copy.find_all("img"):
                    alt = (im.get("alt") or "").strip()
                    if alt:
                        alts.append(alt)
                # Drop images and typical wrappers around images
                for t in cell_copy.find_all(["img", "figure"]):
                    t.decompose()
                for span in cell_copy.find_all("span"):
                    ty = span.get("typeof") or ""
                    if isinstance(ty, list):
                        ty_join = " ".join(ty).lower()
                    else:
                        ty_join = str(ty).lower()
                    if "mw:file" in ty_join:
                        span.decompose()
                text_val = self._extract_cell_text_with_filter(
                    cell_copy, exclude_elements=exclude_elements
                )
                if not (text_val or "").strip() and alts:
                    return " · ".join(alts)
                return text_val
            except Exception:
                return self._extract_cell_text_with_filter(
                    cell, exclude_elements=exclude_elements
                )

        header_out_rows: typing.List[typing.List[str]] = []
        if include_header and header_rows:
            for header_row in header_rows:
                row_out = _start_row_with_rowspans()
                col_i = 0
                for cell in header_row:
                    text = _extract_cell_text_for_table(cell)
                    colspan = _span_count(cell, "colspan")
                    rowspan = _span_count(cell, "rowspan")
                    col_i = _place_cell(row_out, col_i, text, colspan, rowspan)
                header_out_rows.append(row_out)
        grid: typing.List[typing.List[str]] = []
        for tr_cells in body_rows:
            row_out = _start_row_with_rowspans()
            extracted: typing.List[typing.Tuple[Tag, str, bool]] = []
            non_empty_count = 0
            for cell in tr_cells:
                text = _extract_cell_text_for_table(cell)
                normalized = text.replace("\xa0", " ").replace("\u200b", "").strip()
                has_content = bool(normalized)
                if has_content:
                    non_empty_count += 1
                extracted.append((cell, text, has_content))
            if non_empty_count == 1:
                col_i = 0
                for cell, text, has_content in extracted:
                    cs = _span_count(cell, "colspan")
                    if has_content:
                        col_i = _place_cell(row_out, col_i, text, 1, 1)
                    else:
                        col_i += cs
            else:
                col_i = 0
                for cell, text, _has_content in extracted:
                    colspan = _span_count(cell, "colspan")
                    rowspan = _span_count(cell, "rowspan")
                    col_i = _place_cell(row_out, col_i, text, colspan, rowspan)
            grid.append(row_out)
        all_rows = header_out_rows + grid
        if not all_rows:
            return ""
        num_cols = max(len(r) for r in all_rows)
        for r in all_rows:
            if len(r) < num_cols:
                r.extend([""] * (num_cols - len(r)))

        def _normalize(s: str) -> str:
            return " ".join(s.replace("\xa0", " ").replace("\u200b", "").split())

        def _md_escape(s: str) -> str:
            s = _normalize(s).replace("\n", " ")
            return s.replace("|", "\\|")

        try:
            from wcwidth import wcswidth as _wcswidth  # type: ignore

            def _display_width(s: str) -> int:
                return max(0, _wcswidth(s))
        except Exception:

            def _display_width(s: str) -> int:
                return len(s)

        col_widths = [0] * num_cols
        for r in all_rows:
            for i, cell_text in enumerate(r):
                col_widths[i] = max(
                    col_widths[i], _display_width(_md_escape(cell_text))
                )

        def _fmt_row(row: typing.List[str]) -> str:
            padded: typing.List[str] = []
            for i, cell in enumerate(row):
                val = _md_escape(cell)
                pad = col_widths[i] - _display_width(val)
                if pad > 0:
                    val = val + (" " * pad)
                padded.append(val)
            return "| " + " | ".join(padded) + " |"

        lines: typing.List[str] = []
        if header_out_rows:
            for h in header_out_rows:
                lines.append(_fmt_row(h))
            lines.append("| " + " | ".join(["-" * w for w in col_widths]) + " |")
        for r in grid:
            lines.append(_fmt_row(r))
        return "\n".join(lines)

    def _ensure_base_href(self, html: str, metadata: dict | None) -> str:
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urlparse
        except Exception:
            return html
        try:
            soup = BeautifulSoup(html, "lxml")
            base = soup.find("base")
            href = None
            if base is None or not base.get("href"):
                host = None
                if isinstance(metadata, dict):
                    url = metadata.get("url")
                    if url:
                        try:
                            parsed = urlparse(url)
                            host = parsed.netloc or None
                        except Exception:
                            host = None
                    if not host:
                        wikiname = metadata.get("wikiname") or ""
                        if wikiname.endswith("wiki"):
                            lang = wikiname[:-4]
                            if lang:
                                host = f"{lang}.wikipedia.org"
                if (
                    not host
                    and isinstance(self.wiki, str)
                    and self.wiki.endswith("_namespace_0")
                ):
                    # e.g., "enwiki_namespace_0" -> "en"
                    basewiki = self.wiki.removesuffix("_namespace_0")
                    if basewiki.endswith("wiki"):
                        lang = basewiki[:-4]
                        if lang:
                            host = f"{lang}.wikipedia.org"
                # Final fallback to a generic host to avoid crashes inside Article()
                if not host:
                    host = "wikipedia.org"
                href = f"//{host}"
            if href:
                if soup.head is None:
                    head = soup.new_tag("head")
                    if soup.html is not None:
                        soup.html.insert(0, head)
                    else:
                        # Create minimal structure if missing
                        html_tag = soup.new_tag("html")
                        soup.insert(0, html_tag)
                        html_tag.insert(0, head)
                else:
                    head = soup.head
                new_base = soup.new_tag("base", href=href)
                head.insert(0, new_base)
                return str(soup)
        except Exception:
            return html
        return html

    def custom_html_to_plaintext(
        self,
        parent_node,
        transcluded=False,
        parent_types=None,
        para_context=None,
        exclude_elements=None,
    ):
        import typing
        import re
        from mwparserfromhtml.parse.plaintext import _tag_to_element, is_transcluded

        element = _tag_to_element(parent_node)

        def _has_display_none(tag) -> bool:
            try:
                style = tag.get("style") or ""
            except Exception:
                return False
            style_compact = "".join(str(style).split()).lower()
            return "display:none" in style_compact

        # Helper: detect Math descendants so we can make an exception for hidden wrappers around math
        def _contains_math(node) -> bool:
            try:
                return bool(getattr(node, "find", lambda *_a, **_k: None)("math"))
            except Exception:
                return False

        # Allow traversal through hidden wrappers if they contain math (standard Wikimedia pattern)
        if hasattr(parent_node, "attrs") and _has_display_none(parent_node):
            if not _contains_math(parent_node):
                return
        if parent_types is None:
            parent_types = []
        section_layer = False

        # Treat a node as a paragraph node if it is a <p> or contains a descendant <p>
        def _is_para_node(n) -> bool:
            try:
                if getattr(n, "name", None) == "p":
                    return True
                # Avoid descending into math/tables/lists for para detection
                if getattr(n, "name", None) in {"math", "table", "ul", "ol"}:
                    return False
                return hasattr(n, "find") and n.find("p") is not None
            except Exception:
                return False

        if element == "Section":
            section_layer = True
            first_para = None
            last_para = None
            # Work over contents so indices align with the iteration below
            for i, c in enumerate(parent_node.contents):
                if _is_para_node(c):
                    if first_para is None:
                        first_para = i
                    last_para = i
        if element:
            parent_types.append(element)
        # Map certain classes to logical element types (e.g., no-print / navigation / messagebox / category)
        try:
            _classes_attr = parent_node.get("class")
            _classes_list = (
                _classes_attr
                if isinstance(_classes_attr, list)
                else ((_classes_attr or "").split())
            )
            _classes_lc = [str(c).lower() for c in _classes_list]
            if "nomobile" in _classes_lc:
                parent_types.append("nomobile")
            # Treat .noprint, .fmbox, .fmbox-editnotice, and .stub as no-print containers
            if any(
                c in _classes_lc
                for c in ["noprint", "fmbox", "fmbox-editnotice", "stub"]
            ):
                parent_types.append("noprint")
            # Treat .NavFrame and .NavHead containers as Navigation
            if any(c in _classes_lc for c in ["navframe", "navhead"]):
                parent_types.append("Navigation")
            # Treat other nav-like containers as Navigation
            if any(
                c in _classes_lc
                for c in [
                    "navbox",
                    "vertical-navbox",
                    "navbar",
                    "sisterproject",
                    "sistersitebox",
                    "commonscat",
                ]
            ):
                parent_types.append("Navigation")
            # Treat messagebox-like containers as Messagebox
            if any(
                c in _classes_lc
                for c in [
                    "mbox-small",
                    "messagebox",
                    "ambox",
                    "tmbox",
                    "imbox",
                    "cmbox",
                    "pmbox",
                ]
            ):
                parent_types.append("Messagebox")
            # Treat catlinks as Category
            if "catlinks" in _classes_lc:
                parent_types.append("Category")
            # Treat authority control blocks as noprint
            if ("authority-control" in _classes_lc) or (
                "metadata" in _classes_lc and "authority-control" in _classes_lc
            ):
                parent_types.append("noprint")
            # Drop table of contents by tagging as Navigation if id is 'toc'
            _id_attr = str(parent_node.get("id") or "").lower()
            if _id_attr == "toc":
                parent_types.append("Navigation")
            # Treat id='stub' as a no-print container
            if _id_attr == "stub":
                parent_types.append("noprint")
            # Treat stub templates (mw:Transclusion with known stub names) as no-print
            try:
                _typeof_attr = str(parent_node.get("typeof") or "")
                _dmw_raw = str(parent_node.get("data-mw") or "")
                _dmw_lc = _dmw_raw.lower()
                _stub_tokens = [
                    "stub",
                    "ébauche",
                    "esbozo",
                    "esboço",
                    "esborrany",
                    "ciot",
                    "zaląż",
                    "pahýl",
                    "taslak",
                    "заготов",
                    "mrva",
                    "клица",
                    "μικρ",
                    "بذر",
                    "קצרמר",
                    "خرد",
                    "अधूर",
                    "rintisan",
                    "sơ khai",
                    "โครง",
                    "토막글",
                    "スタブ",
                    "小作品",
                ]
                if ("mw:transclusion" in _typeof_attr.lower()) and any(
                    tok in _dmw_lc for tok in _stub_tokens
                ):
                    parent_types.append("noprint")
            except Exception:
                pass
        except Exception:
            pass
        for i, cnode in enumerate(parent_node.contents):
            if hasattr(cnode, "attrs") and _has_display_none(cnode):
                # Exception: keep descending into hidden nodes that contain math
                try:
                    if not _contains_math(cnode):
                        continue
                except Exception:
                    continue
            # Avoid double-rendering: when traversing a Section, do not descend into nested Section nodes here.
            # Those nested sections will be rendered in their own pass by get_plaintext.
            if section_layer and getattr(cnode, "name", None) == "section":
                # Emit a section boundary marker only
                yield (
                    "\n",
                    transcluded or is_transcluded(cnode),
                    parent_types + ["Section"],
                    para_context,
                )
                continue
            # Drop generic banner/hatnote/disambiguation containers
            try:
                classes_attr = cnode.get("class")
                classes_list = (
                    classes_attr
                    if isinstance(classes_attr, list)
                    else ((classes_attr or "").split())
                )
                classes_lower = [str(c).lower() for c in classes_list]
                if (
                    ("hatnote" in classes_lower)
                    or ("dablink" in classes_lower)
                    or ("rellink" in classes_lower)
                    or ("homonymie" in classes_lower)
                    or any(cl.startswith("bandeau") for cl in classes_lower)
                ):
                    continue
            except Exception:
                pass
            cnode_is_para = _is_para_node(cnode)
            if section_layer:
                if first_para is None or i < first_para:
                    para_context = "pre-first-para"
                elif cnode_is_para:
                    para_context = "in-para"
                elif i <= last_para:
                    para_context = "between-paras"
                else:
                    para_context = "post-last-para"

            if cnode.name == "math":
                # Mark that we encountered math for this document
                try:
                    self._had_math = True
                except Exception:
                    pass
                yield (cnode.get("alttext"), transcluded, parent_types, para_context)
            elif cnode.name == "table":
                effective_parent_types = parent_types.copy()
                elem_type = _tag_to_element(cnode)
                if elem_type:
                    effective_parent_types.append(elem_type)
                if "nomobile" in (cnode.get("class", []) or []):
                    effective_parent_types.append("nomobile")
                if "noprint" in (cnode.get("class", []) or []):
                    effective_parent_types.append("noprint")
                # If this table is inside an infobox wrapper, mark it as Infobox to avoid rendering
                try:
                    p = cnode
                    while p is not None and hasattr(p, "get"):
                        classes = p.get("class") or []
                        cls_list = (
                            classes
                            if isinstance(classes, list)
                            else str(classes).split()
                        )
                        cls_lc = [str(c).lower() for c in cls_list]
                        if any(("infobox" in c) or ("taxobox" in c) for c in cls_lc):
                            effective_parent_types.append("Infobox")
                            break
                        p = getattr(p, "parent", None)
                except Exception:
                    pass
                # Tag navigation-like tables (navbox/sidebar) as Navigation to drop them
                try:
                    NAV_TOKENS = {
                        "navbox",
                        "vertical-navbox",
                        "navbar",
                        "sidebar",
                        "navbox-inner",
                        "navbox-list",
                        "navbox-title",
                        "navbox-subgroup",
                        "hlist",
                        "navigation-not-searchable",
                        "navframe",
                        "navhead",
                    }
                    p = cnode
                    while p is not None and hasattr(p, "get"):
                        role = (p.get("role") or "").lower()
                        if role == "navigation":
                            effective_parent_types.append("Navigation")
                            break
                        classes = p.get("class") or []
                        cls_list = (
                            classes
                            if isinstance(classes, list)
                            else str(classes).split()
                        )
                        cls_lc = [str(c).lower() for c in cls_list]
                        if any(any(tok in c for tok in NAV_TOKENS) for c in cls_lc):
                            effective_parent_types.append("Navigation")
                            break
                        p = getattr(p, "parent", None)
                    # Additional structural cues even if main navbox class is missing
                    if "Navigation" not in effective_parent_types:
                        has_navbox_title = False
                        try:
                            for th in cnode.find_all("th"):
                                cls = th.get("class") or []
                                cls_list = (
                                    cls if isinstance(cls, list) else str(cls).split()
                                )
                                cls_lc = [str(c).lower() for c in cls_list]
                                if any("navbox-title" in c for c in cls_lc):
                                    has_navbox_title = True
                                    break
                        except Exception:
                            has_navbox_title = False
                        has_nav_head = False
                        has_navbox_list = False
                        try:
                            for el in cnode.find_all(True):
                                cls = el.get("class") or []
                                cls_list = (
                                    cls if isinstance(cls, list) else str(cls).split()
                                )
                                cls_lc = [str(c).lower() for c in cls_list]
                                if any("navhead" in c for c in cls_lc):
                                    has_nav_head = True
                                if any("navbox-list" in c for c in cls_lc):
                                    has_navbox_list = True
                                if has_nav_head or has_navbox_list:
                                    break
                        except Exception:
                            pass
                        if has_navbox_title or has_nav_head or has_navbox_list:
                            effective_parent_types.append("Navigation")
                except Exception:
                    pass
                # Tag likely infobox-like fact boxes by heuristics (floated right/narrow width), to exclude downstream
                try:
                    style_val = (cnode.get("style") or "").lower()
                    width_attr = (cnode.get("width") or "").lower()
                    class_lc = [str(c).lower() for c in (cnode.get("class") or [])]
                    floated_like = (
                        ("float:right" in "".join(style_val.split()))
                        or ("float: right" in style_val)
                        or ("float-right" in class_lc)
                        or ("floatright" in class_lc)
                    )
                    narrow_like = any(
                        tok in width_attr
                        for tok in ["180", "200", "220", "240", "260", "280"]
                    ) or any(
                        tok in style_val
                        for tok in [
                            "width:180",
                            "width:200",
                            "width:220",
                            "width:240",
                            "width:260",
                            "width:280",
                        ]
                    )
                    is_wikitable = any("wikitable" in c for c in class_lc)
                    if (floated_like or narrow_like) and not is_wikitable:
                        effective_parent_types.append("Infobox")
                except Exception:
                    pass
                yield (
                    self.table_to_markdown(
                        cnode, include_header=True, exclude_elements=exclude_elements
                    ),
                    transcluded,
                    effective_parent_types,
                    para_context,
                )
            elif cnode.name == "ul" or cnode.name == "ol":
                item_index = 1
                effective_parent_types = parent_types.copy() + ["List"]
                # Tag navigation lists (hlist/navbars or inside nav containers) as Navigation
                try:
                    NAV_TOKENS = {
                        "navbox",
                        "vertical-navbox",
                        "navbar",
                        "sidebar",
                        "navbox-inner",
                        "navbox-list",
                        "navbox-title",
                        "navbox-subgroup",
                        "hlist",
                        "navigation-not-searchable",
                        "navframe",
                        "navhead",
                    }
                    p = cnode
                    while p is not None and hasattr(p, "get"):
                        role = (p.get("role") or "").lower()
                        if role == "navigation":
                            effective_parent_types.append("Navigation")
                            break
                        classes = p.get("class") or []
                        cls_list = (
                            classes
                            if isinstance(classes, list)
                            else str(classes).split()
                        )
                        cls_lc = [str(c).lower() for c in cls_list]
                        if any(any(tok in c for tok in NAV_TOKENS) for c in cls_lc):
                            effective_parent_types.append("Navigation")
                            break
                        p = getattr(p, "parent", None)
                except Exception:
                    pass
                if "nomobile" in (cnode.get("class", []) or []):
                    effective_parent_types.append("nomobile")
                if "noprint" in (cnode.get("class", []) or []):
                    effective_parent_types.append("noprint")

                # Compute indentation based on actual UL/OL ancestor depth (2 spaces per nested level)
                def _dom_list_depth(node) -> int:
                    d = 0
                    try:
                        p = node
                        while p is not None and hasattr(p, "name"):
                            if getattr(p, "name", None) in {"ul", "ol"}:
                                d += 1
                            p = getattr(p, "parent", None)
                    except Exception:
                        return 1
                    return max(1, d)

                indent_prefix = "  " * max(0, _dom_list_depth(cnode) - 1)
                for li in cnode.find_all("li", recursive=False):
                    # Buffer LI content to decide if it has any textual content before emitting a marker
                    li_tokens = list(
                        self.custom_html_to_plaintext(
                            li,
                            transcluded or is_transcluded(li),
                            effective_parent_types.copy(),
                            para_context,
                        )
                    )
                    combined = "".join((t[0] or "") for t in li_tokens)
                    has_text = bool(combined) and bool(re.search(r"\w", combined))
                    if not has_text:
                        # Skip empty/image-only list items entirely (no leading dash/number)
                        continue
                    marker = "- " if cnode.name == "ul" else f"{item_index}. "
                    yield (
                        indent_prefix + marker,
                        transcluded or is_transcluded(li),
                        effective_parent_types,
                        para_context,
                    )
                    for tok in li_tokens:
                        yield tok
                    yield (
                        "\n",
                        transcluded or is_transcluded(li),
                        effective_parent_types,
                        para_context,
                    )
                    if cnode.name == "ol":
                        item_index += 1
            elif cnode.name == "br":
                if cnode.find_parent("table") is not None:
                    yield (" · ", transcluded, parent_types, para_context)
                else:
                    yield ("\n", transcluded, parent_types, para_context)
            elif cnode.name == "div" and cnode.find_parent("caption") is not None:
                # Insert the existing separator between caption segments when empty layout <div>s are used
                if not cnode.get_text(strip=True):
                    yield (" · ", transcluded, parent_types, para_context)
                else:
                    yield from self.custom_html_to_plaintext(
                        cnode,
                        transcluded or is_transcluded(cnode),
                        parent_types.copy(),
                        para_context,
                    )
            elif cnode.name == "div" and "Citation" in (exclude_elements or set()):
                classes_attr = cnode.get("class")
                classes = (
                    classes_attr
                    if isinstance(classes_attr, list)
                    else ((classes_attr or "").split())
                )
                if "reflist" in classes or "mw-references-wrap" in classes:
                    continue
                # otherwise recurse into the div so its children (like <p>) produce tokens
                yield from self.custom_html_to_plaintext(
                    cnode,
                    transcluded or is_transcluded(cnode),
                    parent_types.copy(),
                    para_context,
                )
            elif hasattr(cnode, "attrs"):
                yield from self.custom_html_to_plaintext(
                    cnode,
                    transcluded or is_transcluded(cnode),
                    parent_types.copy(),
                    para_context,
                )
            else:
                yield (cnode.text, transcluded, parent_types, para_context)

    def extract_infoboxes_fallback(self, html: str) -> list[dict]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception:
            return []
        try:
            soup = BeautifulSoup(html, "lxml")
            tables = []
            for t in soup.find_all("table"):
                cls = [
                    str(c).lower()
                    for c in (
                        (t.get("class") or [])
                        if isinstance(t.get("class"), list)
                        else str(t.get("class") or "").split()
                    )
                ]
                if any(
                    ("infobox" in c) or ("taxobox" in c) or ("vcard" in c) for c in cls
                ):
                    tables.append(t)
            results: list[dict] = []

            def _txt(tag):
                return (
                    tag.get_text(separator=" ", strip=True) if tag is not None else ""
                )

            for table in tables:
                data: dict[str, object] = {}
                title = ""
                cap = table.find("caption")
                if cap:
                    title = _txt(cap)
                if not title:
                    first_th = table.find("th")
                    title = _txt(first_th)
                # Iterate rows including those under tbody
                for tr in table.find_all("tr"):
                    cells = tr.find_all(["th", "td"], recursive=False)
                    if not cells:
                        continue
                    # Skip full-row headers/subheaders
                    try:
                        colspan_sum = sum(
                            max(1, int(c.get("colspan", 1))) for c in cells
                        )
                        if colspan_sum <= 1 and len(cells) == 1:
                            continue
                    except Exception:
                        pass
                    # Determine label cell
                    label = None
                    for c in cells:
                        ccls = [
                            str(x).lower()
                            for x in (
                                (c.get("class") or [])
                                if isinstance(c.get("class"), list)
                                else str(c.get("class") or "").split()
                            )
                        ]
                        if (c.name == "th" and c.get("scope", "").lower() == "row") or (
                            "infobox-label" in ccls
                        ):
                            label = c
                            break
                    if label is None and cells:
                        label = cells[0]
                    values = []
                    after_label = False
                    for c in cells:
                        if not after_label:
                            if c is label:
                                after_label = True
                            continue
                        values.append(_txt(c))
                    key = _txt(label)
                    val = " · ".join(v for v in values if v)
                    if key and val:
                        if key in data:
                            existing = data[key]
                            if isinstance(existing, list):
                                if val not in existing:
                                    existing.append(val)
                            else:
                                if val != existing:
                                    data[key] = [existing, val]
                        else:
                            data[key] = val
                if data:
                    results.append({"title": title, "data": data})
            return results
        except Exception:
            return []

    def get_plaintext(
        self,
        stew,
        exclude_elements=None,
        exclude_para_context=None,
        exclude_transcluded_paragraphs=False,
        stop_after_reflist: bool = False,
    ):
        import typing
        from bs4.element import Tag

        # Sequential traversal with arbitrary heading depth support (h2-h6)
        def _heading_level(name: str) -> int:
            try:
                if name and name.startswith("h"):
                    lv = int(name[1:])
                    if 2 <= lv <= 6:
                        return lv
            except Exception:
                pass
            return 0

        def _append_from(node: Tag) -> str:
            # If this node is inside an infobox-like table, drop it entirely from body
            try:
                pchk = node
                while pchk is not None and hasattr(pchk, "get"):
                    if getattr(pchk, "name", None) == "table" and _is_nav_or_infobox(
                        pchk
                    ):
                        return ""
                    pchk = getattr(pchk, "parent", None)
            except Exception:
                pass
            # Render a specific subtree to plaintext using the existing tokenizer
            out_parts: typing.List[str] = []
            last_para = None
            for t_text, t_transcluded, t_types, t_para in self.custom_html_to_plaintext(
                node, exclude_elements=exclude_elements
            ):
                # Ensure global drop of hidden/non-content classes
                if t_types and {"noprint", "Navigation", "Infobox"}.intersection(
                    set(t_types)
                ):
                    continue
                if exclude_elements and exclude_elements.intersection(t_types):
                    continue
                if exclude_para_context and t_para in (exclude_para_context or set()):
                    continue
                out_parts.append(t_text)
                last_para = t_para
            return "".join(out_parts)

        def _is_in_noprint(tag: Tag) -> bool:
            try:
                p = tag
                while p is not None and hasattr(p, "get"):
                    classes = p.get("class") or []
                    cls_list = (
                        classes if isinstance(classes, list) else str(classes).split()
                    )
                    cls_lc = [str(c).lower() for c in cls_list]
                    # Treat these containers as non-content blocks to be dropped
                    if (
                        ("noprint" in cls_lc)
                        or ("toccolours" in cls_lc)
                        or ("fmbox" in cls_lc)
                        or ("fmbox-editnotice" in cls_lc)
                        or ("stub" in cls_lc)
                        or ("mbox-small" in cls_lc)
                        or ("messagebox" in cls_lc)
                        or ("ambox" in cls_lc)
                        or ("tmbox" in cls_lc)
                        or ("imbox" in cls_lc)
                        or ("cmbox" in cls_lc)
                        or ("pmbox" in cls_lc)
                        or ("navframe" in cls_lc)
                        or ("navhead" in cls_lc)
                        or ("navbox" in cls_lc)
                        or ("vertical-navbox" in cls_lc)
                        or ("navbar" in cls_lc)
                        or ("sisterproject" in cls_lc)
                        or ("sistersitebox" in cls_lc)
                        or ("commonscat" in cls_lc)
                        or ("catlinks" in cls_lc)
                        or ("authority-control" in cls_lc)
                        or (str((p.get("id") or "")).lower() == "toc")
                        or (str((p.get("id") or "")).lower() == "stub")
                    ):
                        return True
                    # Also detect stub templates via mw:Transclusion + data-mw substrings
                    try:
                        _typeof_attr = str(p.get("typeof") or "")
                        _dmw_raw = str(p.get("data-mw") or "")
                        _dmw_lc = _dmw_raw.lower()
                        _stub_tokens = [
                            "stub",
                            "ébauche",
                            "esbozo",
                            "esboço",
                            "esborrany",
                            "ciot",
                            "zaląż",
                            "pahýl",
                            "taslak",
                            "заготов",
                            "mrva",
                            "клица",
                            "μικρ",
                            "بذر",
                            "קצרמר",
                            "خرد",
                            "अधूर",
                            "rintisan",
                            "sơ khai",
                            "โครง",
                            "토막글",
                            "スタブ",
                            "小作品",
                        ]
                        if ("mw:transclusion" in _typeof_attr.lower()) and any(
                            tok in _dmw_lc for tok in _stub_tokens
                        ):
                            return True
                    except Exception:
                        pass
                    p = getattr(p, "parent", None)
            except Exception:
                return False
            return False

        def _is_nav_or_infobox(table_tag: Tag) -> bool:
            # Check infobox-like by class on self or ancestors
            try:
                p = table_tag
                while p is not None and hasattr(p, "get"):
                    classes = p.get("class") or []
                    cls_list = (
                        classes if isinstance(classes, list) else str(classes).split()
                    )
                    cls_lc = [str(c).lower() for c in cls_list]
                    if any(
                        ("infobox" in c)
                        or ("taxobox" in c)
                        or ("sinottico" in c)
                        or ("vcard" in c)
                        or ("infocaseta" in c)
                        for c in cls_lc
                    ):
                        return True
                    p = getattr(p, "parent", None)
            except Exception:
                pass
            # Navigation-like by role/classes
            try:
                NAV_TOKENS = {
                    "navbox",
                    "vertical-navbox",
                    "navbar",
                    "sidebar",
                    "navbox-inner",
                    "navbox-list",
                    "navbox-title",
                    "navbox-subgroup",
                    "hlist",
                    "navigation-not-searchable",
                    "navframe",
                    "navhead",
                }
                p = table_tag
                while p is not None and hasattr(p, "get"):
                    role = (p.get("role") or "").lower()
                    if role == "navigation":
                        return True
                    classes = p.get("class") or []
                    cls_list = (
                        classes if isinstance(classes, list) else str(classes).split()
                    )
                    cls_lc = [str(c).lower() for c in cls_list]
                    if any(any(tok in c for tok in NAV_TOKENS) for c in cls_lc):
                        return True
                    p = getattr(p, "parent", None)
                # Additional structural cues even if main navbox class is missing
                try:
                    has_navbox_title = False
                    for th in table_tag.find_all("th"):
                        cls = th.get("class") or []
                        cls_list = cls if isinstance(cls, list) else str(cls).split()
                        cls_lc = [str(c).lower() for c in cls_list]
                        if any("navbox-title" in c for c in cls_lc):
                            has_navbox_title = True
                            break
                    if has_navbox_title:
                        return True
                    for el in table_tag.find_all(True):
                        cls = el.get("class") or []
                        cls_list = cls if isinstance(cls, list) else str(cls).split()
                        cls_lc = [str(c).lower() for c in cls_list]
                        if any(
                            ("navhead" in c) or ("navbox-list" in c) for c in cls_lc
                        ):
                            return True
                except Exception:
                    pass
            except Exception:
                pass
            # Heuristic: floated/narrow/right-aligned fact boxes (not wikitable)
            try:
                style_val = (table_tag.get("style") or "").lower()
                width_attr = (table_tag.get("width") or "").lower()
                align_attr = (table_tag.get("align") or "").lower()
                class_lc = [str(c).lower() for c in (table_tag.get("class") or [])]
                floated_like = (
                    ("float:right" in "".join(style_val.split()))
                    or ("float: right" in style_val)
                    or ("float-right" in class_lc)
                    or ("floatright" in class_lc)
                )
                narrow_like = any(
                    tok in width_attr
                    for tok in ["180", "200", "220", "240", "260", "280"]
                ) or any(
                    tok in style_val
                    for tok in [
                        "width:180",
                        "width:200",
                        "width:220",
                        "width:240",
                        "width:260",
                        "width:280",
                    ]
                )
                right_aligned = align_attr == "right"
                is_wikitable = any("wikitable" in c for c in class_lc)
                if (floated_like or narrow_like or right_aligned) and not is_wikitable:
                    return True
            except Exception:
                pass
            return False

        def _render_table(table_tag: Tag) -> str:
            # Skip infobox/navigation tables entirely
            if _is_nav_or_infobox(table_tag):
                return ""
            try:
                return self.table_to_markdown(
                    table_tag, include_header=True, exclude_elements=exclude_elements
                )
            except Exception:
                return _append_from(table_tag)

        def _render_pre(pre_tag: Tag) -> str:
            try:
                text_val = pre_tag.get_text(separator="\n", strip=False)
                text_val = text_val if text_val is not None else ""
                return "```\n" + text_val + "\n```\n"
            except Exception:
                try:
                    return (
                        "```\n" + (pre_tag.get_text(separator="\n") or "") + "\n```\n"
                    )
                except Exception:
                    return ""

        def _render_list(list_tag: Tag) -> str:
            try:
                import re as _re

                marker_index = 1
                parts: typing.List[str] = []
                for li in list_tag.find_all("li", recursive=False):
                    # Render LI content and decide if it has textual content
                    li_tokens = list(
                        self.custom_html_to_plaintext(
                            li,
                            False,
                            [],
                            None,
                            exclude_elements=exclude_elements,
                        )
                    )
                    combined = "".join((t[0] or "") for t in li_tokens)
                    has_text = bool(combined) and bool(_re.search(r"\w", combined))
                    if not has_text:
                        continue

                    # Determine nesting depth for indentation (2 spaces per nested level beyond the first)
                    def _dom_list_depth(node) -> int:
                        d = 0
                        try:
                            p = node
                            while p is not None and hasattr(p, "name"):
                                if getattr(p, "name", None) in {"ul", "ol"}:
                                    d += 1
                                p = getattr(p, "parent", None)
                        except Exception:
                            return 1
                        return max(1, d)

                    indent = "  " * max(0, _dom_list_depth(list_tag) - 1)
                    marker = "- " if list_tag.name == "ul" else f"{marker_index}. "
                    parts.append(indent + marker + combined.strip())
                    if list_tag.name == "ol":
                        marker_index += 1
                return "\n".join(parts) + ("\n" if parts else "")
            except Exception:
                return _append_from(list_tag)

        current_heading: str = "_Lead"
        current_level: int = 2
        buffer: str = ""
        header_stack: list[tuple[str, int]] = []

        def _flush():
            nonlocal buffer
            text = buffer.strip()
            buffer = ""
            return text

        def _walk(node: Tag):
            nonlocal buffer, current_heading, current_level, header_stack
            for child in getattr(node, "contents", []) or []:
                name = getattr(child, "name", None)
                if not name:
                    continue
                # Drop any content under .noprint containers
                if _is_in_noprint(child):
                    continue
                # Drop anything whose nearest ancestor table is infobox-like
                try:
                    pchk = child
                    inside_infobox = False
                    while pchk is not None and hasattr(pchk, "get"):
                        if getattr(
                            pchk, "name", None
                        ) == "table" and _is_nav_or_infobox(pchk):
                            inside_infobox = True
                            break
                        pchk = getattr(pchk, "parent", None)
                    if inside_infobox:
                        continue
                except Exception:
                    pass
                lvl = _heading_level(name)
                if lvl:
                    # New heading encountered: close any inline buffer as a paragraph, and push heading onto stack.
                    text = _flush()
                    if text:
                        yield (current_heading, current_level, text)
                    # Pop headings at same or deeper level; they had no body
                    while header_stack and header_stack[-1][1] >= lvl:
                        header_stack.pop()
                    # Push this heading; emit later when first content under it appears
                    header_stack.append((child.get_text(strip=True) or "", lvl))
                    continue
                if name == "section":
                    # Recurse into section content
                    yield from _walk(child)
                    continue
                if name in {"table", "ul", "ol", "p", "dl", "blockquote", "pre"}:
                    if name == "table":
                        rendered = _render_table(child)
                    elif name in {"ul", "ol"}:
                        rendered = _render_list(child)
                    elif name == "pre":
                        rendered = _render_pre(child)
                    else:
                        rendered = _append_from(child)
                    if rendered.strip():
                        # If we have pending headings, emit them now in order before content
                        if header_stack:
                            # Flush any lingering inline text as its own paragraph first
                            inline_text = _flush()
                            if inline_text:
                                yield (current_heading, current_level, inline_text)
                            for htxt, hlvl in header_stack:
                                yield (htxt, hlvl, "")
                            current_heading, current_level = header_stack[-1]
                            header_stack.clear()
                        # Flush any inline buffer as its own paragraph before emitting a block
                        inline_text = _flush()
                        if inline_text:
                            yield (current_heading, current_level, inline_text)
                        # Emit this block as its own paragraph/item
                        if name in {
                            "table",
                            "ul",
                            "ol",
                            "pre",
                        } and not rendered.endswith("\n"):
                            rendered = rendered + "\n"
                        yield (current_heading, current_level, rendered)
                    continue
                if name == "div":
                    # Recurse to catch nested headings inside wrappers like .mw-heading3
                    yield from _walk(child)
                    continue
                # Fallback recurse
                yield from _walk(child)

        # Walk the entire stew
        for item in _walk(stew):
            yield item
        tail = _flush()
        if tail:
            yield (current_heading, current_level, tail)

    def process_document(self, document):
        from mwparserfromhtml import Article

        html = document.text
        # Ensure a <base> tag exists so Article can infer the wiki language
        meta = document.metadata
        html = self._ensure_base_href(html, meta if isinstance(meta, dict) else None)
        # Reset per-document flags
        self._had_math = False
        article = Article(html)
        infoboxes = self.extract_infoboxes(article.wikistew)
        if not infoboxes:
            fallback = self.extract_infoboxes_fallback(html)
            if fallback:
                infoboxes = fallback
        document.metadata["infoboxes"] = infoboxes

        plaintext = (
            "# " + article.get_title() + "\n"
            if article.wikistew.title is not None
            else ""
        )
        prev_heading = "_Lead"
        prev_level = 2
        last_was_header = False
        for heading, level, paragraph in self.get_plaintext(
            article.wikistew,
            exclude_transcluded_paragraphs=False,
            exclude_para_context=None,
            exclude_elements={
                "Heading",
                "Citation",
                "Reference",
                "Infobox",
                "Navigation",
                "noprint",
                "Messagebox",
                "Category",
                "Media-audio",
                "Media-img",
                "Media-video",
            },
        ):
            if heading in self.ref_words:
                continue
            if heading != prev_heading or level != prev_level:
                hashes = "#" * max(2, min(6, level))
                plaintext += f"\n{hashes} {heading}\n"
                prev_heading = heading
                prev_level = level
                last_was_header = True
            # Append paragraph only if it has content
            if paragraph and paragraph.strip():
                content = paragraph.lstrip() if last_was_header else paragraph
                plaintext += f"{content}\n"
                last_was_header = False
        # Expose whether math appeared in rendered content (set by tokenizer)
        try:
            document.metadata["has_math"] = bool(getattr(self, "_had_math", False))
        except Exception:
            document.metadata["has_math"] = False
        document.text = plaintext
        return document

    def run(self, data, rank=0, world_size=1):
        from datatrove.pipeline.extractors.base import ExtractorSandbox
        from loguru import logger

        self._warned_error = False
        from datatrove.data import Document

        with ExtractorSandbox(
            timeout=self.timeout,
            wamup_text=Document(text="", id="__warmup__", metadata={}),
        ) as extractor:
            for doc in data:
                self.stat_update("total")
                with self.track_time():
                    try:
                        parsed_document = extractor.process_document(
                            doc, self.process_document
                        )
                        self.stat_update("extracted")
                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning(
                            "⏰ Timeout while cleaning record text. Skipping record."
                        )
                        continue
                    except EOFError:
                        # Process died unexpectedly
                        self.stat_update("broken_process")
                        logger.warning(
                            "Process died unexpectedly, will create new process for next document"
                        )
                        continue
                    except Exception as e:
                        self.stat_update("clean_error")
                        if not self._warned_error:
                            logger.warning(
                                f'❌ Error "{e}" while cleaning record text. Skipping record. '
                                f"This message will only appear once."
                            )
                            self._warned_error = True
                        continue

                if parsed_document.text:
                    self.stat_update("forwarded")
                    self.update_doc_stats(parsed_document)
                    yield parsed_document
                else:
                    self.stat_update("dropped")


if __name__ == "__main__":
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.io import get_datafolder

    print(get_datafolder(GCP_RAW_PREFIX))
    wikis = [
        wiki
        for wiki in get_datafolder(GCP_RAW_PREFIX).ls("", detail=False)
        if wiki.removesuffix("_namespace_0").endswith("wiki")
    ]
    print(wikis)
    from datatrove.executor.slurm import SlurmPipelineExecutor

    for wiki in wikis:
        wiki_df = get_datafolder(GCP_RAW_PREFIX + "/" + wiki)
        print(f"{GCP_RAW_PREFIX}/{wiki}")
        # Use underlying filesystem to avoid DirFileSystem path issues
        files = len(wiki_df.fs.ls(wiki_df.path, detail=False))
        SlurmPipelineExecutor(
            pipeline=[
                WikipediaReader(wiki),
                WikipediaParser(wiki),
                JsonlWriter(f"{GCP_PARSED_PREFIX}{wiki}"),
            ],
            tasks=files,
            time=SLURM_TIME,
            partition=SLURM_PARTITION,
            cpus_per_task=SLURM_CPUS_PER_TASK,
            job_name=f"wkp_{wiki}",
            qos=SLURM_QOS,
            logging_dir=str(LOGGING_DIR / wiki),
            sbatch_args={
                "mem-per-cpu": SLURM_MEM_PER_CPU,
            },
        ).run()
