from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz

STAGES = ["establishment", "tillering", "stem_booting", "reproductive", "grain_filling", "ripening"]
HEADING_MARK = "@@H@@"
TOPIC_WORDS = {
    "vi": {
        "water_management": ["quản lý nước", "mực nước", "tưới", "tháo nước", "rút nước", "ngập", "khô ướt", "giữ nước", "tiêu nước"],
        "nutrient_management": ["bón phân", "dinh dưỡng", "phân đạm", "phân lân", "phân kali", "urea", "urê", "npk", "kcl", "dap"],
        "pest_risk": ["sâu hại", "rầy", "sâu cuốn lá", "sâu đục thân", "bọ trĩ", "nhện gié", "ốc bươu", "côn trùng", "ipm"],
        "disease_risk": ["bệnh hại", "đạo ôn", "bạc lá", "đốm vằn", "lem lép", "cháy bìa", "nấm bệnh", "vi khuẩn"],
        "weed_management": ["cỏ dại", "lúa cỏ", "làm cỏ", "khử lẫn", "thuốc cỏ"],
        "harvest_readiness": ["thu hoạch", "chín vàng", "gặt", "sấy", "bảo quản", "ẩm độ", "xay xát"],
        "residue_management": ["rơm rạ", "gốc rạ", "xử lý rạ", "thu gom rơm", "đốt rơm"],
        "climate_adaptation": ["biến đổi khí hậu", "phát thải", "khí nhà kính", "chịu hạn", "chịu mặn", "thích ứng"],
        "general_crop_care": ["làm đất", "chuẩn bị giống", "gieo sạ", "cấy", "mạ", "mật độ", "thời vụ", "canh tác"],
    },
    "en": {
        "water_management": ["water management", "water level", "irrigation", "drain", "flooding", "alternate wetting", "standing water", "puddling"],
        "nutrient_management": ["fertilizer", "fertiliser", "nutrient management", "nitrogen", "phosphorus", "potassium", "npk", "urea", "site-specific nutrient management", "leaf color chart", "topdress"],
        "pest_risk": ["insect", "pest management", "rat", "rats", "rodent", "golden apple snail", "nematode", "stem borer", "planthopper", "natural enemy", "natural enemies"],
        "disease_risk": ["disease", "blast", "bacterial blight", "sheath blight", "tungro", "fungus", "fungal"],
        "weed_management": ["weed", "weed management", "herbicide", "weedy rice"],
        "harvest_readiness": ["harvest", "harvesting", "moisture content", "maturity", "grain moisture"],
        "residue_management": ["straw", "residue", "rice straw", "stubble", "husk", "rice husk", "bran", "rice bran", "by-product", "broken rice", "milling"],
        "climate_adaptation": ["climate change", "greenhouse gas", "emission", "stress tolerant", "drought tolerant", "salinity tolerant", "climate-smart"],
        "general_crop_care": ["land preparation", "seed quality", "seedling", "crop establishment", "planting", "crop calendar", "transplanting", "direct seeding", "rice varieties", "cropping season"],
    },
}
STAGE_WORDS = {
    "vi": {
        0: ["nảy mầm", "thời kỳ mạ", "cây mạ", "gieo sạ", "xuống giống", "cấy"],
        1: ["đẻ nhánh", "nở bụi", "dưỡng chồi", "chồi tối đa"],
        2: ["làm đòng", "đón đòng", "dưỡng đòng", "phân hóa đòng", "đòng ói", "chuẩn bị trổ"],
        3: ["trổ", "trỗ", "ra hoa", "thụ phấn", "thụ tinh"],
        4: ["nuôi hạt", "chắc hạt", "vào gạo", "chín sữa", "ngậm sữa", "cong trái me"],
        5: ["chín vàng", "thu hoạch", "sấy lúa", "bảo quản lúa", "độ ẩm 14%"],
    },
    "en": {
        0: ["germination", "seedling stage", "nursery", "seedbed", "land preparation", "sowing", "direct seeding", "transplanting", "crop establishment"],
        1: ["tillering", "tiller number", "maximum tillering"],
        2: ["panicle initiation", "booting", "stem elongation", "panicle development"],
        3: ["flowering", "heading", "pollination", "anthesis"],
        4: ["grain filling", "milk stage", "dough stage", "grain development"],
        5: ["ripening", "maturity", "grain moisture", "drying", "storage", "milling"],
    },
}
FACET_WORDS = {
    "vi": {
        "fertilizer": ["bón phân", "phân bón", "phân đạm", "phân lân", "phân kali", "liều lượng phân", "npk", "urea", "urê", "dap", "kcl", "dinh dưỡng"],
        "conditions": ["điều kiện", "thời tiết", "nhiệt độ", "độ ẩm", "mực nước", "ánh sáng", "khí hậu", "ngập", "hạn", "mặn"],
        "pest_disease_prevention": ["sâu hại", "rầy", "sâu cuốn lá", "sâu đục thân", "bọ trĩ", "nhện gié", "ốc bươu", "bệnh hại", "đạo ôn", "bạc lá", "đốm vằn", "lem lép", "phòng trừ", "phòng chống", "phòng ngừa", "ipm"],
        "next_stage_action": ["chuẩn bị cho giai đoạn", "trước khi bước sang", "chuẩn bị bước vào", "cần chuẩn bị", "chuẩn bị trổ", "chuẩn bị thu hoạch", "bước tiếp theo"],
    },
    "en": {
        "fertilizer": ["fertilizer", "fertiliser", "nutrient", "nitrogen", "phosphorus", "potassium", "npk", "urea", "topdress", "basal application", "nutrient management"],
        "conditions": ["condition", "weather", "temperature", "humidity", "water level", "soil", "climate", "flood", "drought", "salinity", "rainfall"],
        "pest_disease_prevention": ["pest", "insect", "disease", "rat", "rats", "rodent", "snail", "nematode", "bird", "prevent", "control", "resistant variety", "resistant varieties", "ipm", "natural enemy", "natural enemies"],
        "next_stage_action": ["prepare for", "get ready for", "before the next stage", "next step", "in preparation for", "ahead of harvest", "prior to planting", "before harvest", "before planting", "well before", "at least", "days before", "in advance"],
    },
}
REPLACEMENTS = {"ƣ": "ư", "Ƣ": "Ư", "nƣớc": "nước", "tƣới": "tưới", "lƣợng": "lượng", "trƣớc": "trước", "dƣỡng": "dưỡng", "hƣớng": "hướng", "sinh trƣởng": "sinh trưởng", "môi trƣờng": "môi trường", "phƣơng": "phương", "đƣợc": "được"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"([A-Za-zÀ-ỹ])-\s*\n\s*([A-Za-zÀ-ỹ])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def selected(ranges: list[list[int]], page: int) -> bool:
    return any(start <= page <= end for start, end in ranges)


def heading(line: str) -> bool:
    if line.startswith(HEADING_MARK):
        return True
    letters = [c for c in line if c.isalpha()]
    return bool(re.match(r"^(?:[IVX]+\.|\d+(?:\.\d+){0,3}[.)])\s+", line)) or (len(letters) >= 7 and len(line) < 150 and sum(c.isupper() for c in letters) / len(letters) > 0.83)


def clean_lines(text: str) -> list[str]:
    rows = []
    for raw in text.splitlines():
        line = normalize(raw).strip(" •▪●")
        low = line.lower()
        if not line or re.fullmatch(r"[ivxlcdm\d]+", line, re.I):
            continue
        if low.startswith(("hình ", "bảng ", "tạp chí khoa học và công nghệ nông nghiệp")):
            continue
        if "tài liệu tập huấn khuyến nông canh tác lúa và cà phê" in low:
            continue
        rows.append(line)
    return rows


def semantic_units(lines: list[str], min_words: int = 45, max_words: int = 190) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    current_heading, buffer = "", []

    def flush() -> None:
        nonlocal buffer
        text = " ".join(buffer)
        if len(text.split()) >= 28:
            result.append((current_heading, text))
        buffer = []

    for line in lines:
        if heading(line):
            flush()
            current_heading = line[len(HEADING_MARK):] if line.startswith(HEADING_MARK) else line
            continue
        if re.match(r"^(?:[-+]|\d+[.)]|[a-zđ][.)])\s+", line, re.I) and len(" ".join(buffer).split()) >= 28:
            flush()
        if len(" ".join(buffer + [line]).split()) > max_words:
            flush()
        buffer.append(line)
        if len(" ".join(buffer).split()) >= min_words and line.endswith((".", ";")):
            flush()
    flush()
    return result


def count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(r"(?<![\wÀ-ỹ])" + re.escape(phrase) + r"(?![\wÀ-ỹ])", text.lower()))


def _keyword_score(joined: str, section_lower: str, words: list[str], language: str) -> int:
    if language == "en":
        return sum(count_phrase(joined, word) * (3 if count_phrase(section_lower, word) else 1) for word in words)
    return sum(joined.count(word) * (3 if word in section_lower else 1) for word in words)


def choose_topic(section: str, text: str, language: str = "vi") -> str | None:
    joined = f"{section} {text}".lower()
    section_lower = section.lower()
    scores = {
        name: _keyword_score(joined, section_lower, words, language)
        for name, words in TOPIC_WORDS[language].items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score else None


ESTABLISHMENT_HINTS = {
    "vi": ["làm đất", "gieo sạ", "cấy", "cây mạ"],
    "en": ["land preparation", "seedling", "transplanting", "direct seeding"],
}


def stage_vector(section: str, text: str, topic: str, language: str = "vi") -> list[float]:
    stage_words = STAGE_WORDS[language]
    scores = [
        sum(count_phrase(text, word) + 3 * count_phrase(section, word) for word in words)
        for words in stage_words.values()
    ]
    if topic == "harvest_readiness":
        scores[5] += 4
    if topic == "residue_management":
        scores[5] += 3
    if topic == "weed_management":
        scores[0] += 2
        scores[1] += 2
    if topic == "general_crop_care" and any(word in text.lower() for word in ESTABLISHMENT_HINTS[language]):
        scores[0] += 3
    if max(scores) == 0:
        return [0.35] * 6
    main = max(range(6), key=lambda index: scores[index])
    vector = [0.0] * 6
    vector[main] = 1.0
    if main > 0:
        vector[main - 1] = 0.55
    if main < 5:
        vector[main + 1] = 0.55
    combined = f"{section} {text}"
    for index, words in stage_words.items():
        if any(count_phrase(combined, word) for word in words):
            vector[index] = max(vector[index], 0.85)
    vector[main] = 1.0
    return vector


def choose_facet(section: str, text: str, language: str = "vi") -> str:
    joined = f"{section} {text}".lower()
    section_lower = section.lower()
    scores = {
        name: _keyword_score(joined, section_lower, words, language)
        for name, words in FACET_WORDS[language].items()
    }
    facet, score = max(scores.items(), key=lambda item: item[1])
    return facet if score else "general"


def is_restricted(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in [
        "thuốc bảo vệ thực vật", "phun thuốc", "liều lượng", "ml/ha", "lít/ha",
        "tên sản phẩm", "hoạt chất",
    ])


def quality(text: str) -> bool:
    words = text.split()
    return (
        28 <= len(words) <= 260
        and len({word.lower() for word in words}) >= 16
        and not any(term in text.lower() for term in ["mục lục", "tài liệu tham khảo", "lời giới thiệu", "địa chỉ email"])
    )


def focused(item: dict[str, Any]) -> list[dict[str, Any]]:
    language = item.get("language", "vi")
    output = []
    for topic in ["weed_management", "pest_risk", "disease_risk"]:
        keywords = TOPIC_WORDS[language][topic]
        if item["topic"] == topic or not any(word in item["text"].lower() for word in keywords):
            continue
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", item["text"]) if sentence.strip()]
        indexes = [index for index, sentence in enumerate(sentences) if any(word in sentence.lower() for word in keywords)]
        selected_sentences = []
        for index in indexes:
            selected_sentences.extend(sentences[max(0, index - 1): index + 2])
        snippet = normalize(" ".join(dict.fromkeys(selected_sentences)))
        if quality(snippet):
            clone = dict(item)
            clone["text"] = snippet
            clone["topic"] = topic
            clone["stage_compatibility"] = stage_vector(clone["section"], snippet, topic, language)
            clone["facet"] = choose_facet(clone["section"], snippet, language)
            clone["derived_focus"] = True
            output.append(clone)
    return output


class _ContentExtractor(HTMLParser):
    BLOCK_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "select", "noscript", "iframe", "button"}
    BLOCK_CLASS_TOKENS = {
        "menu", "menu-dropdown", "breadcrumb", "breadcrumbs", "footerbox", "moduletable", "smartsearch",
        "module-title", "block-header", "mod-box-grey", "rl_tabs-tab", "nn_tabs-tab", "nav-tabs",
        "teaser-item", "social", "share", "comment", "comments",
    }
    HEADING_TAGS = {"h1", "h2", "h3", "h4"}
    TEXT_TAGS = {"p", "li", "td", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.skip_depth = 0
        self.lines: list[str] = []
        self.buffer: list[str] = []
        self.in_heading = False
        self.in_text = False

    def _is_blocked(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in self.BLOCK_TAGS:
            return True
        attr_dict = dict(attrs)
        tokens: set[str] = set()
        for key in ("class", "id"):
            tokens.update((attr_dict.get(key) or "").lower().split())
        return bool(tokens & self.BLOCK_CLASS_TOKENS)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        this_skip = self.skip_depth > 0 or self._is_blocked(tag, attrs)
        if this_skip:
            self.skip_depth += 1
        self.stack.append((tag, this_skip))
        if self.skip_depth > 0:
            return
        if tag in self.HEADING_TAGS:
            self._flush("text")
            self.in_heading = True
        elif tag in self.TEXT_TAGS:
            self.in_text = True
        elif tag == "br":
            self.buffer.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        this_skip = False
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                this_skip = self.stack[index][1]
                del self.stack[index]
                break
        if this_skip:
            self.skip_depth -= 1
            return
        if self.skip_depth > 0:
            return
        if tag in self.HEADING_TAGS and self.in_heading:
            self._flush("heading")
            self.in_heading = False
        elif tag in self.TEXT_TAGS and self.in_text:
            self._flush("text")
            self.in_text = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        if self.in_heading or self.in_text:
            self.buffer.append(data)

    def _flush(self, kind: str) -> None:
        text = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
        self.buffer = []
        if not text:
            return
        self.lines.append((HEADING_MARK + text) if kind == "heading" else text)


def extract_html_sections(html_text: str) -> list[str]:
    parser = _ContentExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.lines


def build_web(pages_root: Path, registry_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = json.loads(registry_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for source in source_rows:
        path = pages_root / f"{source['source_id']}.html"
        source = dict(source)
        source["status"] = "available" if path.exists() else "missing"
        source["chunk_count"] = 0
        registry.append(source)
        if not path.exists():
            continue
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        candidates = []
        raw_lines = "\n".join(extract_html_sections(html_text))
        for section, text in semantic_units(clean_lines(raw_lines)):
            text = normalize(text)
            if not quality(text):
                continue
            topic = choose_topic(section, text, language="en")
            if not topic:
                continue
            item = {
                "chunk_id": "",
                "source_id": source["source_id"],
                "text": text,
                "topic": topic,
                "stage_compatibility": stage_vector(section, text, topic, language="en"),
                "facet": choose_facet(section, text, language="en"),
                "authority_score": source["authority_score"],
                "review_status": "machine_curated_pending_domain_review",
                "production_eligible": False,
                "restricted_action": is_restricted(text),
                "evidence_type": "guideline",
                "section": section,
                "page_start": None,
                "page_end": None,
                "source_title": source["title"],
                "source_organization": source["organization"],
                "source_year": None,
                "source_url": "http://knowledgebank.irri.org" + source["url"],
                "source_type": source["source_type"],
                "region": [],
                "varieties": ["general"],
                "use_mode": source["use_mode"],
                "language": "en",
            }
            candidates.append(item)
            candidates.extend(focused(item))
        seen, kept = set(), []
        for item in candidates:
            key = item["topic"] + re.sub(r"\W+", "", item["text"].lower())[:1000]
            digest = hashlib.sha1(key.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            kept.append(item)
        for index, item in enumerate(kept, start=1):
            item["chunk_id"] = f"{source['source_id']}_C{index:04d}"
            chunks.append(item)
        source["chunk_count"] = len(kept)
    return chunks, registry


def build(source_root: Path, registry_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = json.loads(registry_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for source in source_rows:
        path = source_root / source["file_name"]
        source = dict(source)
        source["status"] = "available" if path.exists() else "missing"
        source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        source["chunk_count"] = 0
        registry.append(source)
        if not path.exists() or not source["page_ranges"]:
            continue
        document, candidates = fitz.open(path), []
        for page_number in range(1, document.page_count + 1):
            if not selected(source["page_ranges"], page_number):
                continue
            page_text = document[page_number - 1].get_text("text")
            for section, text in semantic_units(clean_lines(page_text)):
                text = normalize(text)
                if any(term.lower() in f"{section} {text}".lower() for term in source.get("exclude_terms", [])) or not quality(text):
                    continue
                topic = choose_topic(section, text)
                if not topic:
                    continue
                evidence_type = (
                    "research_result" if source["use_mode"] == "research_evidence_only"
                    else "variety_specific_guidance" if source["use_mode"] == "variety_specific_research_only"
                    else "guideline"
                )
                item = {
                    "chunk_id": "",
                    "source_id": source["source_id"],
                    "text": text,
                    "topic": topic,
                    "stage_compatibility": stage_vector(section, text, topic),
                    "facet": choose_facet(section, text),
                    "authority_score": source["authority_score"],
                    "review_status": "machine_curated_pending_domain_review",
                    "production_eligible": False,
                    "restricted_action": is_restricted(text),
                    "evidence_type": evidence_type,
                    "section": section,
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_title": source["title"],
                    "source_organization": source["organization"],
                    "source_year": source["year"],
                    "source_url": source["source_url"],
                    "source_type": source["source_type"],
                    "region": source["region"],
                    "varieties": source["varieties"],
                    "use_mode": source["use_mode"],
                    "language": "vi",
                }
                candidates.append(item)
                candidates.extend(focused(item))
        seen, kept = set(), []
        for item in candidates:
            key = item["topic"] + re.sub(r"\W+", "", item["text"].lower())[:1000]
            digest = hashlib.sha1(key.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            kept.append(item)
        for index, item in enumerate(kept, start=1):
            item["chunk_id"] = f"{source['source_id']}_C{index:04d}"
            chunks.append(item)
        source["chunk_count"] = len(kept)
    return chunks, registry


def write_outputs(
    chunks: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    output: Path,
    *,
    complete_name: str = "rice_knowledge_complete.jsonl",
    nonrestricted_name: str = "rice_knowledge_nonrestricted.jsonl",
    registry_name: str = "source_registry_complete.json",
    report_name: str = "chunking_report.json",
    csv_name: str = "knowledge_chunks_complete.csv",
    review_name: str = "review_queue.csv",
    topic_language: str = "vi",
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(complete_name, chunks)
    write_jsonl(nonrestricted_name, [chunk for chunk in chunks if not chunk["restricted_action"]])
    (output / registry_name).write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "total_chunks": len(chunks),
        "by_topic": {topic: sum(chunk["topic"] == topic for chunk in chunks) for topic in TOPIC_WORDS[topic_language]},
        "by_facet": {facet: sum(chunk["facet"] == facet for chunk in chunks) for facet in FACET_WORDS[topic_language]},
        "by_source": {source["source_id"]: source["chunk_count"] for source in registry},
        "stage_high_compatibility": {
            stage: sum(chunk["stage_compatibility"][index] >= 0.8 for chunk in chunks)
            for index, stage in enumerate(STAGES)
        },
        "restricted_action_chunks": sum(chunk["restricted_action"] for chunk in chunks),
        "production_eligible_chunks": sum(chunk["production_eligible"] for chunk in chunks),
        "valid": len({chunk["chunk_id"] for chunk in chunks}) == len(chunks),
    }
    (output / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "chunk_id", "source_id", "topic", "facet", "text", "stage_compatibility", "authority_score",
        "review_status", "production_eligible", "restricted_action", "evidence_type", "section",
        "page_start", "page_end", "region", "varieties", "source_url",
    ]
    with (output / csv_name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chunk in chunks:
            row = {key: chunk.get(key) for key in fields}
            for key in ["stage_compatibility", "region", "varieties"]:
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    review_fields = [
        "chunk_id", "source_id", "topic", "page_start", "section", "restricted_action",
        "use_mode", "review_status", "text",
    ]
    with (output / review_name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for chunk in sorted(chunks, key=lambda item: (not item["restricted_action"], item["source_id"], item["page_start"] or 0)):
            writer.writerow({key: chunk.get(key) for key in review_fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical CROPSTATE knowledge chunks from registered PDFs.")
    parser.add_argument("--source-root", type=Path, help="Directory of registered source PDFs. Omit to skip the PDF rebuild (e.g. when only refreshing the IRRI web corpus).")
    parser.add_argument("--registry", type=Path, default=Path("configs/knowledge_sources.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--web-pages-dir", type=Path, help="Directory of cached IRRI RKB HTML pages (see scripts/crawl_irri_rkb.py).")
    parser.add_argument("--web-registry", type=Path, default=Path("configs/knowledge_sources_irri.json"))
    args = parser.parse_args()
    summary: dict[str, Any] = {"output_dir": str(args.output_dir)}
    if args.source_root:
        chunks, registry = build(args.source_root, args.registry)
        write_outputs(chunks, registry, args.output_dir)
        summary["total_chunks"] = len(chunks)
    if args.web_pages_dir:
        web_chunks, web_registry = build_web(args.web_pages_dir, args.web_registry)
        write_outputs(
            web_chunks, web_registry, args.output_dir,
            complete_name="rice_knowledge_irri_en.jsonl",
            nonrestricted_name="rice_knowledge_irri_en_nonrestricted.jsonl",
            registry_name="source_registry_irri.json",
            report_name="chunking_report_irri.json",
            csv_name="knowledge_chunks_irri_en.csv",
            review_name="review_queue_irri.csv",
            topic_language="en",
        )
        summary["total_web_chunks"] = len(web_chunks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
