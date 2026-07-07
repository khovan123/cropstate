from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz

STAGES = ["establishment", "tillering", "stem_booting", "reproductive", "grain_filling", "ripening"]
TOPIC_WORDS = {
    "water_management": ["quản lý nước", "mực nước", "tưới", "tháo nước", "rút nước", "ngập", "khô ướt", "giữ nước", "tiêu nước"],
    "nutrient_management": ["bón phân", "dinh dưỡng", "phân đạm", "phân lân", "phân kali", "urea", "urê", "npk", "kcl", "dap"],
    "pest_risk": ["sâu hại", "rầy", "sâu cuốn lá", "sâu đục thân", "bọ trĩ", "nhện gié", "ốc bươu", "côn trùng", "ipm"],
    "disease_risk": ["bệnh hại", "đạo ôn", "bạc lá", "đốm vằn", "lem lép", "cháy bìa", "nấm bệnh", "vi khuẩn"],
    "weed_management": ["cỏ dại", "lúa cỏ", "làm cỏ", "khử lẫn", "thuốc cỏ"],
    "harvest_readiness": ["thu hoạch", "chín vàng", "gặt", "sấy", "bảo quản", "ẩm độ", "xay xát"],
    "residue_management": ["rơm rạ", "gốc rạ", "xử lý rạ", "thu gom rơm", "đốt rơm"],
    "climate_adaptation": ["biến đổi khí hậu", "phát thải", "khí nhà kính", "chịu hạn", "chịu mặn", "thích ứng"],
    "general_crop_care": ["làm đất", "chuẩn bị giống", "gieo sạ", "cấy", "mạ", "mật độ", "thời vụ", "canh tác"],
}
STAGE_WORDS = {
    0: ["nảy mầm", "thời kỳ mạ", "cây mạ", "gieo sạ", "xuống giống", "cấy"],
    1: ["đẻ nhánh", "nở bụi", "dưỡng chồi", "chồi tối đa"],
    2: ["làm đòng", "đón đòng", "dưỡng đòng", "phân hóa đòng", "đòng ói", "chuẩn bị trổ"],
    3: ["trổ", "trỗ", "ra hoa", "thụ phấn", "thụ tinh"],
    4: ["nuôi hạt", "chắc hạt", "vào gạo", "chín sữa", "ngậm sữa", "cong trái me"],
    5: ["chín vàng", "thu hoạch", "sấy lúa", "bảo quản lúa", "độ ẩm 14%"],
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
            current_heading = line
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


def choose_topic(section: str, text: str) -> str | None:
    joined = f"{section} {text}".lower()
    scores = {
        name: sum(joined.count(word) * (3 if word in section.lower() else 1) for word in words)
        for name, words in TOPIC_WORDS.items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score else None


def stage_vector(section: str, text: str, topic: str) -> list[float]:
    scores = [
        sum(count_phrase(text, word) + 3 * count_phrase(section, word) for word in words)
        for words in STAGE_WORDS.values()
    ]
    if topic == "harvest_readiness":
        scores[5] += 4
    if topic == "residue_management":
        scores[5] += 3
    if topic == "weed_management":
        scores[0] += 2
        scores[1] += 2
    if topic == "general_crop_care" and any(word in text.lower() for word in ["làm đất", "gieo sạ", "cấy", "cây mạ"]):
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
    for index, words in STAGE_WORDS.items():
        if any(count_phrase(combined, word) for word in words):
            vector[index] = max(vector[index], 0.85)
    vector[main] = 1.0
    return vector


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
    output = []
    for topic in ["weed_management", "pest_risk", "disease_risk"]:
        keywords = TOPIC_WORDS[topic]
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
            clone["stage_compatibility"] = stage_vector(clone["section"], snippet, topic)
            clone["derived_focus"] = True
            output.append(clone)
    return output


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


def write_outputs(chunks: list[dict[str, Any]], registry: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl("rice_knowledge_complete.jsonl", chunks)
    write_jsonl("rice_knowledge_nonrestricted.jsonl", [chunk for chunk in chunks if not chunk["restricted_action"]])
    (output / "source_registry_complete.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "total_chunks": len(chunks),
        "by_topic": {topic: sum(chunk["topic"] == topic for chunk in chunks) for topic in TOPIC_WORDS},
        "by_source": {source["source_id"]: source["chunk_count"] for source in registry},
        "stage_high_compatibility": {
            stage: sum(chunk["stage_compatibility"][index] >= 0.8 for chunk in chunks)
            for index, stage in enumerate(STAGES)
        },
        "restricted_action_chunks": sum(chunk["restricted_action"] for chunk in chunks),
        "production_eligible_chunks": sum(chunk["production_eligible"] for chunk in chunks),
        "valid": len({chunk["chunk_id"] for chunk in chunks}) == len(chunks),
    }
    (output / "chunking_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "chunk_id", "source_id", "topic", "text", "stage_compatibility", "authority_score",
        "review_status", "production_eligible", "restricted_action", "evidence_type", "section",
        "page_start", "page_end", "region", "varieties", "source_url",
    ]
    with (output / "knowledge_chunks_complete.csv").open("w", encoding="utf-8-sig", newline="") as handle:
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
    with (output / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for chunk in sorted(chunks, key=lambda item: (not item["restricted_action"], item["source_id"], item["page_start"])):
            writer.writerow({key: chunk.get(key) for key in review_fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical CROPSTATE knowledge chunks from registered PDFs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/knowledge_sources.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    chunks, registry = build(args.source_root, args.registry)
    write_outputs(chunks, registry, args.output_dir)
    print(json.dumps({"total_chunks": len(chunks), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
