from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import STAGE_ALIASES, STAGE_NAMES, STAGE_TO_ID

APPROVED_REVIEW_STATUSES = {"reviewed", "domain_reviewed", "approved"}
EXCLUDED_REVIEW_STATUSES = {"excluded", "sample_only_not_agronomic_ground_truth"}


class KnowledgeValidationError(ValueError):
    """Raised when a knowledge-base record does not satisfy the canonical schema."""


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict, bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _parse_list(value: Any) -> list[str]:
    parsed = _parse_jsonish(value, [])
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, tuple):
        return [str(item).strip() for item in parsed if str(item).strip()]
    text = str(parsed).strip()
    if not text:
        return []
    delimiter = "|" if "|" in text else ","
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def _stage_index(value: Any) -> int:
    if isinstance(value, int):
        if 0 <= value < len(STAGE_NAMES):
            return value
        raise KnowledgeValidationError(f"Stage index outside range: {value}")
    text = str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    normalized = STAGE_ALIASES.get(text, text)
    if normalized not in STAGE_TO_ID:
        raise KnowledgeValidationError(f"Unknown stage: {value}")
    return STAGE_TO_ID[normalized]


def parse_stage_compatibility(record: Mapping[str, Any]) -> list[float]:
    raw = record.get("stage_compatibility", record.get("compatibility_vector"))
    if raw is not None and str(raw).strip() != "":
        values = _parse_jsonish(raw, raw)
        if not isinstance(values, (list, tuple)):
            raise KnowledgeValidationError("stage_compatibility must be a JSON/list value")
        result = [float(value) for value in values]
        if len(result) != len(STAGE_NAMES):
            raise KnowledgeValidationError(f"stage_compatibility must contain {len(STAGE_NAMES)} values")
        return result

    columns = [
        "c_establishment", "c_tillering", "c_stem_booting", "c_reproductive",
        "c_grain_filling", "c_ripening",
    ]
    if any(column in record and str(record.get(column, "")).strip() for column in columns):
        return [float(record.get(column, 0.0) or 0.0) for column in columns]

    direct = record.get("direct_applicable_stages", record.get("stage_ids", record.get("applicable_stages")))
    stages = _parse_jsonish(direct, [])
    if isinstance(stages, str):
        stages = _parse_list(stages)
    if stages:
        result = [0.0] * len(STAGE_NAMES)
        for stage in stages:
            index = _stage_index(stage)
            result[index] = 1.0
            if index > 0:
                result[index - 1] = max(result[index - 1], 0.55)
            if index < len(STAGE_NAMES) - 1:
                result[index + 1] = max(result[index + 1], 0.55)
        return result

    return [0.35] * len(STAGE_NAMES)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    topic: str
    stage_compatibility: tuple[float, ...]
    authority_score: float
    facet: str = "general"
    source_id: str = ""
    review_status: str = "unreviewed"
    production_eligible: bool = False
    restricted_action: bool = False
    evidence_type: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    source_title: str = ""
    source_organization: str = ""
    source_year: int | None = None
    source_url: str = ""
    source_type: str = ""
    region: tuple[str, ...] = field(default_factory=tuple)
    varieties: tuple[str, ...] = field(default_factory=tuple)
    use_mode: str = ""
    language: str = "vi"

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], fallback_id: str = "") -> "KnowledgeChunk":
        chunk_id = str(record.get("chunk_id", record.get("id", fallback_id))).strip()
        text = str(record.get("text", record.get("content", record.get("chunk_text", "")))).strip()
        topic = str(record.get("topic", record.get("care_topic", "general_crop_care"))).strip()
        compatibility = tuple(parse_stage_compatibility(record))
        page_start = record.get("page_start", record.get("page"))
        page_end = record.get("page_end", page_start)
        source_year = record.get("source_year", record.get("year"))
        return cls(
            chunk_id=chunk_id,
            text=text,
            topic=topic or "general_crop_care",
            stage_compatibility=compatibility,
            authority_score=float(record.get("authority_score", 0.5) or 0.5),
            facet=str(record.get("facet", "")).strip() or "general",
            source_id=str(record.get("source_id", "")).strip(),
            review_status=str(record.get("review_status", "unreviewed")).strip().lower(),
            production_eligible=_parse_bool(record.get("production_eligible"), False),
            restricted_action=_parse_bool(record.get("restricted_action"), False),
            evidence_type=str(record.get("evidence_type", "")).strip(),
            section=str(record.get("section", "")).strip(),
            page_start=int(page_start) if page_start not in (None, "") else None,
            page_end=int(page_end) if page_end not in (None, "") else None,
            source_title=str(record.get("source_title", "")).strip(),
            source_organization=str(record.get("source_organization", "")).strip(),
            source_year=int(source_year) if source_year not in (None, "") and str(source_year).isdigit() else None,
            source_url=str(record.get("source_url", "")).strip(),
            source_type=str(record.get("source_type", "")).strip(),
            region=tuple(_parse_list(record.get("region", record.get("region_dependency")))),
            varieties=tuple(_parse_list(record.get("varieties", record.get("variety_dependency")))),
            use_mode=str(record.get("use_mode", "")).strip(),
            language=str(record.get("language", "vi")).strip() or "vi",
        )

    def validate(self, min_words: int = 20) -> None:
        if not self.chunk_id:
            raise KnowledgeValidationError("Missing chunk_id")
        if len(self.text.split()) < min_words:
            raise KnowledgeValidationError(f"{self.chunk_id}: text is shorter than {min_words} words")
        if len(self.stage_compatibility) != len(STAGE_NAMES):
            raise KnowledgeValidationError(f"{self.chunk_id}: compatibility length must be {len(STAGE_NAMES)}")
        if any(not 0.0 <= score <= 1.0 for score in self.stage_compatibility):
            raise KnowledgeValidationError(f"{self.chunk_id}: compatibility values must be in [0,1]")
        if not 0.0 <= self.authority_score <= 1.0:
            raise KnowledgeValidationError(f"{self.chunk_id}: authority_score must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stage_compatibility"] = list(self.stage_compatibility)
        result["region"] = list(self.region)
        result["varieties"] = list(self.varieties)
        return result


def _include_for_mode(chunk: KnowledgeChunk, mode: str, include_sample: bool) -> bool:
    if not include_sample and chunk.review_status in EXCLUDED_REVIEW_STATUSES:
        return False
    if mode == "all":
        return True
    if mode == "research":
        return chunk.review_status != "excluded"
    if mode == "production":
        return (
            chunk.review_status in APPROVED_REVIEW_STATUSES
            and chunk.production_eligible
            and not chunk.restricted_action
        )
    raise ValueError("mode must be one of: all, research, production")


def load_knowledge_chunks(
    path: str | Path,
    *,
    mode: str = "research",
    include_sample: bool = False,
    topics: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    varieties: Iterable[str] | None = None,
    min_words: int = 20,
) -> list[KnowledgeChunk]:
    input_path = Path(path)
    topic_filter = set(topics or [])
    region_filter = set(regions or [])
    variety_filter = set(varieties or [])
    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk = KnowledgeChunk.from_mapping(record, fallback_id=f"chunk_{line_number:05d}")
            chunk.validate(min_words=min_words)
            if chunk.chunk_id in seen_ids:
                raise KnowledgeValidationError(f"Duplicate chunk_id: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            if not _include_for_mode(chunk, mode, include_sample):
                continue
            if topic_filter and chunk.topic not in topic_filter:
                continue
            if region_filter and chunk.region and not region_filter.intersection(chunk.region):
                continue
            if variety_filter and chunk.varieties and "general" not in chunk.varieties and not variety_filter.intersection(chunk.varieties):
                continue
            chunks.append(chunk)
    return chunks


def write_knowledge_chunks(chunks: Sequence[KnowledgeChunk], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def knowledge_coverage(chunks: Sequence[KnowledgeChunk]) -> dict[str, Any]:
    by_topic: dict[str, int] = {}
    by_facet: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_review_status: dict[str, int] = {}
    stage_high_compatibility = {name: 0 for name in STAGE_NAMES}
    for chunk in chunks:
        by_topic[chunk.topic] = by_topic.get(chunk.topic, 0) + 1
        by_facet[chunk.facet] = by_facet.get(chunk.facet, 0) + 1
        by_source[chunk.source_id] = by_source.get(chunk.source_id, 0) + 1
        by_review_status[chunk.review_status] = by_review_status.get(chunk.review_status, 0) + 1
        for index, stage in enumerate(STAGE_NAMES):
            if chunk.stage_compatibility[index] >= 0.8:
                stage_high_compatibility[stage] += 1
    return {
        "total_chunks": len(chunks),
        "by_topic": dict(sorted(by_topic.items())),
        "by_facet": dict(sorted(by_facet.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_review_status": dict(sorted(by_review_status.items())),
        "stage_high_compatibility": stage_high_compatibility,
        "restricted_action_chunks": sum(chunk.restricted_action for chunk in chunks),
        "production_eligible_chunks": sum(chunk.production_eligible for chunk in chunks),
    }
