import json
import tempfile
import unittest
from pathlib import Path

from cropstate.knowledge import KnowledgeChunk, load_knowledge_chunks


class KnowledgeTests(unittest.TestCase):
    def test_legacy_mapping_is_canonicalized(self):
        chunk = KnowledgeChunk.from_mapping({
            "id": "C1",
            "content": "Đây là một đoạn nội dung đủ dài để kiểm tra việc chuyển đổi dữ liệu cũ sang schema canonical của knowledge base CROPSTATE trong chế độ nghiên cứu.",
            "topic": "water_management",
            "direct_applicable_stages": [1],
            "authority_score": 0.8,
        })
        self.assertEqual(chunk.chunk_id, "C1")
        self.assertEqual(chunk.text[:3], "Đây")
        self.assertEqual(chunk.stage_compatibility[1], 1.0)
        self.assertEqual(chunk.stage_compatibility[0], 0.55)

    def test_production_filter(self):
        records = [
            {
                "chunk_id": "approved",
                "text": "Nội dung hướng dẫn đã được chuyên gia duyệt và đủ dài để loader giữ lại trong chế độ production của hệ thống retrieval CROPSTATE.",
                "topic": "water_management",
                "stage_compatibility": [1, 0.5, 0, 0, 0, 0],
                "authority_score": 0.9,
                "review_status": "reviewed",
                "production_eligible": True,
            },
            {
                "chunk_id": "pending",
                "text": "Nội dung được máy trích xuất và vẫn đang chờ người có chuyên môn kiểm tra trước khi sử dụng để đưa ra khuyến nghị thực tế.",
                "topic": "water_management",
                "stage_compatibility": [1, 0.5, 0, 0, 0, 0],
                "authority_score": 0.9,
                "review_status": "machine_curated_pending_domain_review",
                "production_eligible": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records), encoding="utf-8")
            loaded = load_knowledge_chunks(path, mode="production", min_words=10)
        self.assertEqual([chunk.chunk_id for chunk in loaded], ["approved"])


if __name__ == "__main__":
    unittest.main()
