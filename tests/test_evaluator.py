from pathlib import Path

from vlm_eval.config import EvalConfig, ModelConfig
from vlm_eval.evaluator import evaluate, find_samples, summarize


def test_find_samples_only_pairs_pagexml_with_archive_image(tmp_path: Path):
    archive_dir = tmp_path / "archive_a"
    page_dir = archive_dir / "page"
    alto_dir = archive_dir / "alto"
    page_dir.mkdir(parents=True)
    alto_dir.mkdir(parents=True)

    image_path = archive_dir / "page_0001.jpg"
    image_path.write_bytes(b"image")

    page_xml_path = page_dir / "page_0001.xml"
    page_xml_path.write_text("<xml />", encoding="utf-8")

    alto_xml_path = alto_dir / "page_0001.xml"
    alto_xml_path.write_text("<xml />", encoding="utf-8")

    samples = list(find_samples(tmp_path, [".jpg"], ".xml"))

    assert samples == [(image_path, page_xml_path)]


def test_find_samples_skips_xml_without_image_in_archive_dir(tmp_path: Path):
    archive_dir = tmp_path / "archive_b"
    page_dir = archive_dir / "page"
    page_dir.mkdir(parents=True)

    (page_dir / "missing.xml").write_text("<xml />", encoding="utf-8")

    samples = list(find_samples(tmp_path, [".jpg"], ".xml"))

    assert samples == []


def test_evaluate_emits_page_folder_and_testset_rows(tmp_path: Path, monkeypatch):
    archive_a = tmp_path / "archive_a"
    archive_b = tmp_path / "archive_b"
    (archive_a / "page").mkdir(parents=True)
    (archive_b / "page").mkdir(parents=True)

    img_a = archive_a / "page_0001.jpg"
    img_b = archive_b / "page_0001.jpg"
    img_a.write_bytes(b"img")
    img_b.write_bytes(b"img")

    xml_a = archive_a / "page" / "page_0001.xml"
    xml_b = archive_b / "page" / "page_0001.xml"
    xml_a.write_text("<xml />", encoding="utf-8")
    xml_b.write_text("<xml />", encoding="utf-8")

    references = {str(xml_a): "a b", str(xml_b): "c d"}

    class DummyClient:
        name = "dummy-model"

        def transcribe(self, image_path: str, prompt: str) -> str:
            return {str(img_a): "a b", str(img_b): "c"}[image_path]

    monkeypatch.setattr("vlm_eval.evaluator.build_client", lambda _: DummyClient())
    monkeypatch.setattr("vlm_eval.evaluator.parse_pagexml_text", lambda p: references[str(p)])

    logs: list[str] = []
    cfg = EvalConfig(
        dataset_dir=tmp_path,
        image_extensions=[".jpg"],
        pagexml_extension=".xml",
        models=[
            ModelConfig(
                name="dummy-model",
                provider="openai_compatible",
                model="x",
                api_key_env="DUMMY",
            )
        ],
    )

    rows = evaluate(cfg, progress_logger=logs.append)

    page_rows = [r for r in rows if r.level == "page"]
    folder_rows = [r for r in rows if r.level == "folder"]
    testset_rows = [r for r in rows if r.level == "testset"]

    assert len(page_rows) == 2
    assert len(folder_rows) == 2
    assert len(testset_rows) == 1
    assert any(log.startswith("[PAGE]") for log in logs)
    assert any(log.startswith("[FOLDER]") for log in logs)

    folder_scores = {r.scope_id: r.bow_f1 for r in folder_rows}
    assert folder_scores["archive_a"] == 1.0
    assert folder_scores["archive_b"] < 1.0


def test_summarize_prefers_testset_rows():
    class Row:
        def __init__(self, level, model, bow_precision, bow_recall, bow_f1):
            self.level = level
            self.model = model
            self.bow_precision = bow_precision
            self.bow_recall = bow_recall
            self.bow_f1 = bow_f1

    rows = [
        Row("page", "m", 0.1, 0.1, 0.1),
        Row("testset", "m", 0.8, 0.7, 0.75),
    ]

    summary = summarize(rows)

    assert summary["m"]["bow_precision"] == 0.8
    assert summary["m"]["bow_recall"] == 0.7
    assert summary["m"]["bow_f1"] == 0.75
