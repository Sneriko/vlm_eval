from pathlib import Path

from vlm_eval.evaluator import find_samples


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
