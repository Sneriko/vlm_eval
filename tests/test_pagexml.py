from pathlib import Path

from vlm_eval.pagexml import parse_pagexml_text


def test_parse_pagexml_text(tmp_path: Path):
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'>
  <Page>
    <TextRegion>
      <TextLine>
        <TextEquiv><Unicode>Rad ett</Unicode></TextEquiv>
      </TextLine>
      <TextLine>
        <TextEquiv><Unicode>Rad två</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(xml, encoding="utf-8")

    assert parse_pagexml_text(xml_path) == "Rad ett\nRad två"


def test_parse_pagexml_prefers_line_level_when_word_level_also_exists(tmp_path: Path):
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'>
  <Page>
    <TextRegion>
      <TextLine>
        <TextEquiv><Unicode>God dag</Unicode></TextEquiv>
        <Word><TextEquiv><Unicode>God</Unicode></TextEquiv></Word>
        <Word><TextEquiv><Unicode>dag</Unicode></TextEquiv></Word>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    xml_path = tmp_path / "mixed_levels.xml"
    xml_path.write_text(xml, encoding="utf-8")

    assert parse_pagexml_text(xml_path) == "God dag"
