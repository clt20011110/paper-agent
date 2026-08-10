import json
from contextlib import closing
import sqlite3

from paper_agent.cli import main


def test_import_seeds_supports_identifiers_bibliographies_zotero_and_pdf(tmp_path, capsys) -> None:
    bib = tmp_path / "library.bib"
    bib.write_text(
        "@article{one,title={Bib One},author={Ada Lovelace},year={2024},doi={10.1000/bib-one}}\n"
        "@article{two,title={Bib Two},author={Grace Hopper},year={2025},doi={10.1000/bib-two}}\n",
        encoding="utf-8",
    )
    ris = tmp_path / "library.ris"
    ris.write_text(
        "TY  - JOUR\nTI  - RIS One\nAU  - Lin Chen\nPY  - 2023\nDO  - 10.1000/ris-one\nER  -\n",
        encoding="utf-8",
    )
    zotero = tmp_path / "zotero.json"
    zotero.write_text(
        json.dumps(
            [
                {
                    "key": "Z1",
                    "itemType": "journalArticle",
                    "title": "Zotero One",
                    "creators": [
                        {"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"}
                    ],
                    "date": "2022-06-01",
                    "DOI": "10.1000/zotero-one",
                }
            ]
        ),
        encoding="utf-8",
    )
    pdf = tmp_path / "Local Paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    database = tmp_path / "papers.sqlite3"
    command = [
        "--run-id",
        "seed-run",
        "import-seeds",
        "--database",
        str(database),
        "--seed",
        "doi:10.1000/direct",
        "--seed",
        "2401.00001v2",
        "--input",
        str(bib),
        "--input",
        str(ris),
        "--input",
        str(zotero),
        "--input",
        str(pdf),
    ]

    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["input_count"], first["imported_count"], first["status"]) == (7, 7, "complete")
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["paper_ids"] == first["paper_ids"]

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0] == 7
        assert connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'seed-run'"
        ).fetchone()[0] == "complete"


def test_import_seeds_dry_run_does_not_create_database(tmp_path, capsys) -> None:
    database = tmp_path / "papers.sqlite3"
    assert main(
        ["--dry-run", "import-seeds", "--database", str(database), "--seed", "10.1000/example"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert not database.exists()
