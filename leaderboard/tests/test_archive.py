import datetime
from pathlib import Path

from aletheia_runner.archive import SubmissionArchive

WHEN = datetime.datetime(2026, 6, 12, 20, 2, 48, tzinfo=datetime.timezone.utc)


def test_local_archive_saves_zip_under_sanitized_team(tmp_path):
    a = SubmissionArchive(str(tmp_path / "subs"))
    data = b"PK\x03\x04 fake-but-zippy bytes"
    path = Path(a.save("Team A!/x", data, WHEN))
    assert path.exists() and path.read_bytes() == data
    assert path.parent.name == "Team_A_x"                 # sanitized team
    assert path.name.startswith("20260612T200248Z-") and path.name.endswith(".zip")


def test_archive_names_differ_by_content(tmp_path):
    a = SubmissionArchive(str(tmp_path / "subs"))
    p1 = a.save("t", b"aaa", WHEN)
    p2 = a.save("t", b"bbb", WHEN)
    assert p1 != p2                                       # content hash distinguishes them
    # identical content at the same instant collapses to one object
    assert a.save("t", b"aaa", WHEN) == p1
