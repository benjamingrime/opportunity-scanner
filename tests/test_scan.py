import pytest
from fastapi.testclient import TestClient

from scanner import core
from scanner.app import app


def test_industries_have_mixes_that_sum_to_one():
    inds = core._load("industries")["industries"]
    assert len(inds) > 80
    for i in inds:
        assert 0.98 <= sum(m["share"] for m in i["mix"]) <= 1.0001, i["code"]
        assert i["mix_from"] in ("division", f"section {i['section']}")


def test_small_industries_borrow_their_section():
    inds = {i["code"]: i for i in core._load("industries")["industries"]}
    assert inds["69"]["mix_from"] == "division"          # legal and accounting: large
    assert inds["15"]["mix_from"] == "section C"         # leather: too few survey responses


def test_exposure_shares_sum_to_one():
    for soc, o in core.occupations().items():
        if o["exposure"]:
            assert sum(o["exposure"].values()) == pytest.approx(1, abs=1e-3), soc
            assert o["beta"] == pytest.approx(o["exposure"]["E1"] + 0.5 * o["exposure"]["E2"], abs=1e-3)


def test_default_mix_is_percentages():
    mix = core.default_mix("69")
    assert sum(r["value"] for r in mix["rows"]) == pytest.approx(100, abs=0.5)
    assert 0 < mix["covers"] <= 1


def test_scan_percent_and_count_agree():
    rows = [{"soc": "2412", "value": 50}, {"soc": "4122", "value": 50}]
    by_pct = core.scan(rows, headcount=10, mode="percent")
    by_count = core.scan([{"soc": "2412", "value": 5}, {"soc": "4122", "value": 5}], mode="count")
    assert by_pct["firm"]["shares"] == by_count["firm"]["shares"]
    assert by_count["headcount"] == 10


def test_scan_firm_shares_are_headcount_weighted():
    occ = core.occupations()
    res = core.scan([{"soc": "2412", "value": 3}, {"soc": "4122", "value": 1}], mode="count")
    expected = (3 * occ["2412"]["exposure"]["E1"] + occ["4122"]["exposure"]["E1"]) / 4
    assert res["firm"]["shares"]["E1"] == pytest.approx(expected, abs=1e-3)


def test_scan_counts_people_without_data_separately():
    missing = next(s for s, o in core.occupations().items() if o["exposure"] is None)
    res = core.scan([{"soc": "2412", "value": 4}, {"soc": missing, "value": 1}], mode="count")
    assert res["firm"]["people_without_data"] == 1
    assert res["firm"]["shares"]["E1"] == core.occupations()["2412"]["exposure"]["E1"]


def test_percent_mode_needs_headcount():
    with pytest.raises(ValueError):
        core.scan([{"soc": "2412", "value": 100}], mode="percent")


def test_search_by_job_title():
    assert core.search("payroll")[0]["soc"] == "4122"
    assert core.search("") == []


def test_tasks_grouped_by_bucket():
    t = core.tasks("4122")
    assert set(t["tasks"]) == {"E1", "E2", "E0"}
    assert sum(len(v) for v in t["tasks"].values()) > 0


def test_api_round_trip():
    c = TestClient(app)
    assert c.get("/").status_code == 200
    mix = c.get("/api/industries/62").json()
    res = c.post("/api/scan", json={"rows": [{"soc": r["soc"], "value": r["value"]} for r in mix["rows"]],
                                    "headcount": 50, "mode": "percent"}).json()
    assert res["headcount"] == 50 and res["quick_wins"]
    assert c.get("/api/industries/00").status_code == 404
    assert c.post("/api/scan", json={"rows": [{"soc": "abc", "value": 1}]}).status_code == 422
