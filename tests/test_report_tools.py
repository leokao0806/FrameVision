import json
import os

from report.build_tables import ablation_table, build, main_table
from report.check_numbers import find_unbacked_numbers
from report.generate_report import final_problems, strip_comments, zip_members

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "final_small.json")


def _final():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_check_numbers_flags_typed_number():
    assert find_unbacked_numbers("Our model gets 0.512.", [{"x": 0.417}], set()) == ["0.512"]


def test_check_numbers_accepts_value_from_any_source():
    assert find_unbacked_numbers("Qwen 0.417, leakage 0.203.", [{"m": {"t": 0.4171}}, {"by_article": [0.2031]}], set()) == []


def test_check_numbers_accepts_negative_differences_and_the_allow_list():
    text = "The difference is −0.076 [−0.156, 0.020]; the old README said 0.4421."
    assert find_unbacked_numbers(text, [{"d": -0.0762, "ci": [-0.1556, 0.0199]}], {"0.4421"}) == []


def test_check_numbers_ignores_versions_and_html_comments():
    assert find_unbacked_numbers("PyTorch 2.11, scorer v1.05 <!-- old 0.999 -->", [], set()) == []


def test_build_tables_replaces_markers():
    out = build("Intro\n<!-- table:main -->\nMiddle\n<!-- table:ablation -->\nEnd\n", _final())
    assert "<!-- table:" not in out
    assert out.startswith("Intro\n| Model |")
    assert "0.417 [0.292, 0.553]" in out
    assert "**Qwen2.5-VL, title + full text (E12b)**" in out  # the model selected on dev is bold


def test_main_table_skips_models_without_test_and_lists_trained_best_only():
    table = main_table(_final())
    assert "e6" not in table.lower() and "E6" not in table
    assert "E5" in table and "E2 " not in table  # only the best trained model joins the main table


def test_ablation_table_reports_mean_and_std_over_seeds():
    table = ablation_table(_final())
    assert "| E2: image only (global) | 0.410 ± 0.020 | 0.400 ± 0.015 [0.300, 0.510] |" in table


def test_ablation_table_without_trained_models_says_so():
    final = _final()
    final["models"] = {k: v for k, v in final["models"].items() if not k.startswith("e")}
    assert "No trained-model results yet" in ablation_table(final)


def test_every_number_in_the_tables_is_backed_by_the_results():
    final = _final()
    assert find_unbacked_numbers(main_table(final) + ablation_table(final), [final], set()) == []


def test_strip_comments_removes_multiline_html_comments():
    assert strip_comments("a\n<!--\nnote 0.9\n-->\nb\n") == "a\n\nb\n"


def test_final_problems_lists_placeholders_and_missing_figures(tmp_path):
    (tmp_path / "ok.png").write_bytes(b"png")
    md = "Text [[TBD: x]] and [[LEO: y]].\n![a](ok.png)\n![b](missing.png)\n"
    problems = final_problems(md, str(tmp_path))
    assert problems == ["2 placeholder(s) left: [[TBD: x]], [[LEO: y]]", "missing figure: missing.png"]


def test_zip_members_keep_sources_and_skip_caches_and_outputs(tmp_path):
    files = ["README.md", "requirements.txt", "src/a.py", "src/__pycache__/a.cpython-312.pyc",
             "predictions/subtask2/m/seed0/dev.tsv", "docs/notes/x.md", "results/final.json",
             "results/figures/main.png", "report/final_report.md", "report/FrameVision_Final_Report.pdf",
             "baselines/results/subtask1_metrics.json", "baselines/results/cache/feat.npy", "data/img.jpg"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert sorted(zip_members(str(tmp_path))) == sorted([
        "README.md", "requirements.txt", "src/a.py", "predictions/subtask2/m/seed0/dev.tsv", "docs/notes/x.md",
        "results/final.json", "report/final_report.md", "baselines/results/subtask1_metrics.json"])
