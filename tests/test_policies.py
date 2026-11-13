"""Policy files are checked at load time, so a bad or colliding policy never reaches the pipeline."""

import pytest
import yaml

from priorauth.policies import PolicyError, fingerprint, load_policies, load_policy

URL = "https://example.org/lcd?ver=1"
CRITERION = {
    "criterion_id": "facet.duration",
    "text": "Pain for at least 3 months",
    "source_quotes": ["minimum of 3 months"],
}
SOURCE = "Coverage\nPain present for minimum of 3\nmonths. The patient\u2019s pain is axial."


def policy_data(pid="facet", criteria=None, source=URL):
    return {
        "policy_id": pid,
        "title": "Test policy",
        "source": source,
        "criteria": criteria or [CRITERION | {"criterion_id": f"{pid}.duration"}],
    }


def write(directory, name, data):
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def write_source(directory, name="lcd_v1", url=URL, text=SOURCE):
    (directory / "sources").mkdir(exist_ok=True)
    (directory / "sources" / f"{name}.txt").write_text(f"# source: {url}\n# section: Coverage\n\n{text}\n")


def test_valid_policy_loads(tmp_path):
    policy = load_policy(write(tmp_path, "facet", policy_data()))
    assert policy.policy_id == "facet"
    assert policy.criteria[0].required is True


def test_directory_loads_keyed_by_policy_id_and_ignores_other_files(tmp_path):
    write_source(tmp_path)
    write(tmp_path, "facet", policy_data("facet"))
    write(tmp_path, "mri", policy_data("mri"))
    (tmp_path / "README.md").write_text("notes")
    assert sorted(load_policies(tmp_path)) == ["facet", "mri"]


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(PolicyError, match="no policy files"):
        load_policies(tmp_path)


def test_policy_id_must_match_file_name(tmp_path):
    with pytest.raises(PolicyError, match="must match the file name"):
        load_policy(write(tmp_path, "renamed", policy_data("facet")))


@pytest.mark.parametrize("pid", ["Facet", "facet.v2", "facet-v2"])
def test_policy_id_format_is_enforced(tmp_path, pid):
    with pytest.raises(PolicyError, match="lowercase"):
        load_policy(write(tmp_path, pid, policy_data(pid)))


def test_criterion_id_must_carry_its_policy_prefix(tmp_path):
    data = policy_data(criteria=[{"criterion_id": "duration", "text": "t"}])
    with pytest.raises(PolicyError, match="must look like"):
        load_policy(write(tmp_path, "facet", data))


def test_criterion_id_from_another_policy_is_rejected(tmp_path):
    """The collision case: a shared name like 'duration' can no longer mean two different rules."""
    data = policy_data(criteria=[{"criterion_id": "mri.duration", "text": "Pain for at least 6 weeks"}])
    with pytest.raises(PolicyError, match="must look like"):
        load_policy(write(tmp_path, "facet", data))


def test_duplicate_criterion_ids_are_rejected(tmp_path):
    criterion = {"criterion_id": "facet.duration", "text": "t"}
    with pytest.raises(PolicyError, match="unique"):
        load_policy(write(tmp_path, "facet", policy_data(criteria=[criterion, criterion])))


def test_schema_violation_names_the_file(tmp_path):
    data = policy_data()
    del data["title"]
    with pytest.raises(PolicyError, match="facet.yaml"):
        load_policy(write(tmp_path, "facet", data))


def test_invalid_yaml_is_a_policy_error(tmp_path):
    path = tmp_path / "facet.yaml"
    path.write_text("criteria: [unclosed")
    with pytest.raises(PolicyError):
        load_policy(path)


def test_yaml_cannot_run_code(tmp_path):
    """safe_load refuses Python object tags; an unsafe loader would create the marker file."""
    marker = tmp_path / "marker"
    path = tmp_path / "facet.yaml"
    path.write_text(f"!!python/object/apply:builtins.open ['{marker}', 'w']")
    with pytest.raises(PolicyError):
        load_policy(path)
    assert not marker.exists()


def test_quotes_match_across_line_breaks_and_curly_apostrophes(tmp_path):
    write_source(tmp_path)
    quotes = ["Pain present for minimum of 3 months.", "The patient's pain is axial."]
    criteria = [{"criterion_id": "facet.duration", "text": "t", "source_quotes": quotes}]
    write(tmp_path, "facet", policy_data(criteria=criteria))
    assert load_policies(tmp_path)["facet"].criteria[0].source_quotes == quotes


def test_quote_missing_from_the_source_is_rejected(tmp_path):
    write_source(tmp_path)
    criteria = [{"criterion_id": "facet.duration", "text": "t", "source_quotes": ["minimum of 6 weeks"]}]
    write(tmp_path, "facet", policy_data(criteria=criteria))
    with pytest.raises(PolicyError, match="facet.duration"):
        load_policies(tmp_path)


def test_criterion_without_quotes_is_rejected(tmp_path):
    write_source(tmp_path)
    write(tmp_path, "facet", policy_data(criteria=[{"criterion_id": "facet.duration", "text": "t"}]))
    with pytest.raises(PolicyError, match="without source_quotes"):
        load_policies(tmp_path)


def test_policy_citing_a_source_with_no_extracted_text_is_rejected(tmp_path):
    write_source(tmp_path)
    write(tmp_path, "facet", policy_data(source="https://example.org/lcd?ver=2"))
    with pytest.raises(PolicyError, match="no extracted text"):
        load_policies(tmp_path)


def test_source_file_needs_a_source_header(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "lcd_v1.txt").write_text(SOURCE)
    write(tmp_path, "facet", policy_data())
    with pytest.raises(PolicyError, match="first line"):
        load_policies(tmp_path)


def test_two_source_files_with_the_same_url_are_rejected(tmp_path):
    write_source(tmp_path, "lcd_v1")
    write_source(tmp_path, "lcd_v1_copy")
    write(tmp_path, "facet", policy_data())
    with pytest.raises(PolicyError, match="already has"):
        load_policies(tmp_path)


def test_fingerprint_ignores_formatting_comments_and_explicit_defaults(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    criterion = CRITERION | {"required": True}
    reformatted = yaml.safe_dump(policy_data(criteria=[criterion]), default_flow_style=True)
    path = tmp_path / "facet.yaml"
    path.write_text("# reviewed Nov 2026\n" + reformatted)
    assert fingerprint(load_policy(path)) == fingerprint(original)


def test_fingerprint_changes_when_criterion_text_changes(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    edited = policy_data(criteria=[CRITERION | {"text": "Pain for at least 6 weeks"}])
    assert fingerprint(load_policy(write(tmp_path, "facet", edited))) != fingerprint(original)


def test_fingerprint_changes_when_required_flag_changes(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    optional = CRITERION | {"required": False}
    edited = load_policy(write(tmp_path, "facet", policy_data(criteria=[optional])))
    assert fingerprint(edited) != fingerprint(original)