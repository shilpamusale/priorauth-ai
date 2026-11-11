"""Policy files are checked at load time, so a bad or colliding policy never reaches the pipeline."""

import pytest
import yaml

from priorauth.policies import PolicyError, fingerprint, load_policies, load_policy


def policy_data(pid="facet", criteria=None):
    return {
        "policy_id": pid,
        "title": "Test policy",
        "source": "illustrative",
        "criteria": criteria or [{"criterion_id": f"{pid}.duration", "text": "Pain for at least 3 months"}],
    }


def write(directory, name, data):
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_valid_policy_loads(tmp_path):
    policy = load_policy(write(tmp_path, "facet", policy_data()))
    assert policy.policy_id == "facet"
    assert policy.criteria[0].required is True


def test_directory_loads_keyed_by_policy_id_and_ignores_other_files(tmp_path):
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


def test_fingerprint_ignores_formatting_comments_and_explicit_defaults(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    criterion = {"criterion_id": "facet.duration", "text": "Pain for at least 3 months", "required": True}
    reformatted = yaml.safe_dump(policy_data(criteria=[criterion]), default_flow_style=True)
    path = tmp_path / "facet.yaml"
    path.write_text("# reviewed Nov 2026\n" + reformatted)
    assert fingerprint(load_policy(path)) == fingerprint(original)


def test_fingerprint_changes_when_criterion_text_changes(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    edited = policy_data(criteria=[{"criterion_id": "facet.duration", "text": "Pain for at least 6 weeks"}])
    assert fingerprint(load_policy(write(tmp_path, "facet", edited))) != fingerprint(original)


def test_fingerprint_changes_when_required_flag_changes(tmp_path):
    original = load_policy(write(tmp_path, "facet", policy_data()))
    optional = {"criterion_id": "facet.duration", "text": "Pain for at least 3 months", "required": False}
    edited = load_policy(write(tmp_path, "facet", policy_data(criteria=[optional])))
    assert fingerprint(edited) != fingerprint(original)