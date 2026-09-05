import pandas as pd

from collect_linked_github import linked_repository_map


def test_linked_repository_map_deduplicates_case_and_preserves_spaces() -> None:
    spaces = pd.DataFrame(
        [
            {
                "space_id": "one/app",
                "github_repositories": "Owner/Repo;third/tool",
            },
            {"space_id": "two/app", "github_repositories": "owner/repo"},
        ]
    )
    result = linked_repository_map(spaces)
    assert result["Owner/Repo"] == ["one/app", "two/app"]
    assert result["third/tool"] == ["one/app"]
