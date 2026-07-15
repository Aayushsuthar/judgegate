"""Create or update the judgegate comment on the current pull request.

Used by the composite GitHub Action. Requires GITHUB_TOKEN,
GITHUB_REPOSITORY, and GITHUB_EVENT_PATH in the environment. Finds an
existing comment by the judgegate marker and updates it in place, so a
pull request never accumulates duplicate reports. Comment failures are
warnings, never gate failures: fork pull requests receive read-only
tokens and must not break the check.
"""

import json
import os
import sys
import urllib.error
import urllib.request

MARKER = "<!-- judgegate-report -->"
API_ROOT = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def _request(method: str, url: str, token: str, payload: dict | None = None) -> object:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _pull_number(event: dict) -> int | None:
    pull = event.get("pull_request")
    if isinstance(pull, dict) and "number" in pull:
        return int(pull["number"])
    issue = event.get("issue")
    if isinstance(issue, dict) and "pull_request" in issue:
        return int(issue["number"])
    return None


def _find_existing(comments_url: str, token: str) -> int | None:
    page = 1
    while True:
        comments = _request("GET", f"{comments_url}?per_page=100&page={page}", token)
        if not isinstance(comments, list) or not comments:
            return None
        for comment in comments:
            if MARKER in comment.get("body", ""):
                return int(comment["id"])
        page += 1


def _upsert(repository: str, number: int, token: str, body: str) -> None:
    comments_url = f"{API_ROOT}/repos/{repository}/issues/{number}/comments"
    existing_id = _find_existing(comments_url, token)
    if existing_id is not None:
        _request(
            "PATCH",
            f"{API_ROOT}/repos/{repository}/issues/comments/{existing_id}",
            token,
            {"body": body},
        )
        print(f"updated comment {existing_id} on PR #{number}")
    else:
        created = _request("POST", comments_url, token, {"body": body})
        comment_id = created.get("id") if isinstance(created, dict) else "unknown"
        print(f"created comment {comment_id} on PR #{number}")


def _warn_http(exc: urllib.error.HTTPError) -> None:
    detail = exc.read().decode("utf-8", errors="replace")
    print(f"warning: GitHub API error {exc.code}: {detail}", file=sys.stderr)
    if exc.code in (401, 403, 404):
        print(
            "warning: could not post the comment (token lacks pull-requests: write, "
            "which is expected on pull requests from forks); the verdict is still "
            "enforced and the report is in the job summary",
            file=sys.stderr,
        )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: upsert_comment.py <report-file>", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not token or not repository or not event_path:
        print(
            "missing GITHUB_TOKEN, GITHUB_REPOSITORY, or GITHUB_EVENT_PATH",
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(sys.argv[1]):
        print(f"no report at {sys.argv[1]}; nothing to post", file=sys.stderr)
        return 0

    with open(sys.argv[1], encoding="utf-8") as handle:
        body = handle.read()
    if MARKER not in body:
        body = f"{MARKER}\n{body}"

    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)
    number = _pull_number(event)
    if number is None:
        print("not a pull request event; skipping comment", file=sys.stderr)
        return 0

    try:
        _upsert(repository, number, token, body)
    except urllib.error.HTTPError as exc:
        _warn_http(exc)
    except urllib.error.URLError as exc:
        print(f"warning: could not reach the GitHub API: {exc.reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
