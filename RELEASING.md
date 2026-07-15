# Releasing

judgegate publishes to PyPI through [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API tokens are stored anywhere.

## One-time setup (maintainer)

1. Sign in at https://pypi.org.
2. Go to https://pypi.org/manage/account/publishing/ and add a
   **pending publisher** with exactly these values:
   - PyPI project name: `judgegate`
   - Owner: `yashchimata`
   - Repository name: `judgegate`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. That is all. The first successful run of the Release workflow claims
   the `judgegate` name and publishes.

## Cutting a release

1. Update the version in `pyproject.toml` and `src/judgegate/__about__.py`
   (keep them identical) and add a section to `CHANGELOG.md`.
2. Commit, push, and confirm CI is green.
3. Tag and publish a GitHub release:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title "vX.Y.Z" --notes-file notes.md
   ```

4. Publishing the GitHub release triggers `.github/workflows/release.yml`.
   The `pypi` environment requires a manual approval: GitHub shows a
   "Review deployments" button on the run, and the publish job starts only
   after the maintainer approves it. The workflow can also be started
   manually from the Actions tab, or re-run if the PyPI side was not
   ready yet.
5. Verify with `pip install judgegate==X.Y.Z` in a fresh environment.

## Regenerating README screenshots

```bash
python scripts/render_assets.py
```

The SVGs in `assets/` are produced from real output on `examples/`, so
regenerate them whenever report rendering changes.
