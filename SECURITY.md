# Security Policy

## Supported versions

The latest release on PyPI receives security fixes.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
security advisories: open the repository's Security tab and choose
"Report a vulnerability". Do not open a public issue for anything you
believe has security impact.

You can expect an acknowledgment within a few days. Please include a
minimal reproduction if you can.

## Scope notes

- judgegate contacts exactly one network endpoint: the judge URL you
  configure. Offline mode contacts nothing.
- API keys are read from environment variables named in the config, never
  from the config file itself, and are never written to logs, reports, or
  the response cache.
- The response cache stores judge response text in plain SQLite, keyed by
  a content hash; prompts themselves are hashed, never stored. Treat the
  cache file with the same sensitivity as your eval data, and add
  `.judgegate-cache.sqlite` to your own repository's .gitignore when
  running in live mode.
- Labels files are treated strictly as data. No field is ever executed,
  templated into shell commands, or fetched as a URL.
- The GitHub Action passes all inputs through environment variables
  rather than interpolating them into scripts.
