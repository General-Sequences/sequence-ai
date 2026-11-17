# Releasing sequence-ai

Versions follow [Semantic Versioning](https://semver.org). Before 1.0.0, a minor version may change an interface;
[CHANGELOG.md](CHANGELOG.md) says so under **Changed** when one does.

## A release

A release is cut by General Sequences. Each one is:

1. One commit on `main` that holds the whole tree of that version: the components with the pull requests merged since
   the last release, and the hosted platform's client as that version ships it.
2. The tag `vX.Y.Z` on that commit, the version `pyproject.toml` and `CITATION.cff` state.
3. The package on PyPI, `sequence-ai X.Y.Z`: its `seq/` and `sequence_ai/` are, file for file, the ones of that
   commit. A version that is on PyPI is never replaced.
4. A GitHub release, written from the version's section of `CHANGELOG.md`.

The lint, the tests and a build of the package ([tests.yml](.github/workflows/tests.yml)) run on that commit as on any
other.

## What a release must not break

- A component stays usable alone: one file, importing none of the others.
- A number in the README or a docstring still states what it was measured with.
- Nothing changes a policy's outputs or a machine's settings: the output of `cuda_graphs` still equals eager mode's
  byte for byte.
