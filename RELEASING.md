# Releasing SequenceLink

Versions follow [Semantic Versioning](https://semver.org). Before 1.0, a minor version may change or remove an
interface, and [CHANGELOG.md](CHANGELOG.md) lists each under Changed or Removed. A patch version changes no interface.
[What is open source, and what does General Sequences host?](docs/what-is-open-source-and-what-is-hosted.md#versions)
says the same.

## A release

A release is cut by General Sequences. Each one is:

1. One commit on `main` that holds the whole tree of that version: the components with the pull requests merged since
   the last release, and the hosted platform's client as that version ships it.
2. The tag `vX.Y.Z` on that commit, the version `pyproject.toml` and `CITATION.cff` state.
3. The package on PyPI, `sequence-ai X.Y.Z`: its `seq/` and `sequence_ai/` are, file for file, the ones of that
   commit. A version that is on PyPI is never replaced.
4. A GitHub release of that tag, written from the version's section of `CHANGELOG.md` and created by General
   Sequences' GitHub account, `GeneralSequences`, never by a person's: either right after the tag by `gh` signed in as
   that account where the release is cut (`gh auth login`, its one-time code approved in a browser signed in as it),
   or on GitHub in a browser signed in as it. The newest version's release is marked Latest.

The lint, the tests and a build of the package ([tests.yml](.github/workflows/tests.yml)) run on that commit as on any
other.

## What a release must not break

- A component stays usable alone: one file, importing none of the others.
- A number in the README or a docstring still states what it was measured with.
- Nothing changes a policy's outputs or a machine's settings: the output of `cuda_graphs` still equals eager mode's
  byte for byte.
