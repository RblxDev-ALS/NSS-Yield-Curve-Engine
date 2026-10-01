# Releasing nss-engine to PyPI

Releases are published by [`.github/workflows/release.yml`](../.github/workflows/release.yml),
started from the Actions tab (no git commands needed) or by pushing a tag
`v<version>`. It uses PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts
this workflow in this repository, so no API token is created or stored.

## One-time setup (repository owner)

1. **License.** Done: `LICENSE` (MIT) is in the repository and
   `pyproject.toml` declares `license = "MIT"` and `license-files`.
2. **Create a PyPI account** at <https://pypi.org/account/register/> and enable
   two-factor authentication (PyPI requires it to publish).
3. **Register the trusted publisher.** The name `nss-engine` is not taken yet,
   so use a *pending* publisher, which also reserves the name: go to
   <https://pypi.org/manage/account/publishing/>, section *Add a new pending
   publisher → GitHub*, and enter

   | field | value |
   |---|---|
   | PyPI Project Name | `nss-engine` |
   | Owner | `RblxDev-ALS` |
   | Repository name | `NSS-Yield-Curve-Engine` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. **Create the `pypi` environment on GitHub**: *Settings → Environments → New
   environment → `pypi`*. Optionally add yourself under *Required reviewers*,
   so that every upload waits for a click. If you restrict *Deployment
   branches and tags*, allow both the branch `main` (for releases started from
   the Actions tab) and the tag pattern `v*`.

Optional: repeat steps 2–4 on <https://test.pypi.org> with an environment
named `testpypi` to rehearse; the workflow would need a second publish job
with `repository-url: https://test.pypi.org/legacy/`.

## Every release

1. Update the version in **four** places: `pyproject.toml`,
   `src/nss_engine/__init__.py`, `CITATION.cff`, and a new `## <version> — …`
   section at the top of `CHANGELOG.md`. A test checks that they agree.
2. Merge to `main` and wait for CI to pass.
3. Publish, either way:

   * **From the browser**: *Actions → Release → Run workflow*, branch `main`,
     tick *Publish the version in pyproject.toml to PyPI and tag it*, and
     click *Run workflow*. The release is tagged `v<version>` on the commit
     it was built from.
   * **From a terminal**: tag the merge commit and push the tag:

     ```bash
     git checkout main && git pull
     git tag -a v2.5.0 -m "nss-engine 2.5.0"
     git push origin v2.5.0
     ```

The workflow then

* checks that the tag matches the version in `pyproject.toml` (or, when
  started from the Actions tab, that it runs on `main` and that the version
  has not been released yet),
* builds the sdist and wheel, runs `twine check --strict`, installs the wheel
  in a clean environment and fits a curve with it,
* uploads to PyPI from the `pypi` environment, and
* creates a GitHub release with the distributions attached and this version's
  CHANGELOG section as the notes.

A version number can be uploaded to PyPI only once. If something is wrong
after publishing, fix it and release a new patch version (`2.4.1`).

Every push or pull request that touches `pyproject.toml`, `README.md` or the
workflow runs the build-and-check job without publishing, so packaging
problems show up before a release.

## After the first release

```bash
pip install nss-engine               # the engine
pip install "nss-engine[surveys]"    # plus openpyxl and xlrd, to read the SPF and New York Fed files
```
