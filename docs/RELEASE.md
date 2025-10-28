# Release Checklist & DOI Registration

This guide documents how to prepare a tagged release and obtain a Zenodo DOI
for the lightmatter IVI analysis code.

## 1. Prepare repository

1. Ensure `python3 scripts/check_data_integrity.py` passes.
2. Run
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.lock
   pytest -q
   make quick SEED=123
   ```
   to confirm tests and quick analysis succeed.
3. Update version number in `pyproject.toml` and `CITATION.cff`.
4. Replace the placeholder DOI in `CITATION.cff` with `10.5281/zenodo.XXXXXXX`
   once a DOI is minted (see below).
5. Commit the release notes and tag (e.g., `git tag -a v0.2.0 -m "Release"`).

## 2. Zenodo DOI steps

1. Connect the GitHub repository to Zenodo (https://zenodo.org/account/settings/github/).
2. Push a GitHub release (GitHub → Releases → Draft a new release).
3. Zenodo automatically archives the release and provides a DOI.
4. Update `CITATION.cff` and `README.md` with the new DOI.

## 3. Publish bundle

1. Run
   ```bash
   make full RAD_MAP=/path/to/radiation.fits KAPPA_MAP=/path/to/kappa.fits SEED=123
   ```
   to generate figures and the publish JSON.
2. Package the release artifacts (e.g., `zip -r lightmatter-v0.2.0.zip results/full_run docs DATA_SOURCES.md`).
3. Attach the archive to the GitHub release and upload to Zenodo.

## 4. Post-release

- Announce DOI in README.
- Update `docs/ROADMAP.md` if applicable.
- Create an issue for any follow-up work.
