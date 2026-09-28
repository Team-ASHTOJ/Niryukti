# Publish VANTAGE on PyPI and npm

## License

The project uses **AGPL-3.0-only**, chosen to keep covered modified
redistributions open and require corresponding source for users interacting
with modified network deployments. It permits copying, modification and
commercial use; it does not promise to prevent imitation or every independent
proprietary integration. Third-party code retains its own licenses in NOTICE.
Contributors must have the rights to submit code under the project license.

Primary reference: https://www.gnu.org/licenses/agpl-3.0.en.html

Registry owner usernames supplied by the team: **shoaibssm** on PyPI and npm.
These are distinct from the GitHub repository owner **shoaib2000857**.

## PyPI: one-time owner setup

1. Create/sign into a PyPI account with two-factor authentication.
2. In account Publishing settings, add a **pending GitHub publisher**:
   - Project name: `niryukti`
   - Owner: `shoaib2000857`
   - Repository: `Vantage`
   - Workflow filename: `release.yml`
   - Environment: `pypi`
3. In GitHub repository settings, create environment `pypi`; configure a required
   reviewer if your GitHub plan supports it.
4. Push the release code, run **Build and publish VANTAGE packages** with
   target `build`, inspect the artifacts/tests, then run with target `pypi`
   for the first Python upload (`both` after npm OIDC setup). Only successful Python and Node build/test jobs allow publication.

No PyPI API token is needed with this OIDC setup. The pending publisher creates
an unclaimed project on the first successful upload; name availability is not
reserved merely by a local package name.

https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## npm: first publication and subsequent OIDC setup

npm trusted-publisher configuration is attached to an existing package. For the
first release, enable account 2FA/passkey and sign into your own npm account
locally (do not send credentials in chat), then publish the reviewed source tarball:

```sh
npm login --registry=https://registry.npmjs.org/ --auth-type=web
npm whoami --registry=https://registry.npmjs.org/
npm publish /absolute/path/to/niryukti-0.2.0.tgz --access public
```

Complete any npm account/2FA prompts in your own terminal/browser. The workflow's
npm publish job will fail until package ownership and OIDC are configured.

In `niryukti` package settings, configure GitHub trusted publishing:
owner `shoaib2000857`, repository `Vantage`, workflow `release.yml`, environment
`npm`. Create that GitHub environment as well. Allow direct `npm publish` for
this workflow. Node 24 runners supply a sufficiently recent npm CLI; current npm
OIDC requires npm >=11.5.1 and Node >=22.14.0.

If the unscoped name is claimed, choose an account/organization-scoped name,
update package metadata and smoke-test import names, then rebuild before upload.

https://docs.npmjs.com/trusted-publishers/

## Distribution contract

Initial public wheels target **manylinux 2.28 x86_64**, CPU/FP64, Python 3.10+.
They include both our native executable and shared C ABI; the source tarball
contains corresponding C++ and Python source plus required numerical headers.
The release workflow runs installed Python solves inside the wheel build/test
process and repairs library dependencies with cibuildwheel/auditwheel. This
workflow must actually pass before advertising portable wheel availability.
Other platforms can build the sdist with CMake 3.24+ and a C++20 compiler; they
are not promised prebuilt binaries. CUDA users can supply their own CUDA build
via `VANTAGE_BINARY`/`VANTAGE_LIBRARY`. npm currently builds the bundled source
locally; it needs CMake/compiler and install-script approval where applicable.

Wheel, source and npm tarballs include LICENSE and NOTICE. Preserve release
source alongside binaries. Do not upload the Arch-host wheel as a portable
manylinux wheel. Public version numbers cannot be reused: bump both Python/npm
metadata and CMake project version for later releases.

After publishing, verify from clean directories:

```sh
python -m venv /tmp/vantage-public-test
/tmp/vantage-public-test/bin/pip install niryukti
/tmp/vantage-public-test/bin/niryukti devices
mkdir /tmp/vantage-public-node-test && cd /tmp/vantage-public-node-test
npm install niryukti
node node_modules/niryukti/install.js
node node_modules/niryukti/smoke.cjs
```

These commands describe the post-publication test, not evidence that upload has
already happened. No passwords, recovery codes, API tokens or signing keys
should be placed in source files, documentation or chat.

The GitHub repository is currently private. Publishing packages does not change
repository visibility. Public source tarballs/npm packages must include the
corresponding source; a private GitHub link alone is insufficient. npm provenance
availability differs for private repositories; do not advertise provenance until
the registry confirms it. No repository visibility setting has been changed.
