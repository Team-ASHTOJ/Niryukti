# Eigen numerical kernel

Pinned Eigen 5.0.0 numerical headers provide sparse LU basis-system factorization for the independently implemented revised primal simplex algorithm. This is numerical linear algebra, not an external mathematical optimization solver. No Eigen unsupported optimization modules are included.

Source/archive SHA-256: [provenance](eigen/provenance.json). Retain the upstream copying notices and each header's copyright/licence. Primary documentation: https://libeigen.gitlab.io/ and https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1SparseLU.html .

`-DVANTAGE_SPARSE_LU=OFF` builds the original bounded dense-basis numerical kernel instead. No network access is required to build either configuration.
