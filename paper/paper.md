---
title: 'Haarpy: a Python library for Weingarten calculus and integration of classical compact groups and ensembles'
tags:
  - random matrices
  - Haar measure
  - classical compact groups
authors:
 - name: Yanic Cardin
   orcid: 0009-0005-6858-705X
   affiliation: 1
 - name: Hubert de Guise
   orcid: 0000-0002-1904-4287
   affiliation: 2
 - name: Nicolás Quesada
   orcid: 0000-0002-0175-1688
   affiliation: 1
affiliations:
 - name: Département de génie physique, École polytechnique de Montréal, Montréal, QC H3T 1J4, Canada
   index: 1
 - name: Department of Physics, Lakehead University, Thunder Bay, ON P7B 5E1, Canada
   index: 2
date: 15 June 2026
bibliography: paper.bib
---

# Summary

[Haarpy](https://github.com/polyquantique/haarpy) is a Python library for the symbolic calculation of Weingarten functions and related integals (also called moments or averages) of matrix ensembles under their Haar measure.
There is a multiplicity of applications, both in physics and in mathematics, requiring the computation of averages of various polynomial functions over the classical compact groups (unitary, orthogonal and symplectic), their associated circular ensembles, the group of permutation and centered permutation matrices, and the quantum groups (free symmetric and free orthogonal) [@potters2020first].
Rather than relying on Monte Carlo simulation, which provides approximate results [@mezzadri2006generate], Haarpy enables exact analytical computation of such moments.

Under the hood, Haarpy reduces the computation of integrals over the relevant ensembles to Weingarten calculus.
This field has grown rapidly over the past few decades, following the foundational work of Collins, who introduced the terminology [@collins2003moments] [@collins2006integration].
This machinery is exposed through functions such as `weingarten_unitary`, `weingarten_orthogonal` and `weingarten_symplectic` while the associated integrals are accessed through functions such as `haar_integral_unitary`, `haar_integral_orthogonal` and `haar_integral_symplectic`.
A full description of Haarpy's functionalities can be found in the library [documentation](https://haarpy.readthedocs.io).

Built on top of the SymPy symbolic engine, Haarpy allows users to retain symbolic parameters and derive general formulas applicable across entire classes of problems [@10.7717/peerj-cs.103].

# Statement of need

The computation of averages over Haar-distributed random matrices is of growing importance in quantum information theory [@martinez2024linear; @cardin2024photon; @turkeshi2025magic], random matrix theory [@bordenave2024strong; @daigle2025mixed], and statistical physics [@cotler2017chaos; @potters2020first].
These averages describe the typical behavior of complex systems, including entanglement properties, randomness, and statistical correlations.

Standard approaches typically rely either on Monte Carlo simulation, which approximates integrals through sampling, or on manual analytical derivations, which often involves intricate combinatorics.

Haarpy is intended for researchers throughout physics, mathematicians working in representation theory, and researchers studying random matrix models.
By making Weingarten calculus readily accessible in software, Haarpy enhances reproducibility, reduces computational overhead, and lowers the barrier to performing rigorous analytical calculations.

# State of the field
Several software packages support symbolic Haar integration and related calculations.
Early implementations were developed in Mathematica, while more recent projects such as RTNI2 [@fukuda2023symbolically] and IntegrateUnitary.jl [@pawela2026integrateunitary] provide advanced functionality for tensor-network-based computations and symbolic integration within their respective ecosystems.
In particular, RTNI2 offers a powerful framework for diagrammatic calculations based on tensor network methods, while IntegrateUnitary.jl provides symbolic tools for Haar integration in Julia.

Haarpy complements these efforts by providing an open-source, Python-native implementation tightly integrated with the scientific Python ecosystem and with SymPy.
In addition to supporting integrations over the unitary and orthogonal groups, the package implements moment calculations for symplectic groups, circular ensembles, permutation and centered permutation matrices, and the free symmetric and free orthogonal groups within a unified interface.
To the best of our knowledge, Haarpy is the first Python package to provide direct computation of Haar integrals within a unified symbolic framework, rather than requiring users to reconstruct integrals from intermediate outputs.

The package places a strong emphasis on verification, with an extensive test suite covering both symbolic and numerical computations.
Rather than extending an existing codebase, the underlying Weingarten calculus machinery has been re-implemented in Python to provide a tool that integrates naturally with widely used Python scientific workflows.
For classical compact groups, Haarpy additionally provides an alternative moment computation method based on the recursive algorithm of Gorin [@gorin2008monomial], which does not rely on Weingarten calculus.
In many settings, this latter approach can offer improved performance compared to Weingarten-based computations.

# Software design
Haarpy is designed with a focus on mathematical fidelity, symbolic flexibility, and usability.
A key consideration is the trade-off between symbolic and numerical computation: while purely numerical methods can offer speed, they often obscure structure and reduce reproducibility, whereas symbolic computation preserves the exact algebraic form needed for derivations and verification.
For this reason, the implementation is deliberately built around a lightweight symbolic stack centered on SymPy, avoiding heavier external dependencies in order to keep the system transparent, portable, and easier to validate.
At the same time, efficiency is addressed through selective caching of intermediate combinatorial quantities, which reduces redundant work without relying on a precomputed database of results.
In particular, moments are computed on demand rather than retrieved from tabulated Weingarten values.

Core Weingarten functionality is complemented by auxiliary tools, including:

- Implementation of the Murnaghan-Nakayama rule for characters of irreducible representations of the symmetric group [@james2006representation],

- Functions for computing dimensions of irreducible representations of the symmetric and unitary groups [@james2006representation],

- Generators yielding the permutations of the hyperoctahedral group and Young group [@macdonald1998symmetric],

- Generators yielding the sets of non-crossing partitions and non-crossing pairings [@nica2006lectures].

This modular design enables reuse in broader algebraic and combinatorial computations.
Exposing group-theoretic primitives increases usability for diverse research applications.

# Research impact statement

Haarpy has been cited in an arXiv preprint, indicating external use beyond the original development context [@duschenes2025moments].
The library is also used in ongoing collaborative research on Weingarten calculus and related random matrix theory problems.
The project is publicly available on [GitHub](https://github.com/polyquantique/haarpy) and is accompanied by comprehensive documentation hosted on [Read the Docs](https://haarpy.readthedocs.io).
The repository includes continuous integration with automated testing, with all tests passing and full test coverage reported via standard coverage tooling.
These components provide a reproducible and verifiable implementation of algorithms for integration over a wide range of ensembles equipped with the Haar measure.

# AI usage disclosure

No AI tools were used in the development of Haarpy.

# References
