# Version 0.0.5

### New features

`haar_integral()` function has been added to the library. This function takes sequences of matrix indices and the dimension of the unitary group as arguments and returns the Haar integral over the unitary group. See README for an example on how to use this function.

### Breaking changes

Certain functions have been renamed to improve clarity:
* `get_class()` -> `get_conjugacy_class()`
* `ssyt()` -> `semi_standard_young_tableaux()`
* `bad_mapping()` -> `proper_border_strip()`
* `murn_naka()` -> `murn_naka_rule()`
* `sn_dimension()` -> `irrep_dimension()`
* `ud_dimension()` -> `representation_dimension()`

### Improvements

README has been updated with the previous function name changes. Some typos have been fixed. Examples of unitary Haar integrals have also been added to the README.

Dependencies to numpy have been removed.

### Bug fixes

Fixed a bug in `murn_naka_rule()` where certain Young tableaux with 2x2 square values were flagged as proper border-strip tableaux. In order to do so, the function `bad_mapping()` (now called `border_strip_tableau()`) has been rewritten so that it now returns True if the input Young tableau is a valid border-strip tableau.

### Contributors

This release contains contributions from (in alphabetical order):

Yanic Cardin, Nicolás Quesada

# Version 0.0.4

### New features

### Improvements

### Bug fixes

Fixed a bug in `murn_naka()` where invalid tableau were not properly removed.

### Contributors

This release contains contributions from (in alphabetical order):

Yanic Cardin, Hubert de Guise, Matthew Duschenes, Nicolás Quesada

# Version 0.0.3

### New features

First release with Weingarten functions for the unitary group.

### Improvements

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

Yanic Cardin, Hubert de Guise, Nicolás Quesada
