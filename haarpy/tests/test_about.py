# Copyright 2024 Polyquantique

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
about tests
"""

import contextlib
import io
import re

import haarpy as ap


def test_about():
    """
    about: Tests if the about string prints correctly.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ap.about()
    out = f.getvalue().strip()

    assert "Python version:" in out
    pl_version_match = re.search(r"Haarpy version:\s+([\S]+)\n", out).group(1)
    assert ap.version() in pl_version_match
    assert "SymPy version" in out
