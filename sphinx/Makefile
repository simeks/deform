# README
# ======
#
# Requirements: sphinx, apidoc, napoleon, m2r
#
# This Makefile performs a clean build of the docs. The API is extracted with
# sphinx-apidoc, the GitHub readme is converted to rst (so no duplicate text),
# and if images from the Sphinx resources are included in the GitHub readme,
# their relative path is converted when including the readme in the documentation.
#
# Caveats: the markup in the GitHub readme must be supported by m2r for rst conversion. 

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pydeform
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	rm -rf _build/*
	SPHINX_APIDOC_OPTIONS=members,undoc-members,imported-members \
			sphinx-apidoc -f -o source ../pydeform
	python3 -m m2r ../README.md # convert GitHub README to rst
	mv ../README.rst readme.rst
	sed -i 's,https://.*sphinx/\([^?]\+\)?raw=true,\1,g' readme.rst # convert image paths
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
