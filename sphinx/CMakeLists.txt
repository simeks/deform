find_package(Sphinx REQUIRED)
 
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

set(SPHINX_STATIC "${CMAKE_CURRENT_SOURCE_DIR}/static")
set(SPHINX_TEMPLATES "${CMAKE_CURRENT_SOURCE_DIR}/templates")
 
configure_file(
    "conf.py"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)
 
add_custom_target(docs ALL
    ${SPHINX_EXECUTABLE}
        -q -b html
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML docs with Sphinx")

