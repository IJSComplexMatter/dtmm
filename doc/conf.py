# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'dtmm'
copyright = '2018, Andrej Petelin'
author = 'Andrej Petelin'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = '0.6.0'

numfig = True

import sys,os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
# custom matplotlib plot_template

if sys.argv[2] in ('latex', 'latexpdf'):
    plot_template = """
{% for img in images %}
.. figure:: {{ build_dir }}/{{ img.basename }}.pdf
    {%- for option in options %}
    {{ option }}
    {% endfor %}
    
    \t{{caption}}
{% endfor %}
"""

else:
    plot_template = """
{% for img in images %}

.. figure:: {{ build_dir }}/{{ img.basename }}.png
    {%- for option in options %}
    {{ option }}
    {% endfor %}

    \t{% if html_show_formats and multi_image -%}
    (
    {%- for fmt in img.formats -%}
    {%- if not loop.first -%}, {% endif -%}
    `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    )
    {%- endif -%}

    {{ caption }} {% if source_link or (html_show_formats and not multi_image) %} (
{%- if source_link -%}
`Source code <{{ source_link }}>`__
{%- endif -%}
{%- if html_show_formats and not multi_image -%}
    {%- for img in images -%}
    {%- for fmt in img.formats -%}
        {%- if source_link or not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    {%- endfor -%}
{%- endif -%}
)
{% endif %}
{% endfor %}
"""


if sys.argv[2] in ('latex', 'latexpdf'):
    plot_template = """
{% for img in images %}
.. figure:: {{ build_dir }}/{{ img.basename }}.pdf
    {%- for option in options %}
    {{ option }}
    {% endfor %}
    
    \t{{caption}}
{% endfor %}
"""

else:
    plot_template = """
{% for img in images %}

.. figure:: {{ build_dir }}/{{ img.basename }}.png
    {%- for option in options %}
    {{ option }}
    {% endfor %}

    \t{% if html_show_formats and multi_image -%}
    (
    {%- for fmt in img.formats -%}
    {%- if not loop.first -%}, {% endif -%}
    `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    )
    {%- endif -%}

    {{ caption }} {% if source_link or (html_show_formats and not multi_image) %} (
{%- if source_link -%}
`Source code <{{ source_link }}>`__
{%- endif -%}
{%- if html_show_formats and not multi_image -%}
    {%- for img in images -%}
    {%- for fmt in img.formats -%}
        {%- if source_link or not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    {%- endfor -%}
{%- endif -%}
)
{% endif %}
{% endfor %}
"""






# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.githubpages',
    "sphinx.ext.doctest",
    "sphinx.ext.imgmath",
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    #"sphinx.ext.jsmath",
    #'matplotlib.sphinxext.plot_directive',
    'plot_directive'
]

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
               "sphinx.ext.imgmath",
    'sphinx.ext.napoleon',
	"sphinx.ext.doctest",
    'sphinx.ext.inheritance_diagram',    
	'autoapi.extension',
	'matplotlib.sphinxext.plot_directive'
    ]

autoapi_keep_files = False
napoleon_numpy_docstring = True

autoapi_dirs = ['../dtmm']
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'special-members']
autoapi_options = ['members', 'show-inheritance']
autoapi_ignore = ["*/test/*.py","*/test"]

numfig = True

import os 

doctest_global_setup = '''
try:
    import numpy as np
    import dtmm
    from dtmm.fft import * 
    from dtmm.color import * 
    from dtmm.data import * 
    from dtmm.tmm import *
    from dtmm.jones4 import *
    from dtmm.jones import *
    from dtmm.window import *
    from dtmm.linalg import *
    from dtmm.rotation import *
except ImportError:
	pass

field_in = (np.ones((1,4,6,6))+0j, np.array((3.,)), 100)
field_data_in = field_in
field_data_out = (np.ones((1,4,6,6))+0j, np.array((3.,)), 100)
field_bulk_data = (np.ones((1,2,1,4,6,6))+0j, np.array((3.,)), 100)
field = field_in
optical_data = np.array((1.,)), np.ones((1,6,6,3))*2+0j, np.zeros((1,6,6,3))
data = optical_data
NLAYERS, HEIGHT, WIDTH = 1,6,6
WAVELENGTHS = [500]
PIXELSIZE = 200

'''


plot_working_directory = "examples"#os.path.abspath("../examples")

imgmath_image_format = "svg"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'default'
#html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []



