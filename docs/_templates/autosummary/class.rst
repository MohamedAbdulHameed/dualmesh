{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:

{%- set own_attributes = [] %}
{%- for item in attributes %}{% if not item.startswith("_") %}{% set _ = own_attributes.append(item) %}{% endif %}{% endfor %}
{%- set own_methods = [] %}
{%- for item in methods %}{% if not item.startswith("_") %}{% set _ = own_methods.append(item) %}{% endif %}{% endfor %}
{% if own_attributes %}

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:
{% for item in own_attributes %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% if own_methods %}

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
{% for item in own_methods %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
