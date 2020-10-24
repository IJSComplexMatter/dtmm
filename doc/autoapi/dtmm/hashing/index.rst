:mod:`dtmm.hashing`
===================

.. py:module:: dtmm.hashing

.. autoapi-nested-parse::

   Taken from dask.hasking



Module Contents
---------------

.. function:: hash_buffer(buf, hasher=None)

   Hash a bytes-like (buffer-compatible) object.  This function returns
   a good quality hash but is not cryptographically secure.  The fastest
   available algorithm is selected.  A fixed-length bytes object is returned.


.. function:: hash_buffer_hex(buf, hasher=None)

   Same as hash_buffer, but returns its result in hex-encoded form.


