:mod:`dtmm.data_viewer`
=======================

.. py:module:: dtmm.data_viewer

.. autoapi-nested-parse::

   Created on Mon Mar 19 10:32:48 2018

   @author: andrej



Module Contents
---------------

.. function:: plot_material(data, eps=None, center=False, xlim=None, ylim=None, zlim=None, ax=None)

   Plots material (in color) of the optical data.

   :param data: A valid optical data tuple, eps array
   :type data: optical_data or material (eps)
   :param eps: Specifies which eps values to plot. If not given, all data is plotted.
   :type eps: array or None, optional
   :param center: Whether to view coordinates from the center of the box.
   :type center: bool, optional
   :param xlim: A tuple describing the coordinate limits in the x direction (width).
   :type xlim: (low, high) or None, optional
   :param ylim: A tuple describing the coordinate limits in the y direction (height).
   :type ylim: (low, high) or None, optional
   :param zlim: A tuple describing the coordinate limits in the z direction (layer index).
   :type zlim: (low, high) or None, optional
   :param ax: If specified, plot to ax.
   :type ax: matplotlib.axes or None, optional


.. function:: plot_director(director, fig=None, center=False, xlim=None, ylim=None, zlim=None, ax=None)

   Plots director data.

   :param director: A valid optical data tuple, eps array
   :type director: optical_data or material (eps)
   :param eps: Specifies which eps values to plot. If not given, all data is plotted.
   :type eps: array or None, optional
   :param center: Whether to view coordinates from the center of the box.
   :type center: bool, optional
   :param xlim: A tuple describing the coordinate limits in the x direction (width).
   :type xlim: (low, high) or None, optional
   :param ylim: A tuple describing the coordinate limits in the y direction (height).
   :type ylim: (low, high) or None, optional
   :param zlim: A tuple describing the coordinate limits in the z direction (layer index).
   :type zlim: (low, high) or None, optional
   :param ax: If specified, plot to ax.
   :type ax: matplotlib.axes or None, optional


.. function:: plot_angles(data, **kwargs)

   Plots eps angles for optical data or angles data.

   :param data: A valid optical data tuple, eps array
   :type data: optical_data or material (eps)
   :param eps: Specifies which eps values to plot. If not given, all data is plotted.
   :type eps: array or None, optional
   :param center: Whether to view coordinates from the center of the box.
   :type center: bool, optional
   :param xlim: A tuple describing the coordinate limits in the x direction (width).
   :type xlim: (low, high) or None, optional
   :param ylim: A tuple describing the coordinate limits in the y direction (height).
   :type ylim: (low, high) or None, optional
   :param zlim: A tuple describing the coordinate limits in the z direction (layer index).
   :type zlim: (low, high) or None, optional
   :param ax: If specified, plot to ax.
   :type ax: matplotlib.axes or None, optional


