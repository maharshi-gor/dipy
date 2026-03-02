"""SH Glyph Slicer Visualization for Skyline.

This module provides an ``SHGlyph3D`` visualization class that wraps
the interactive SH glyph slicer for use in the Skyline viewer.

The slicer renders spherical harmonic (SH) glyphs from dMRI data
with interactive per-axis slice control via imgui widgets.
"""

from fury.actor import Group
import numpy as np

from dipy.viz.sh_billboard import sph_glyph_billboard
from dipy.viz.skyline.UI.elements import (
    thin_slider,
)
from dipy.viz.skyline.render.renderer import Visualization


class SHSlicer:
    """Interactive SH glyph slicer for 3D dMRI volumes.

    One billboard actor is created per Z-slice (and optionally per X/Y slice).
    Visibility is controlled by toggling ``actor.visible``.

    Parameters
    ----------
    coeffs_4d : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    voxel_sizes : tuple of 3 floats
        Physical voxel sizes in mm.
    scale : float
        Per-glyph scale factor.
    l_max : int
        Maximum SH order.
    spacing : float
        Spacing multiplier between glyphs.
    lut_res : int
        LUT resolution for billboard precomputation.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    basis_type : str
        SH basis convention (``"standard"``, ``"descoteaux07"``,
        ``"tournier07"``).
    color_type : str
        Colour mapping type (``"orientation"``, ``"radius"``).
    """

    def __init__(
        self,
        coeffs_4d,
        voxel_sizes=(1.0, 1.0, 1.0),
        scale=1.0,
        l_max=8,
        spacing=1.0,
        lut_res=28,
        mask=None,
        basis_type="standard",
        color_type="orientation",
    ):
        self.coeffs_4d = coeffs_4d
        self.shape = coeffs_4d.shape[:3]
        self.n_coeffs = coeffs_4d.shape[-1]
        self.voxel_sizes = np.array(voxel_sizes)
        self.scale = scale
        self.l_max = l_max
        self.spacing = spacing
        self.lut_res = lut_res
        self.mask = mask
        self.basis_type = basis_type
        self.color_type = color_type

        # ref_size = self.voxel_sizes.min()
        self.spacing_ratios = self.voxel_sizes

        self._z_slice_actors = [None] * self.shape[2]
        self._x_slice_actors = [None] * self.shape[0]
        self._y_slice_actors = [None] * self.shape[1]

        self._built = False
        self.actor = Group()

    def build(self, axes="z"):
        """Build slice actors for the requested axes.

        Parameters
        ----------
        axes : str
            ``'z'``, ``'x'``, ``'y'``, ``'xy'``, ``'xz'``, ``'yz'``,
            ``'xyz'``, or ``'all'``.

        Returns
        -------
        Group
            Root actor group.
        """
        if self._built:
            return self.actor

        axes = axes.lower()
        if axes == "all":
            axes = "xyz"

        total_glyphs = 0

        if "z" in axes:
            for z in range(self.shape[2]):
                sl = self.coeffs_4d[:, :, z, :]
                a, n = self._build_slice_actor(sl, z, axis="z")
                if a is not None:
                    self._z_slice_actors[z] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "x" in axes:
            for x in range(self.shape[0]):
                sl = self.coeffs_4d[x, :, :, :]
                a, n = self._build_slice_actor(sl, x, axis="x")
                if a is not None:
                    self._x_slice_actors[x] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "y" in axes:
            for y in range(self.shape[1]):
                sl = self.coeffs_4d[:, y, :, :]
                a, n = self._build_slice_actor(sl, y, axis="y")
                if a is not None:
                    self._y_slice_actors[y] = a
                    self.actor.add(a)
                    total_glyphs += n

        self._built_axes = axes
        self._built = True
        return self.actor

    def _build_slice_actor(self, slice_coeffs, slice_idx, axis="z"):
        """Create a billboard actor for one slice."""
        sx, sy, sz = self.spacing_ratios * self.spacing

        if axis == "z":
            nx, ny = slice_coeffs.shape[0], slice_coeffs.shape[1]
            ii, jj = np.meshgrid(
                np.arange(nx, dtype=np.float32),
                np.arange(ny, dtype=np.float32),
                indexing="ij",
            )
            positions = np.column_stack(
                (
                    ii.ravel() * sx,
                    jj.ravel() * sy,
                    np.full(nx * ny, slice_idx * sz, dtype=np.float32),
                )
            )
        elif axis == "x":
            ny, nz = slice_coeffs.shape[0], slice_coeffs.shape[1]
            jj, kk = np.meshgrid(
                np.arange(ny, dtype=np.float32),
                np.arange(nz, dtype=np.float32),
                indexing="ij",
            )
            positions = np.column_stack(
                (
                    np.full(ny * nz, slice_idx * sx, dtype=np.float32),
                    jj.ravel() * sy,
                    kk.ravel() * sz,
                )
            )
        elif axis == "y":
            nx, nz = slice_coeffs.shape[0], slice_coeffs.shape[1]
            ii, kk = np.meshgrid(
                np.arange(nx, dtype=np.float32),
                np.arange(nz, dtype=np.float32),
                indexing="ij",
            )
            positions = np.column_stack(
                (
                    ii.ravel() * sx,
                    np.full(nx * nz, slice_idx * sy, dtype=np.float32),
                    kk.ravel() * sz,
                )
            )

        flat_coeffs = slice_coeffs.reshape(-1, self.n_coeffs)

        valid_mask = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            if axis == "z":
                sm = self.mask[:, :, slice_idx].ravel()
            elif axis == "x":
                sm = self.mask[slice_idx, :, :].ravel()
            else:
                sm = self.mask[:, slice_idx, :].ravel()
            valid_mask &= sm

        if not np.any(valid_mask):
            return None, 0

        valid_coeffs = flat_coeffs[valid_mask]
        valid_positions = positions[valid_mask]

        glyph = sph_glyph_billboard(
            valid_coeffs,
            centers=valid_positions,
            scale=self.scale,
            l_max=self.l_max,
            basis_type=self.basis_type,
            color_type=self.color_type,
            use_precomputation=True,
            use_bicubic=False,
            lut_res=self.lut_res,
        )
        return glyph, len(valid_coeffs)

    def set_z_range(self, z_min, z_max):
        for z, a in enumerate(self._z_slice_actors):
            if a is not None:
                a.visible = z_min <= z < z_max

    def set_z_slice(self, z):
        self.set_z_range(z, z + 1)

    def show_all_z(self):
        self.set_z_range(0, self.shape[2])

    def set_x_range(self, x_min, x_max):
        for x, a in enumerate(self._x_slice_actors):
            if a is not None:
                a.visible = x_min <= x < x_max

    def set_x_slice(self, x):
        self.set_x_range(x, x + 1)

    def show_all_x(self):
        self.set_x_range(0, self.shape[0])

    def set_y_range(self, y_min, y_max):
        for y, a in enumerate(self._y_slice_actors):
            if a is not None:
                a.visible = y_min <= y < y_max

    def set_y_slice(self, y):
        self.set_y_range(y, y + 1)

    def show_all_y(self):
        self.set_y_range(0, self.shape[1])

    def set_range(self, axis, min_idx, max_idx):
        axis = axis.lower()
        if axis == "x":
            self.set_x_range(min_idx, max_idx)
        elif axis == "y":
            self.set_y_range(min_idx, max_idx)
        elif axis == "z":
            self.set_z_range(min_idx, max_idx)

    def hide_all(self, axis=None):
        if axis is None:
            for actors in (
                self._x_slice_actors,
                self._y_slice_actors,
                self._z_slice_actors,
            ):
                for a in actors:
                    if a is not None:
                        a.visible = False
        else:
            self.set_range(axis, 0, 0)


class SHGlyph3D(Visualization):
    """SH glyph slicer visualization for the Skyline viewer.

    Wraps :class:`SHSlicer` and exposes imgui widgets for interactive
    per-axis slice control, scale, and SH-order adjustment.

    Parameters
    ----------
    name : str
        Display name in the UI sidebar.
    coeffs : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    affine : ndarray (4, 4), optional
        Voxel-to-world affine.  Voxel sizes are extracted from the first
        three diagonal elements.
    render_callback : callable, optional
        Called after parameter changes to trigger a scene re-render.
    scale : float
        Initial per-glyph scale.
    l_max : int
        Maximum SH order.
    lut_res : int
        LUT resolution.
    basis_type : str
        SH basis convention (``"standard"``, ``"descoteaux07"``,
        ``"tournier07"``).
    color_type : str
        Colour mapping.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    """

    def __init__(
        self,
        name,
        coeffs,
        *,
        affine=None,
        render_callback=None,
        scale=1.0,
        l_max=8,
        lut_res=8,
        basis_type="standard",
        color_type="orientation",
        mask=None,
    ):
        super().__init__(name, render_callback)
        # print("Affine:\n", affine)
        # if affine is not None:
        #     voxel_sizes = tuple(np.abs(np.diag(affine)[:3]).astype(float))
        # else:
        voxel_sizes = (1.0, 1.0, 1.0)

        self._slicer = SHSlicer(
            coeffs,
            voxel_sizes=voxel_sizes,
            scale=scale,
            l_max=l_max,
            lut_res=lut_res,
            mask=mask,
            basis_type=basis_type,
            color_type=color_type,
        )
        self._slicer.build(axes="z")

        self.shape = coeffs.shape[:3]
        self._x_idx = self.shape[0] // 2
        self._y_idx = self.shape[1] // 2
        self._z_idx = self.shape[2] // 2

        self._slicer.set_x_slice(self._x_idx)
        self._slicer.set_y_slice(self._y_idx)
        self._slicer.set_z_slice(self._z_idx)

        self.bounds = [
            [0, 0, 0],
            list(self.shape),
        ]
        # self._slicer.actor.transform(affine)

    @property
    def actor(self):
        return self._slicer.actor

    def render_widgets(self):
        """Render per-axis slice control widgets in the Skyline sidebar."""
        changed, new_x = thin_slider(
            "X Slice",
            self._x_idx,
            0,
            self.shape[0] - 1,
            value_type="int",
            text_format=".0f",
            step=1,
        )
        if changed:
            self._x_idx = int(new_x)
            self._slicer.set_x_slice(self._x_idx)

        changed, new_y = thin_slider(
            "Y Slice",
            self._y_idx,
            0,
            self.shape[1] - 1,
            value_type="int",
            text_format=".0f",
            step=1,
        )
        if changed:
            self._y_idx = int(new_y)
            self._slicer.set_y_slice(self._y_idx)

        changed, new_z = thin_slider(
            "Z Slice",
            self._z_idx,
            0,
            self.shape[2] - 1,
            value_type="int",
            text_format=".0f",
            step=1,
        )
        if changed:
            self._z_idx = int(new_z)
            self._slicer.set_z_slice(self._z_idx)
