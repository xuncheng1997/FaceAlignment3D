from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time

from .cython import mesh_core_cython


def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''

    # initial 
    depth_buffer = np.zeros([h, w]) - 999999.  # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype=np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype=np.float32)  #

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()

    mesh_core_cython.rasterize_triangles_core(
        vertices, triangles,
        depth_buffer, triangle_buffer, barycentric_weight,
        vertices.shape[0], triangles.shape[0],
        h, w)


def render_colors(vertices, triangles, colors, h, w, c=3, BG=None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width  
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    '''

    # initial 
    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    ###
    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c=3, mapping_type='nearest', BG=None):
    ''' render mesh with texture map
    Args:
        vertices: [3, nver]
        triangles: [3, ntri]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    # initial 
    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG

    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)

    # -> C order
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    texture = texture.astype(np.float32).copy()
    tex_coords = tex_coords.astype(np.float32).copy()
    tex_triangles = tex_triangles.astype(np.int32).copy()

    mesh_core_cython.render_texture_core(
        image, vertices, triangles,
        texture, tex_coords, tex_triangles,
        depth_buffer,
        vertices.shape[0], tex_coords.shape[0], triangles.shape[0],
        h, w, c,
        tex_h, tex_w, tex_c,
        mt)
    return image
