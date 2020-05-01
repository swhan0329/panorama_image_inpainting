import numpy as np

from . import sphere_utils

def e2c(e_img, face_w=256, mode='bilinear', cube_format='dice'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''

    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = sphere_utils.xyzcube(face_w)
    uv = sphere_utils.xyz2uv(xyz)
    coor_xy = sphere_utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        sphere_utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = sphere_utils.cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = sphere_utils.cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap, cubelist = sphere_utils.cube_h2dice(cubemap)
    else:
        raise NotImplementedError()

    return cubemap, cubelist