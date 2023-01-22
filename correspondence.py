import numpy as np
import obj_io
import json
import math


def calcBarycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    def my_dot(x, y):
        return np.sum(x * y, axis=-1)
    d00 = my_dot(v0, v0)
    d01 = my_dot(v0, v1)
    d11 = my_dot(v1, v1)
    d20 = my_dot(v2, v0)
    d21 = my_dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    # TODO: avoid zero div
    inv_denom = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    u = 1.0 - v - w

    return u, v, w


def point_line_distance(p, v0, v1):
    if p.shape != v0.shape != v1.shape:
        raise ValueError("All points must have the same number of coordinates")

    def my_dot(x, y):
        return np.sum(x * y, axis=-1)

    kEpsilon = 1e-10
    v1v0 = v1 - v0
    l2 = my_dot(v1v0, v1v0)  # |v1 - v0|^2
    invalid_mask = l2 <= kEpsilon

    t = my_dot(v1v0, p - v0) / l2
    t = np.clip(t, 0.0, 1.0)
    t = t[..., None]
    p_proj = v0 + t * v1v0
    delta_p = p_proj - p
    result = np.sqrt(my_dot(delta_p, delta_p))
    if np.sum(invalid_mask) > 0:
        result[invalid_mask] = my_dot(p - v1, p - v1)[invalid_mask]
    return result, p_proj


def point_triangle_distance(p, v0, v1, v2):
    if p.shape != v0.shape != v1.shape != v2.shape:
        raise ValueError("All points must have the same number of coordinates")

    e01_dist, p_proj0 = point_line_distance(p, v0, v1)
    e02_dist, p_proj1 = point_line_distance(p, v0, v2)
    e12_dist, p_proj2 = point_line_distance(p, v1, v2)

    dists = np.stack([e01_dist, e02_dist, e12_dist], axis=-1)
    points = np.stack(
        [p_proj0, p_proj1, p_proj2], axis=-1)

    edge_dists_min_indices = np.argmin(dists, axis=-1)[..., None]

    edge_dists_min = np.take_along_axis(dists, edge_dists_min_indices, axis=-1)
    edge_points_min = np.take_along_axis(
        points, np.repeat(edge_dists_min_indices, 3, axis=-1)[..., None], axis=-1)

    return edge_dists_min[..., 0], edge_points_min[..., 0]


def computeCorrespondence(src, dst, invalid_dst_face_mask=None, normal_deg_th=45.0):
    # Plane equation of dst faces
    dst_face_normals = dst.face_normals  # (DF, 3)
    v0 = dst.verts[dst.indices[..., 0]]
    v1 = dst.verts[dst.indices[..., 1]]
    v2 = dst.verts[dst.indices[..., 2]]
    dst_face_centers = (v0 + v1 + v2) / 3

    dst_face_d = - np.sum(dst_face_normals *
                          dst_face_centers, axis=1)  # (DF, 1 )

    # Batched src vertices
    # (SV, DF, 3)
    src_verts = np.repeat(src.verts[:, None], dst_face_d.shape[0], axis=1)

    # Distance from src vertex to dst face plane
    dist = np.sum(dst_face_normals * src_verts, axis=-1) + \
        dst_face_d  # (SV, DF, 1)

    if invalid_dst_face_mask is not None:
        dist[:, invalid_dst_face_mask] = np.inf

    abs_dist = np.abs(dist)
    # sorted_indices = np.argsort(abs_dist, axis=-1)  # [SV, DF]

    points_on_plane = - dist[..., None] * np.repeat(
        dst_face_normals[None, ], src_verts.shape[0], axis=0) + src_verts  # (SV, DF, 3)

    # Distance from src vertex to dst face edges
    # This is for fail safe
    edge_dist, edge_points = point_triangle_distance(src_verts, v0, v1, v2)

    # TODO: inf may cause these warnings
    # site-packages\numpy\core\fromnumeric.py:87: RuntimeWarning: invalid value encountered in reduce
    #   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    # correspondence.py:26: RuntimeWarning: invalid value encountered in multiply
    #   return np.sum(x * y, axis=-1)
    if invalid_dst_face_mask is not None:
        edge_dist[:, invalid_dst_face_mask] = np.inf

    abs_edge_dist = np.abs(edge_dist)
    #sorted_indices2 = np.argsort(abs_edge_dist, axis=-1)

    # Concat dist and indices
    merged_abs_dist = np.concatenate(
        [abs_dist, abs_edge_dist], axis=1)  # (SV, DF*2)
    DF = abs_edge_dist.shape[1]
    merged_sorted_indices = np.argsort(merged_abs_dist, axis=-1)

    # Compute barycentric
    # Plane points
    us, vs, ws = calcBarycentric(points_on_plane, v0, v1, v2)
    uvws = np.stack([us, vs, ws], axis=-1)
    # Edge points
    eus, evs, ews = calcBarycentric(edge_points, v0, v1, v2)
    euvws = np.stack([eus, evs, ews], axis=-1)

    # Select valid [0, 1] barycentric with minimum distance
    result = []

    def check_range_eps(val, min_val, max_val, eps=1e-6):
        if min_val - eps <= val and val <= max_val + eps:
            return True

    # TODO: Batch
    normal_rad_th = math.radians(normal_deg_th)
    normal_cos_th = np.cos(normal_rad_th)
    for vid in range(len(src_verts)):
        valid = False
        for j, merged_fid in enumerate(merged_sorted_indices[vid]):
            fid = merged_fid
            if merged_fid >= DF:
                fid -= DF
                dist_type = 'edge'
                d = edge_dist[vid][fid]
                u, v, w = euvws[vid][fid]
            else:
                dist_type = 'plane'
                d = dist[vid][fid]
                u, v, w = uvws[vid][fid]
            if dst.face_normals[fid].dot(src.normals[vid]) < normal_cos_th:
                continue
            if check_range_eps(u, 0, 1) and check_range_eps(v, 0, 1)\
                    and check_range_eps(w, 0, 1):
                valid = True
                u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
                result.append((int(fid), u, v, dist_type, d))
                break
        if not valid:
            # This may happen
            # if you set normal threshold with very different src and dst
            # Especially dst mesh is too small for src correspondences
            print(vid, 'is invalid!')
            result.append(None)
    return result


if __name__ == '__main__':
    src_path = './data/neutral_iphone_to_flame.obj'
    dst_path = './data/neutral_flame.obj'
    src = obj_io.loadObj(src_path)
    dst = obj_io.loadObj(dst_path)
    src.recomputeNormals()
    dst.recomputeNormals()

    invalid_dst_face_mask = None
    ignore_faces_paths = ['./data/eyeballs_faces.json',
                          './data/eyeholes_faces.json']
    ignore_fids = set()
    for ignore_face_path in ignore_faces_paths:
        with open(ignore_face_path, 'r') as fp:
            j = json.load(fp)
            ignore_fids = ignore_fids.union(set(j))
    invalid_dst_face_mask = np.zeros(dst.indices.shape[0], dtype=bool)
    invalid_dst_face_mask[list(ignore_fids)] = True

    corresp = computeCorrespondence(src, dst, invalid_dst_face_mask)
    wrap3_compat = {}
    for i in range(len(corresp)):
        name = str(i)
        if corresp[i] is not None:
            wrap3_compat[name] = corresp[i][:3]

    with open('./data/corresp_wrap3_named_points_on_triangle.json', 'w') as fp:
        json.dump(wrap3_compat, fp)

    with open('./data/corresp_all_info.json', 'w') as fp:
        json.dump(corresp, fp)

    with open('./data/src_corresp_line.obj', 'w') as fp:
        recomputed_points = []
        skipped = set()
        for i, c in enumerate(corresp):
            if c is None:
                skipped.add(i)
                continue
            fid, u, v, corresp_type, dist = c
            f = dst.indices[fid]
            v0 = dst.verts[f[0]]
            v1 = dst.verts[f[1]]
            v2 = dst.verts[f[2]]
            p = u * v0 + v * v1 + (1-u-v) * v2
            fp.write('v ' + str(p[0]) + ' ' +
                     str(p[1]) + ' ' + str(p[2]) + '\n')
        for i, p in enumerate(src.verts):
            if i in skipped:
                continue
            fp.write('v ' + str(p[0]) + ' ' +
                     str(p[1]) + ' ' + str(p[2]) + '\n')
        num = len(src.verts) - len(skipped)
        for i in range(1, num+1):
            fp.write('l ' + str(i) + ' ' + str(i + num) + '\n')
