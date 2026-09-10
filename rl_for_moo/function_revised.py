import numpy as np
from scipy.spatial import ConvexHull,Delaunay
import alphashape
from scipy.spatial.distance import cdist
from matplotlib.path import Path
from numpy.linalg import norm, inv
import sys
import traceback
import time

MODE = 2
def RotMatrix(angle, Mode):
    if Mode == 1:
        alpha = angle[0]
        beta = angle[1]
        gamma = angle[2]

        R = np.array([[np.cos(beta) * np.cos(gamma), -np.cos(beta) * np.sin(gamma), np.sin(beta)],
                      [np.sin(alpha) * np.sin(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
                       -np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
                       -np.sin(alpha) * np.cos(beta)],
                      [-np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
                       np.cos(alpha) * np.sin(beta) * np.sin(gamma) + np.sin(alpha) * np.cos(gamma),
                       np.cos(alpha) * np.cos(beta)]])
    elif Mode == 2:
        phi = angle[0]
        theta = angle[1]
        psi = angle[2]

        R = np.array([[np.cos(phi) * np.cos(theta) * np.cos(psi - phi) - np.sin(phi) * np.sin(psi - phi),
                       -np.cos(phi) * np.cos(theta) * np.sin(psi - phi) - np.sin(phi) * np.cos(psi - phi),
                       np.cos(phi) * np.sin(theta)],
                      [np.sin(phi) * np.cos(theta) * np.cos(psi - phi) + np.cos(phi) * np.sin(psi - phi),
                       -np.sin(phi) * np.cos(theta) * np.sin(psi - phi) + np.cos(phi) * np.cos(psi - phi),
                       np.sin(phi) * np.sin(theta)],
                      [-np.sin(theta) * np.cos(psi - phi), np.sin(theta) * np.sin(psi - phi), np.cos(theta)]])
    elif Mode == 3:
        alpha = angle[0]
        beta = angle[1]
        gamma = angle[2]

        R = np.array([[np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma),
                       -np.cos(alpha) * np.cos(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
                       np.cos(alpha) * np.sin(beta)],
                      [np.sin(alpha) * np.cos(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
                       -np.sin(alpha) * np.cos(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
                       np.sin(alpha) * np.sin(beta)],
                      [-np.sin(beta) * np.cos(gamma), np.sin(beta) * np.sin(gamma), np.cos(beta)]])

    return R

def paramInitialize(a, c, e, init_height, switch_pm):
    param = {}

    # Gravitational acceleration
    param['g'] = np.array([0, 0, 9.807])

    # Dimension parameters
    param['a'] = a
    param['init_height'] = init_height
    param['c'] = c
    param['e'] = e
    param['l_len'] = np.sqrt(init_height**2 + np.sum((a - c)**2, axis=0)).reshape((1,6))
    param['switch_pm'] = switch_pm
    # Dynamic parameters
    param['m_p'] = 0  # Platform mass
    param['I_p'] = np.diag([0, 0, 0])  # Platform moment of inertia
    param['Ipc'] = param['I_p']
    param['r_cmp_p'] = np.array([0, 0, 0])  # Center of mass position in platform coordinates
    param['m_l'] = 0  # Rod mass
    param['I_l'] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Rod moment of inertia
    param['m_sl'] = 0  # Slider mass

    return param

def density_sampling(points, distance):
    i = 0
    size_p = points.shape[0]
    while i != size_p:
        distance_i = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        points = np.vstack((points[i], points[distance_i >= distance]))

        size_p = points.shape[0]
        if i == size_p - 1:
            break
        else:
            i += 1

    return points

def isLineCrossPolygon(starts, ends, base_points, e):
    """
    判断直线是否穿过一个水平面多边形
    本函数不具有通用性，专为计算工作空间连杆是否与框架干涉而设计

    参数：
    starts (numpy.ndarray): 连杆起点，形状为 (3, 6)
    ends (numpy.ndarray): 连杆终点，形状为 (3, 6)
    base_points (numpy.ndarray): 丝杠起点，形状为 (3, 6)
    e (numpy.ndarray): 丝杠方向，形状为 (3, 6)

    输出：
    result (numpy.ndarray): 1x6，True为发生了干涉，False为没有发生干涉
    """
    result = np.zeros(6, dtype=bool)
    l_vecs = ends - starts  # 连杆向量

    for i in range(6):
        # 计算c在a角度顺时针还是逆时针
        cross_l = np.cross(np.append(base_points[:2, (i - 1) % 6], 0), np.append(ends[:2, i], 0))
        cross_r = np.cross(np.append(base_points[:2, (i + 1) % 6], 0), np.append(ends[:2, i], 0))

        if cross_l[2] >= 0 and cross_r[2] <= 0:
            continue
        elif cross_l[2] < 0:
            baseP_vec = base_points[:, (i - 1) % 6] - base_points[:, i]
            cross_0 = np.cross(np.append(l_vecs[:2, i], 0), np.append(baseP_vec[:2], 0))
            if cross_0[2] > 0:
                normal_vec = np.cross(baseP_vec, e[:, (i - 1) % 6])
                if np.dot(normal_vec, l_vecs[:, i]) < 0:
                    result[i] = True
        elif cross_r[2] > 0:
            baseP_vec = base_points[:, (i + 1) % 6] - base_points[:, i]
            cross_0 = np.cross(np.append(l_vecs[:2, i], 0), np.append(baseP_vec[:2], 0))
            if cross_0[2] < 0:
                normal_vec = np.cross(baseP_vec, e[:, (i + 1) % 6])
                if np.dot(normal_vec, l_vecs[:, i]) > 0:
                    result[i] = True

    return result


def ws_inside_boundaries(points1, points2):
    """
    Calculate the proportion of points from points2 that lie inside the convex hull of points1,
    and calculate the difference in volumes of their convex hulls.

    Args:
    - points1: A numpy array of points (n x 3) defining the first point cloud.
    - points2: A numpy array of points (m x 3) defining the second point cloud.

    Returns:
    - ws_insideProportion: The proportion of points2 that are inside the convex hull of points1.
    - ws_volumeDiff: The absolute difference in the volumes of the convex hulls of points1 and points2.
    """

    # Compute convex hull of points1
    hull1 = ConvexHull(points1)
    v1 = hull1.volume  # Volume of the convex hull of points1

    # Compute convex hull of points2
    hull2 = ConvexHull(points2)
    v2 = hull2.volume  # Volume of the convex hull of points2
    index = np.array(hull1.simplices)
    boundary_points = np.array(hull1.vertices)
    # Check if points from points2 are inside the convex hull of points1
    is_inside = isPointInsideBoundary_simple(points1, index, points2,boundary_points,2)
    # print(f"ws_insideBoundaries::sum_isInside: {np.sum(is_inside)}")

    p2_inside = points2[is_inside == 1]
    if p2_inside.shape[0] < 4:
        v3 = 0  # If fewer than 4 points inside, set v3 to 0
    else:
        hull3 = ConvexHull(p2_inside)
        v3 = hull3.volume  # Volume of the convex hull of points inside points2

    # Proportion of points2 inside points1's convex hull
    ws_insideProportion = v3 / v1 if v1 != 0 else 0

    # Absolute volume difference
    ws_volumeDiff = abs(v1 + v2 - 2 * v3)

    return ws_insideProportion, ws_volumeDiff
def isPointInsideBoundary_simple(points, convexHullIndices, testPoints, boundary_points,mode):
    if mode == 2:
        P = points
    elif mode == 1:
        P = boundary_points
    else:
        raise ValueError("Mode should be 1 or 2")

    # 创建 Delaunay 三角剖分
    tri = Delaunay(P[convexHullIndices.flatten()])

    # 使用 find_simplex 方法批量判断
    simplex = tri.find_simplex(testPoints)
    result = (simplex >= 0).astype(int)

    return result
def isPointInsideBoundary(points, convex_hull_indices, test_points, boundary_points,mode):
    #mode=1,alphashape;2:凸包
    num_test_points = test_points.shape[0]
    result = np.ones(num_test_points)

    c1, c4, c5 = 0, 0, 0

    if mode ==1:
        face_points_1 = boundary_points[convex_hull_indices[:, 0], :]
        face_points_2 = boundary_points[convex_hull_indices[:, 1], :]
        face_points_3 = boundary_points[convex_hull_indices[:, 2], :]
    elif mode==2:
        face_points_1 = points[convex_hull_indices[:, 0], :]
        face_points_2 = points[convex_hull_indices[:, 1], :]
        face_points_3 = points[convex_hull_indices[:, 2], :]
    face_mid_points = (face_points_1 + face_points_2 + face_points_3) / 3

    normal = np.cross(face_points_2 - face_points_1, face_points_3 - face_points_1)

    # 检验每个test_point是否在点云内部
    for i in range(num_test_points):
        test_point = test_points[i, :]
        # 找出点在其外侧的三角形，然后向这些三角形的中心发射线，看这条射线穿过的三角形的奇偶
        i_out = np.sum((face_mid_points - test_point) * normal, axis=1) < 0

        face_out_mid_points = face_mid_points[i_out, :]

        if face_out_mid_points.shape[0] == 0:
            continue

        for j in range(face_out_mid_points.shape[0]):
            x_range = np.array([min(face_out_mid_points[j, 0], test_point[0]), max(face_out_mid_points[j, 0], test_point[0])]) + np.array([-0.1, 0.1])
            y_range = np.array([min(face_out_mid_points[j, 1], test_point[1]), max(face_out_mid_points[j, 1], test_point[1])]) + np.array([-0.1, 0.1])
            z_range = np.array([min(face_out_mid_points[j, 2], test_point[2]), max(face_out_mid_points[j, 2], test_point[2])] )+ np.array([-0.1, 0.1])

            # 过滤出在范围内的中点
            i_mid1 = (face_points_1[:, 0] > x_range[0]) & (face_points_1[:, 0] < x_range[1]) & \
                     (face_points_1[:, 1] > y_range[0]) & (face_points_1[:, 1] < y_range[1]) & \
                     (face_points_1[:, 2] > z_range[0]) & (face_points_1[:, 2] < z_range[1])

            i_mid2 = (face_points_2[:, 0] > x_range[0]) & (face_points_2[:, 0] < x_range[1]) & \
                     (face_points_2[:, 1] > y_range[0]) & (face_points_2[:, 1] < y_range[1]) & \
                     (face_points_2[:, 2] > z_range[0]) & (face_points_2[:, 2] < z_range[1])

            i_mid3 = (face_points_3[:, 0] > x_range[0]) & (face_points_3[:, 0] < x_range[1]) & \
                     (face_points_3[:, 1] > y_range[0]) & (face_points_3[:, 1] < y_range[1]) & \
                     (face_points_3[:, 2] > z_range[0]) & (face_points_3[:, 2] < z_range[1])

            i_mid = i_mid1 | i_mid2 | i_mid3

            if np.sum(i_mid) == 0:
                result[i] = False
                c1 += 1
                break
            else:
                points_mid1 = face_points_1[i_mid, :]
                points_mid2 = face_points_2[i_mid, :]
                points_mid3 = face_points_3[i_mid, :]
                num_points_mid = points_mid1.shape[0]

                e1 = points_mid2 - points_mid1
                e2 = points_mid3 - points_mid1
                s = test_point - points_mid1
                d = np.tile((face_out_mid_points[j, :] - test_point) / np.linalg.norm(face_out_mid_points[j, :] - test_point),
                            (num_points_mid, 1))

                s1 = np.cross(d, e2)
                s2 = np.cross(s, e1)

                t = np.sum(s2 * e2, axis=1) / np.sum(e1 * s1, axis=1)
                b1 = np.sum(s1 * s, axis=1) / np.sum(e1 * s1, axis=1)
                b2 = np.sum(s2 * d, axis=1) / np.sum(e1 * s1, axis=1)

                bool_b = (b1 >= 0) & (b1 <= 1) & (b2 >= 0) & (b2 <= 1) & ((b1 + b2) <= 1) & (t > 0)
                t_limit = (t - np.linalg.norm(face_out_mid_points[j, :] - test_point)) < -1e-6
                bool_b = bool_b & t_limit
                sum_bool_b = np.sum(bool_b)

                if sum_bool_b % 2 == 1:
                    c5 += 1
                    continue
                else:
                    result[i] = False
                    c4 += 1
                    break

    return result



def Limit_L(a1, a2, b1, b2, d_min):
    """
    Link interference detection (from MATLAB Limit_L.m)
    a1, a2: Endpoints of link 1 (3x1 each)
    b1, b2: Endpoints of link 2 (3x1 each)
    d_min: Minimum allowed distance between links
    Returns: 1 if interference, 0 if no interference
    """
    Bool_L = 0
    
    a1 = np.array(a1).flatten()
    a2 = np.array(a2).flatten()
    b1 = np.array(b1).flatten()
    b2 = np.array(b2).flatten()
    
    a = a2 - a1  # Direction vector of the first rod
    b = b2 - b1  # Direction vector of the second rod
    n = np.cross(a, b)  # Vector of the common perpendicular line
    
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # If cross product is zero, lines are parallel
        d = np.linalg.norm(b1 - a1)
    else:
        # Minimum distance between two lines: |n · (b1-a1)| / ||n||
        d = abs(np.dot(n, (b1 - a1))) / n_norm

    if d > d_min:
        Bool_L = 0
    else:
        # Build matrix for solving t = [a, -b, n] \ (b1-a1)
        A = np.column_stack([a, -b, n])
        
        try:
            t = np.linalg.solve(A, b1 - a1)
        except np.linalg.LinAlgError:
            # If solve fails, use endpoint distance
            d = min(np.linalg.norm(b1 - a1), np.linalg.norm(b2 - a2),
                    np.linalg.norm(b2 - a1), np.linalg.norm(b1 - a2))
            Bool_L = 1 if d <= d_min else 0
            return Bool_L
        
        # Check if both intersections fall within segments
        if np.all(t[:2] >= 0) and np.all(t[:2] <= 1):
            Bool_L = 1
        # Check if both intersections are outside segments
        elif np.all((t[:2] < 0) | (t[:2] > 1)):
            d = min(np.linalg.norm(b1 - a1), np.linalg.norm(b2 - a2),
                    np.linalg.norm(b2 - a1), np.linalg.norm(b1 - a2))
            Bool_L = 1 if d <= d_min else 0
        else:
            # One inside, one outside - use point-to-line distance
            if t[0] > 1:
                d = d_Point_Line(a2, b1, b2)
            elif t[0] < 0:
                d = d_Point_Line(a1, b1, b2)
            elif t[1] > 1:
                d = d_Point_Line(b2, a1, a2)
            elif t[1] < 0:
                d = d_Point_Line(b1, a1, a2)
            else:
                d = d_min + 1  # Default to no interference
            
            Bool_L = 1 if d <= d_min else 0
    
    return Bool_L


def d_Point_Line(p1, p2, p3):
    return np.linalg.norm(np.cross((p1 - p3), (p2 - p3))) / np.linalg.norm(p2 - p3)

def ws_facePoints_Ver4(a, c, l_len, e, switch_pm, init_height, dz, guide_len, dl_min, delta_rho, rot_vec):
    """
    Workspace boundary detection using spiral search method.
    Simplified version without U/S limits.
    """
    try:
        Points_set = np.empty((0, 3))

        rot_vecs = rot_vec
        size_rot_vecs = rot_vecs.shape[0]

        delta_angle = 0.15
        point = np.array([[0, 0, 0, 0, 0, 0]])

        count_framework = 0
        count_singl = 0
        count_l = 0
        count_q = 0

        z_values = np.arange(init_height, guide_len + init_height, dz)
        for z in z_values:
            rho = 0
            Rotate_angle = 0
            bool_inside = -1
            while Rotate_angle <= np.pi * 2:
                bool_inside_ = bool_inside
                point_xyz = [rho * np.cos(Rotate_angle), rho * np.sin(Rotate_angle), z]

                for i_rot_vec in range(size_rot_vecs):
                    point = np.array([*point_xyz, *rot_vecs[i_rot_vec, :]])
                    q, a0, b0, _, _, singularity = p2q(point, a, c, l_len, e, switch_pm, 1)
                    
                    if singularity == -1:
                        bool_inside = 2
                        count_singl += 1
                        break

                    if np.sum(q >= -1e-8) < 6 or np.sum(q <= guide_len) < 6:
                        bool_inside = 2
                        count_q += 1
                        break

                    # Link interference condition (all 6 pairs)
                    if (Limit_L(b0[:, 0], a0[:, 0], b0[:, 1], a0[:, 1], dl_min) or
                        Limit_L(b0[:, 1], a0[:, 1], b0[:, 2], a0[:, 2], dl_min) or
                        Limit_L(b0[:, 2], a0[:, 2], b0[:, 3], a0[:, 3], dl_min) or
                        Limit_L(b0[:, 3], a0[:, 3], b0[:, 4], a0[:, 4], dl_min) or
                        Limit_L(b0[:, 4], a0[:, 4], b0[:, 5], a0[:, 5], dl_min) or
                        Limit_L(b0[:, 5], a0[:, 5], b0[:, 0], a0[:, 0], dl_min)):
                        bool_inside = 2
                        count_l += 1
                        break
                    
                    bool_inside = 1
                    point_ = point

                if bool_inside == 2 and bool_inside_ == 1:
                    Points_set = np.vstack([Points_set, point_[:3].reshape(1, 3)])
                    Rotate_angle += delta_angle
                    bool_inside = -1
                elif bool_inside == 1 and bool_inside_ == 2:
                    Points_set = np.vstack([Points_set, point_[:3].reshape(1, 3)])
                    Rotate_angle += delta_angle
                    bool_inside = -1
                elif bool_inside == 1:
                    rho += delta_rho
                elif bool_inside == 2:
                    rho -= delta_rho
                    if rho < 0:
                        rho = 0
                        Rotate_angle += delta_angle

    except:
        print("points wrong")
        print(e)
        print(sys.exc_info())
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        print('\n', '>>>' * 20)

    if len(Points_set) == 0:
        Points_set = []
    Points_set = np.array(Points_set)
    limit = {'framework': count_framework, 'l': count_l, 'q': count_q, 'singl': count_singl}
    return Points_set, limit


def Limit_U(l):
    """
    U-joint rotation angle limit check (from MATLAB Limit_U.m)
    l: Link vectors, 3x6 matrix
    Returns: 1 if limit exceeded (interference), 0 if within limits
    """
    gamma = np.array([-np.pi/3, np.pi/3, np.pi])  # Slider rotation angles around base坐标系
    
    Bool_U = 0
    # Check each U-joint: 1,2 with gamma(1); 3 with gamma(2); 4,5 with gamma(2); 6 with gamma(1)
    Bool_U = U_bool(gamma[0], l[:, 0], Bool_U)
    if Bool_U == 0:
        Bool_U = U_bool(gamma[1], l[:, 1], Bool_U)
        if Bool_U == 0:
            Bool_U = U_bool(gamma[1], l[:, 2], Bool_U)
            if Bool_U == 0:
                Bool_U = U_bool(gamma[2], l[:, 3], Bool_U)
                if Bool_U == 0:
                    Bool_U = U_bool(gamma[2], l[:, 4], Bool_U)
                    if Bool_U == 0:
                        Bool_U = U_bool(gamma[0], l[:, 5], Bool_U)
    return Bool_U


def U_bool(gamma, l_i, Bool_U):
    """
    Sub-function for U-joint rotation limit
    gamma: rotation angle
    l_i: link vector (3x1)
    Bool_U: current boolean value
    """
    # Rotation matrix around z-axis
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    y = np.array([0, 1, 0])
    vy = Rz @ y
    
    # Calculate angles
    l_norm = np.linalg.norm(l_i)
    if l_norm < 1e-10:
        return 1
    
    alpha = abs(np.arccos(np.dot(l_i, vy) / l_norm) - np.pi/2)
    
    # beta calculation using cross product
    cross_l_vy = np.cross(l_i, vy)
    cross_norm = np.linalg.norm(cross_l_vy)
    if cross_norm < 1e-10:
        beta = np.pi/2
    else:
        beta = np.pi/2 - np.arccos(np.dot(np.array([0, 0, 1]), cross_l_vy) / cross_norm)
    
    # Check limits based on beta value
    if beta >= -0.4363 and beta <= 0:
        if not (abs(alpha) <= (-0.4 * beta + 0.1734)):
            Bool_U = 1
    elif beta >= (-np.pi/2) and beta < -0.4363:
        if not (abs(alpha) <= 0.3449):
            Bool_U = 1
    else:
        Bool_U = 1
    
    return Bool_U


def Limit_S(l, n_tool):
    """
    S-joint (spherical joint) rotation angle limit check (from MATLAB Limit_S.m)
    l: Link vectors, 3x6 matrix
    n_tool: Tool platform normal vector (3x1)
    Returns: 1 if limit exceeded (interference), 0 if within limits
    """
    p = 62.8490  # p = L * cos(theta), L is link length, theta is angle between link and cross shaft
    
    Bool_S = 0
    # Check pairs: (1,2), (3,4), (5,6)
    Bool_S = S_bool(0, p, l[:, 0], l[:, 1], n_tool) + S_bool(1, p, l[:, 0], l[:, 1], n_tool)
    if Bool_S == 0:
        Bool_S = S_bool(0, p, l[:, 2], l[:, 3], n_tool) + S_bool(1, p, l[:, 2], l[:, 3], n_tool)
        if Bool_S == 0:
            Bool_S = S_bool(0, p, l[:, 4], l[:, 5], n_tool) + S_bool(1, p, l[:, 4], l[:, 5], n_tool)
    
    return 1 if Bool_S > 0 else 0


def S_bool(v_bool, p, s, m, n_tool):
    """
    Sub-function for S-joint rotation limit
    v_bool: 0 or 1, selects which solution for v
    p: L * cos(theta)
    s: link 1 vector (3x1)
    m: link 2 vector (3x1)
    n_tool: tool normal vector (3x1)
    """
    Bool_S = 0
    
    # Prevent division by zero
    deno = s[0] * m[1] - m[0] * s[1]
    if s[0] == 0:
        s[0] = 0.000001
    if m[1] == 0:
        m[1] = 0.000001
    if deno == 0:
        deno = 0.000001
    
    # Calculate parameters
    C = -p * (s[0] + m[0]) / deno
    D = (m[0] * s[2] - m[2] * s[0]) / deno
    A = (p - C * s[1]) / s[0]
    B = -(D * s[1] + s[2]) / s[0]
    
    E = B**2 + D**2 + 1
    F = 2 * (A * B + C * D)
    G = A**2 + C**2 - 1
    
    # Calculate v based on v_bool
    if v_bool == 0:
        v3 = (-F + np.sqrt(max(F**2 - 4*E*G, 0))) / (2 * E)
    else:
        v3 = (-F - np.sqrt(max(F**2 - 4*E*G, 0))) / (2 * E)
    
    v1 = A + B * v3
    v2 = C + D * v3
    v = np.array([v1, v2, v3])
    
    # Calculate beta
    n_tool_norm = np.linalg.norm(n_tool)
    v_norm = np.linalg.norm(v)
    if n_tool_norm < 1e-10 or v_norm < 1e-10:
        beta = 0
    else:
        beta = np.arccos(np.dot(n_tool, v) / (n_tool_norm * v_norm)) - np.pi/2
    
    # Calculate alpha angles
    nxv = np.cross(n_tool, v)
    nxv_norm = np.linalg.norm(nxv)
    
    cross_s_v = np.cross(s, v)
    cross_s_v_norm = np.linalg.norm(cross_s_v)
    cross_m_v = np.cross(m, v)
    cross_m_v_norm = np.linalg.norm(cross_m_v)
    
    if nxv_norm < 1e-10 or cross_s_v_norm < 1e-10:
        alpha1 = 0
    else:
        alpha1 = abs(np.arccos(np.dot(nxv, cross_s_v) / (nxv_norm * cross_s_v_norm)))
    
    if nxv_norm < 1e-10 or cross_m_v_norm < 1e-10:
        alpha2 = 0
    else:
        alpha2 = abs(np.arccos(np.dot(nxv, cross_m_v) / (nxv_norm * cross_m_v_norm)))
    
    # Check limits based on beta value
    if beta < -0.66183 or beta > 0.66183:
        Bool_S = 1
    else:
        if beta > -0.26756 and beta < -0.23003:
            alpha1_max = -10073 * beta**3 - 7364.2 * beta**2 - 1797.4 * beta - 143.53
            if alpha1 >= alpha1_max:
                Bool_S = 1
        elif beta >= -0.23003 and beta < 0.66183:
            alpha1_max = -5.0588 * beta**3 + 1.862 * beta**2 - 0.6155 * beta + 2.0261
            if alpha1 >= alpha1_max:
                Bool_S = 1
        
        if Bool_S == 0:
            if beta > -0.66183 and beta <= 0.23003:
                alpha2_max = 5.0588 * beta**3 + 1.862 * beta**2 + 0.6155 * beta + 2.0261
                if alpha2 >= alpha2_max:
                    Bool_S = 1
            elif beta > 0.23003 and beta < 0.26756:
                alpha2_max = 10073 * beta**3 - 7364.2 * beta**2 + 1797.4 * beta - 143.53
                if alpha2 >= alpha2_max:
                    Bool_S = 1
    
    return Bool_S


def ws_facePoints_PABD(a, c, l_len, e, switch_pm, init_height, dz, guide_len, dl_min, edge_k_init, rot_vec):
    """
    PABD (Polar Adaptive Boundary Detection) workspace boundary search method.
    Simplified version without U/S joint limits.
    
    Parameters:
    - a, c: Robot platform parameters (3x6 matrices)
    - l_len: Link lengths (1x6)
    - e: Screw directions (3x6)
    - switch_pm: Inverse solution sign array (1x6)
    - init_height: Initial height (mm)
    - dz: Z-axis step size (mm)
    - guide_len: Guide rail length (mm)
    - dl_min: Minimum distance for link interference detection (mm)
    - edge_k_init: Initial grid spacing for PABD (mm)
    - rot_vec: Orientation vectors to test (Nx3)
    
    Returns:
    - Points_set: Boundary points (Nx3)
    - limit: Dictionary with counts of various rejection reasons
    """
    try:
        Points_set = np.empty((0, 3))
        
        rot_vecs = rot_vec
        size_rot_vecs = rot_vecs.shape[0]
        
        count_framework = 0
        count_singl = 0
        count_l = 0
        count_q = 0
        
        # Priority matrix (same as MATLAB valMat0)
        valMat0 = np.array([[0.49, 0.49, 0.05, 0.5, 0.53],
                            [0.54, 0.51, 0.1, 0.52, 0.58],
                            [0.55, 0.57, 0.0, 0.6, 0.65],
                            [0.72, 0.7, 0.71, 0.8, 0.88],
                            [0.74, 0.73, 0.75, 0.9, 1]])
        
        z_values = np.arange(init_height, guide_len + init_height, dz)
        
        for z in z_values:
            P_center = None
            Edge_k = edge_k_init
            Edge_R = np.eye(2)
            Rotate_angle = 0
            count_pabd = 0
            points_set_z_start = Points_set.shape[0]  # snapshot to detect empty result at this z

            # Sweep +x, keep last reachable point as the initial boundary seed.
            # (Break on first non-reachable rho; use the PREVIOUS rho as P_center.)
            delta_rho_init = 0.5
            rho_max = guide_len + init_height
            prev_reachable_point = None
            rho = 0.0
            hit_non_reachable = False
            while rho <= rho_max:
                point_xyz = [rho, 0, z]
                reachable = True

                for i_rot_vec in range(size_rot_vecs):
                    point = np.array([*point_xyz, *rot_vecs[i_rot_vec, :]])
                    q, a0, b0, _, _, singularity = p2q(point, a, c, l_len, e, switch_pm, 1)

                    if singularity == -1:
                        reachable = False
                        count_singl += 1
                        break

                    if np.sum(q >= -1e-8) < 6 or np.sum(q <= guide_len) < 6:
                        reachable = False
                        count_q += 1
                        break

                    if (Limit_L(b0[:, 0], a0[:, 0], b0[:, 1], a0[:, 1], dl_min) or
                        Limit_L(b0[:, 1], a0[:, 1], b0[:, 2], a0[:, 2], dl_min) or
                        Limit_L(b0[:, 2], a0[:, 2], b0[:, 3], a0[:, 3], dl_min) or
                        Limit_L(b0[:, 3], a0[:, 3], b0[:, 4], a0[:, 4], dl_min) or
                        Limit_L(b0[:, 4], a0[:, 4], b0[:, 5], a0[:, 5], dl_min) or
                        Limit_L(b0[:, 5], a0[:, 5], b0[:, 0], a0[:, 0], dl_min)):
                        reachable = False
                        count_l += 1
                        break

                if not reachable:
                    hit_non_reachable = True
                    break

                prev_reachable_point = np.array([rho, 0, z, 0, 0, 0])
                rho += delta_rho_init

            # rho=0 itself was non-reachable: no valid seed for this z
            if prev_reachable_point is None:
                continue

            P_center = prev_reachable_point.copy()

            # If the last-reachable point is still at the origin AND the sweep did
            # hit a non-reachable rho right after, the workspace at this z is just
            # the origin neighborhood — record it and move on (matches MATLAB).
            if hit_non_reachable and np.linalg.norm(P_center[:2]) <= 1e-6:
                Points_set = np.vstack([Points_set, P_center[:3].reshape(1, 3)])
                continue
            
            # Main boundary tracking loop
            while True:
                count_pabd += 1
                
                if count_pabd >= 1000:
                    Edge_k = Edge_k * 2
                    count_pabd = 0
                
                if abs(Rotate_angle) >= 2 * np.pi:
                    break
                
                # 5x5 grid (extended to 7x7 for boundary access)
                Point_BOOL = np.ones((7, 7), dtype=int)
                Edge_BOOL = np.zeros((5, 5), dtype=int)
                Edge_Point = {}
                
                # Evaluate all 25 grid points
                for ii in range(1, 6):
                    for jj in range(1, 6):
                        # Calculate offset in world coordinates
                        offset = Edge_k * Edge_R @ np.array([ii - 3, jj - 3])
                        point = np.array([P_center[0] + offset[0], 
                                         P_center[1] + offset[1], 
                                         P_center[2], 0, 0, 0])
                        Edge_Point[(ii-1, jj-1)] = point.copy()
                        
                        # Check reachability
                        reachable = True
                        
                        for i_rot_vec in range(size_rot_vecs):
                            full_point = np.array([point[0], point[1], point[2], 
                                                  rot_vecs[i_rot_vec, 0], 
                                                  rot_vecs[i_rot_vec, 1], 
                                                  rot_vecs[i_rot_vec, 2]])
                            q, a0, b0, _, _, singularity = p2q(full_point, a, c, l_len, e, switch_pm, 1)
                            
                            if singularity == -1:
                                reachable = False
                                count_singl += 1
                                break
                            
                            if np.sum(q >= -1e-8) < 6 or np.sum(q <= guide_len) < 6:
                                reachable = False
                                count_q += 1
                                break
                            
                            # Link interference check (all 6 pairs)
                            if (Limit_L(b0[:, 0], a0[:, 0], b0[:, 1], a0[:, 1], dl_min) or
                                Limit_L(b0[:, 1], a0[:, 1], b0[:, 2], a0[:, 2], dl_min) or
                                Limit_L(b0[:, 2], a0[:, 2], b0[:, 3], a0[:, 3], dl_min) or
                                Limit_L(b0[:, 3], a0[:, 3], b0[:, 4], a0[:, 4], dl_min) or
                                Limit_L(b0[:, 4], a0[:, 4], b0[:, 5], a0[:, 5], dl_min) or
                                Limit_L(b0[:, 5], a0[:, 5], b0[:, 0], a0[:, 0], dl_min)):
                                reachable = False
                                count_l += 1
                                break
                        
                        # MATLAB: 1 = reachable, 0 = not reachable
                        if reachable:
                            Point_BOOL[ii, jj] = 1
                        else:
                            Point_BOOL[ii, jj] = 0
                
                # Find edge points (same logic as MATLAB)
                # Edge point: current is reachable (1), neighbors have both reachable and unreachable
                for ii in range(1, 6):
                    for jj in range(1, 6):
                        if Point_BOOL[ii, jj] == 1:
                            # Check 4 neighbors: up, left, right, down
                            n_up = Point_BOOL[ii-1, jj]
                            n_left = Point_BOOL[ii, jj-1]
                            n_right = Point_BOOL[ii, jj+1]
                            n_down = Point_BOOL[ii+1, jj]
                            
                            # MATLAB: neighbors_sum ~= 0 (at least one reachable) AND neighbors_prod == 0 (at least one unreachable)
                            neighbors_sum = n_up + n_left + n_right + n_down
                            neighbors_prod = n_up * n_left * n_right * n_down
                            
                            if neighbors_sum > 0 and neighbors_prod == 0:
                                Edge_BOOL[ii-1, jj-1] = 1
                                edge_pt = Edge_Point[(ii-1, jj-1)]
                                Points_set = np.vstack([Points_set, np.array([edge_pt[0], edge_pt[1], z])])
                
                # Select next center point
                sub = np.where(Edge_BOOL == 1)
                if len(sub[0]) == 0:
                    Edge_k = Edge_k / 2
                    if Edge_k < 0.01:
                        break
                else:
                    valMat = np.zeros((5, 5))
                    valMat[sub[0], sub[1]] = valMat0[sub[0], sub[1]]
                    linear_idx = np.argmax(valMat)
                    row, col = np.unravel_index(linear_idx, (5, 5))
                    
                    # Update center
                    if (row, col) in Edge_Point:
                        P_center = Edge_Point[(row, col)].copy()
                    
                    # Update rotation matrix
                    theta_temp = np.arctan2(col - 3, row - 3)
                    R_theta = np.array([[np.cos(theta_temp), -np.sin(theta_temp)],
                                       [np.sin(theta_temp), np.cos(theta_temp)]])
                    Edge_R = R_theta @ Edge_R
                    
                    # Update cumulative rotation angle
                    if (2, 2) in Edge_Point and (row, col) in Edge_Point:
                        pt_center = Edge_Point[(2, 2)]
                        pt_next = Edge_Point[(row, col)]
                        if pt_center[0] != 0 or pt_center[1] != 0:
                            relative_angle = np.arctan2(pt_next[1], pt_next[0]) - np.arctan2(pt_center[1], pt_center[0])
                            relative_angle = ((relative_angle + np.pi) % (2 * np.pi)) - np.pi
                            Rotate_angle = Rotate_angle + relative_angle
                
                # If no boundary point was ever appended at this z, stop this z (matches MATLAB)
                if Points_set.shape[0] == points_set_z_start:
                    break

                # Safety check
                if count_pabd > 5000:
                    print("    (PABD reached max iterations, stopping early)")
                    break

    except Exception as ex:
        print("points wrong:", str(ex))
        print(sys.exc_info())
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        print('\n', '>>>' * 20)

    if len(Points_set) == 0:
        Points_set = np.array([])
    else:
        Points_set = np.array(Points_set)
    
    limit = {'framework': count_framework, 'l': count_l, 'q': count_q, 'singl': count_singl}
    return Points_set, limit


def chamfer_distance(cloud1, cloud2):
    """
    计算 Chamfer 距离
    输入：
    cloud1: 参考点云 (m x 3)，原始的点云1
    cloud2: 验证点云 (n x 3)，原始的点云2

    返回：
    chamfer_dist: 归一化的 Chamfer 距离
    """
    # 获取点云的大小
    m = cloud1.shape[0]  # cloud1 的点的数量
    n = cloud2.shape[0]  # cloud2 的点的数量

    # 初始化 Chamfer 距离
    chamfer_dist = 0

    # 计算从 cloud1 到 cloud2 的距离
    dist1 = np.zeros(m)
    for i in range(m):
        # 计算 cloud1 中每个点到 cloud2 中的所有点的距离
        dists = np.linalg.norm(cloud2 - cloud1[i, :], axis=1)
        # 找到 cloud2 中距离 cloud1(i, :) 最近的点
        dist1[i] = np.min(dists)

    # 计算从 cloud2 到 cloud1 的距离
    dist2 = np.zeros(n)
    for i in range(n):
        # 计算 cloud2 中每个点到 cloud1 中的所有点的距离
        dists = np.linalg.norm(cloud1 - cloud2[i, :], axis=1)
        # 找到 cloud1 中距离 cloud2(i, :) 最近的点
        dist2[i] = np.min(dists)

    # 计算归一化 Chamfer 距离：两部分距离之和，归一化到每个点云的数量
    chamfer_dist = (1 / m) * np.sum(dist1**2) + (1 / n) * np.sum(dist2**2)

    return chamfer_dist

def extendPoints(points, num, mode):
    size_points = len(points)
    # print(f"extendPoints::size_points: {size_points}")

    tp_x = [np.min(points[:, 0]), np.max(points[:, 0])]
    tp_y = [np.min(points[:, 1]), np.max(points[:, 1])]
    tp_z = [np.min(points[:, 2]), np.max(points[:, 2])]

    x = (tp_x[1] - tp_x[0]) * np.random.rand(num) + tp_x[0]  # x coordinates
    y = (tp_y[1] - tp_y[0]) * np.random.rand(num) + tp_y[0]  # y coordinates
    z = (tp_z[1] - tp_z[0]) * np.random.rand(num) + tp_z[0]  # z coordinates
    result = np.vstack([x, y, z]).T
    # Calculate the convex hull for the input points
    if mode == 1:
        alpha_shape = alphashape.alphashape(points, 3)
        if alpha_shape.vertices.shape[0] < 10:
            print("extendPoints::err: length(hull.vertices) < 10")
            return []
        index = np.array(alpha_shape.faces)
        boundary_points = np.array(alpha_shape.vertices)
    if mode == 2:
        hull = ConvexHull(points)
        if hull.vertices.shape[0] < 10:
            print("extendPoints::err: length(hull.vertices) < 10")
            return []
        index = np.array(hull.simplices)
        boundary_points = np.array(hull.vertices)
    is_inside = isPointInsideBoundary_simple(points, index, result, boundary_points, mode)
    result = result[is_inside == 1]

    # print(f"extendPoints::size_result: {len(result)}")
    return result

def p2q_dynamics(P, a, c, l_len, e, r_cmp_p, switch_pm, Mode):
    q = []
    a0 = []
    b0 = []
    R = []
    G = []
    singularity = 1

    if Mode == 1:
        alpha, beta, gamma = P[3], P[4], P[5]
        R = np.array([
            [np.cos(beta) * np.cos(gamma), -np.cos(beta) * np.sin(gamma), np.sin(beta)],
            [np.sin(alpha) * np.sin(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
             -np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
             -np.sin(alpha) * np.cos(beta)],
            [-np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
             np.cos(alpha) * np.sin(beta) * np.sin(gamma) + np.sin(alpha) * np.cos(gamma),
             np.cos(alpha) * np.cos(beta)]
        ])
    elif Mode == 2:
        phi, theta, psi = P[3], P[4], P[5]
        R = np.array([
            [np.cos(phi) * np.cos(theta) * np.cos(psi - phi) - np.sin(phi) * np.sin(psi - phi),
             -np.cos(phi) * np.cos(theta) * np.sin(psi - phi) - np.sin(phi) * np.cos(psi - phi),
             np.cos(phi) * np.sin(theta)],
            [np.sin(phi) * np.cos(theta) * np.cos(psi - phi) + np.cos(phi) * np.sin(psi - phi),
             -np.sin(phi) * np.cos(theta) * np.sin(psi - phi) + np.cos(phi) * np.cos(psi - phi),
             np.sin(phi) * np.sin(theta)],
            [-np.sin(theta) * np.cos(psi - phi), np.sin(theta) * np.sin(psi - phi), np.cos(theta)]
        ])

    tool_loca = np.reshape(P[:3], (3, 1))
    a00 = R @ a
    a01 = R @ (a - np.broadcast_to(r_cmp_p.reshape((3,1)), a.shape))
    a0 = np.broadcast_to(tool_loca, a00.shape) + a00
    H = a0 - c

    for i in range(6):
        temp_0 = (e[:, i].T @ H[:, i]) ** 2 - (H[:, i].T @ H[:, i]) + l_len[0][i] ** 2
        if temp_0 > 0:
            q.append(e[:, i].T @ H[:, i] + switch_pm[i] * np.sqrt(temp_0))
        else:
            singularity = -1
            return [],[],[],[],singularity#报错，进行下一轮

    b0 = c + np.array(q) * e
    l_vec = a0 - b0

    temp_1 = np.sum(l_vec * e, axis=0)
    temp_2 = l_vec / temp_1
    cross_results0 = np.concatenate((
        np.cross(a01[:, 0], temp_2[:, 0]).reshape((3, 1)),
        np.cross(a01[:, 1], temp_2[:, 1]).reshape((3, 1)),
        np.cross(a01[:, 2], temp_2[:, 2]).reshape((3, 1)),
        np.cross(a01[:, 3], temp_2[:, 3]).reshape((3, 1)),
        np.cross(a01[:, 4], temp_2[:, 4]).reshape((3, 1)),
        np.cross(a01[:, 5], temp_2[:, 5]).reshape((3, 1))), axis=1)
    cross_results1 = np.concatenate((
        np.cross(a00[:, 0], temp_2[:, 0]).reshape((3, 1)),
        np.cross(a00[:, 1], temp_2[:, 1]).reshape((3, 1)),
        np.cross(a00[:, 2], temp_2[:, 2]).reshape((3, 1)),
        np.cross(a00[:, 3], temp_2[:, 3]).reshape((3, 1)),
        np.cross(a00[:, 4], temp_2[:, 4]).reshape((3, 1)),
        np.cross(a00[:, 5], temp_2[:, 5]).reshape((3, 1))), axis=1)
    G = np.concatenate((temp_2, cross_results0), axis=0)
    G_for_FL =  np.concatenate((temp_2, cross_results1), axis=0)
    q = np.array(q).reshape(1, 6)
    return q, l_vec, R, G, G_for_FL

def p2q(P, a, c, l_len, e, switch_pm, Mode):
    q = []
    a0 = []
    b0 = []
    R = []
    G = []
    singularity = 1

    if Mode == 1:
        alpha, beta, gamma = P[3], P[4], P[5]
        R = np.array([
            [np.cos(beta) * np.cos(gamma), -np.cos(beta) * np.sin(gamma), np.sin(beta)],
            [np.sin(alpha) * np.sin(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
             -np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
             -np.sin(alpha) * np.cos(beta)],
            [-np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
             np.cos(alpha) * np.sin(beta) * np.sin(gamma) + np.sin(alpha) * np.cos(gamma),
             np.cos(alpha) * np.cos(beta)]
        ])
    elif Mode == 2:
        phi, theta, psi = P[3], P[4], P[5]
        R = np.array([
            [np.cos(phi) * np.cos(theta) * np.cos(psi - phi) - np.sin(phi) * np.sin(psi - phi),
             -np.cos(phi) * np.cos(theta) * np.sin(psi - phi) - np.sin(phi) * np.cos(psi - phi),
             np.cos(phi) * np.sin(theta)],
            [np.sin(phi) * np.cos(theta) * np.cos(psi - phi) + np.cos(phi) * np.sin(psi - phi),
             -np.sin(phi) * np.cos(theta) * np.sin(psi - phi) + np.cos(phi) * np.cos(psi - phi),
             np.sin(phi) * np.sin(theta)],
            [-np.sin(theta) * np.cos(psi - phi), np.sin(theta) * np.sin(psi - phi), np.cos(theta)]
        ])

    tool_loca = np.reshape(P[:3], (3, 1))
    a00 = R @ a
    a0 = np.broadcast_to(tool_loca, a00.shape) + a00
    H = a0 - c

    for i in range(6):
        temp_0 = (e[:, i].T @ H[:, i]) ** 2 - (H[:, i].T @ H[:, i]) + l_len[i] ** 2
        if temp_0 > 0:
            q.append(e[:, i].T @ H[:, i] + switch_pm[i] * np.sqrt(temp_0))
        else:
            singularity = -1
            return [], [], [], [], [], singularity

    b0 = c + np.array(q) * e
    l_vec = a0 - b0

    temp_1 = np.sum(l_vec * e, axis=0)
    temp_2 = l_vec / temp_1
    cross_results = np.concatenate((
        np.cross(a00[:, 0], temp_2[:, 0]).reshape((3, 1)),
        np.cross(a00[:, 1], temp_2[:, 1]).reshape((3, 1)),
        np.cross(a00[:, 2], temp_2[:, 2]).reshape((3, 1)),
        np.cross(a00[:, 3], temp_2[:, 3]).reshape((3, 1)),
        np.cross(a00[:, 4], temp_2[:, 4]).reshape((3, 1)),
        np.cross(a00[:, 5], temp_2[:, 5]).reshape((3, 1))), axis=1)
    G = np.concatenate((temp_2, cross_results), axis=0)
    q = np.array(q).reshape(1,6)
    return q, a0, b0, R, G, singularity


def q2p(q, P_, a, c, length_l, e, switch_pm, Mode):
    # 将q转置为列向量，P_保持为列向量
    q = np.transpose(q)  # 转换为列向量（如果 q 是行向量）
    count = 0
    dP_ = 1e12
    P__ = P_
    a0_ = []
    b0_ = []
    R_ = []

    while True:
        count += 1

        # 假设 p2q 是另外一个函数，返回的值应该是与 MATLAB 中一样的形式
        q_, a0, b0, R, G_, singularity = p2q(P_, a, c, length_l, e, switch_pm, Mode)

        if singularity == -1:
            Pos = P__
            break

        q_ = np.transpose(q_)  # 转置 q_
        P = P_ - np.squeeze(np.linalg.inv(G_.T) @ (q_ - q) / 2)

        dP = np.linalg.norm(P - P_)  # 计算欧几里得距离
        # 如果需要根据精度要求调整计算，可以根据需求修改此部分
        if dP < dP_:
            P__ = P
            dP_ = dP
            a0_ = a0
            b0_ = b0
            R_ = R

        # 判断停止条件：精度小于阈值或达到最大迭代次数
        if np.max(dP) < 1e-4 or count > 200 or np.max(np.abs(q_ - q)) < 1e-4:
            Pos = P
            break
        else:
            P_ = P

    return Pos, a0_, b0_, R_, dP, count

def other_performance(points, a, c, l_len, e, switch_pm, init_height, guide_len, dl_min, vel_limit):
    stiffness_weight = np.array([1, 1.2])  # 位置变形刚度和角度变形刚度
    # 速度表现
    size_point = len(points)

    vel_all = np.zeros((size_point, 6))  # n*6
    stiff_all = np.zeros(size_point)  # n

    for i in range(size_point):
        _, _, _, _, G, _ = p2q(points[i, :], a, c, l_len, e, switch_pm, 1)

        # 速度计算方法
        temp_vel = np.sum(np.abs(G.T @ vel_limit.reshape(6,1)), axis=1)
        vel_all[i, :] = temp_vel

        # 刚度矩阵计算
        temp_stiff = calc_stiffness(G, stiffness_weight)
        stiff_all[i] = temp_stiff

    vel_value = np.max(vel_all)
    stiff_value = np.max(stiff_all)

    # 间隙误差
    gap_err_value = calc_pose_error(points[0, :], a, c, l_len, e, switch_pm, 1)

    return vel_value, stiff_value, gap_err_value

def calc_pose_error(p, a, c, l_len, e, switch_pm, Mode):
    temp_q, _, _, R, _, _ = p2q(p, a, c, l_len, e, switch_pm, Mode)
    # step1: 生成新a、c的依据：l_len的百分比
    err_percent = 0.0005
    gap_err = np.max(l_len) * err_percent

    # 用蒙特卡洛代替3^36种可能
    num_samples = 100
    gap_err_value = 0
    for i in range(num_samples):
        # 随机生成a,c的误差
        new_a = a + gap_err * (np.random.randint(1, 4, (3, 6)) - 2)
        new_c = c + gap_err * (np.random.randint(1, 4, (3, 6)) - 2)

        # 计算新的末端位姿
        new_p, _, _, new_R, _, _ = q2p(temp_q, p, new_a, new_c, l_len, e, switch_pm, Mode)

        # 归一化末端位置和姿态误差
        weight = [((np.sqrt(6) - np.sqrt(2)) / 2) / (2 * np.sqrt(3) + (np.sqrt(6) - np.sqrt(2)) / 2),
                  2 * np.sqrt(3) / (2 * np.sqrt(3) + (np.sqrt(6) - np.sqrt(2)) / 2)]
        U, S, _ = np.linalg.svd(new_R - R)
        err_pos = weight[0] * np.linalg.norm(new_p[:3] - p[:3]) + weight[1] * np.max(S) * gap_err

        # 归一化误差
        err = err_pos / gap_err
        if err > gap_err_value:
            gap_err_value = err

    return gap_err_value


def calc_stiffness(G, stiffness_weight):
    # 计算刚度矩阵的最大奇异值后，按权重求和
    C = np.linalg.inv(G @ G.T)
    _, S_F, _ = np.linalg.svd(C[:3, :])
    _, S_T, _ = np.linalg.svd(C[3:6, :])

    stiff_value = stiffness_weight[0] * np.max(S_F) + stiffness_weight[1] * np.max(S_T)
    return stiff_value


def robot_dynamics(param, motion, F_load, M_load):
    # Extract parameters
    g = param['g']
    a = param['a']
    c = param['c']
    e = param['e']
    l_length = param['l_len']
    m_p = param['m_p']
    I_p = param['I_p']
    Ipc = param['Ipc']
    r_cmp_p = param['r_cmp_p']
    init_height = param['init_height']
    m_l = param['m_l']
    I_l = param['I_l']
    m_sl = param['m_sl']
    switch_pm = param['switch_pm']

    size_Points = motion['x'].shape[0]
    F_A = np.empty((6, 0))

    for F_i in range(size_Points):
        P = motion['x'][F_i, :]
        v_p = motion['v'][F_i, :]
        a_p = motion['a'][F_i, :]

        q, l, R, G, G_for_FL = p2q_dynamics(P, a, c, l_length, e, r_cmp_p, switch_pm, 1)

        v_q = G.T @ v_p
        a_q = np.zeros((6, 1))

        r_cmp_o = R @ r_cmp_p
        v_c = v_p[:3] + np.cross(v_p[3:6], r_cmp_o)
        a_c = a_p[:3] + np.cross(a_p[3:6], r_cmp_o) + np.cross(v_p[3:6],np.cross(v_p[3:6],r_cmp_o))

        omega_l = np.zeros((3, 6))
        nu_l = np.zeros((3, 6))
        v_l = np.zeros((3, 6))
        a_l = np.zeros((3, 6))

        for i in range(6):
            omega_l[:, i] = 1 / norm(l[:, i]) ** 2 * np.cross(l[:, i], (
                        v_p[:3] + np.cross(v_p[3:6], (R @ a[:, i]) - v_q[i] * e[:, i])).flatten())
            v_l[:, i] = v_q[i] * e[:, i] + 0.5 * np.cross(omega_l[:, i], l[:, i])

            a_q[i] = G[:, i].T @ a_p + (np.dot(l[:, i].flatten(), (
                        np.cross(v_p[3:6].flatten(), np.cross(v_p[3:6].flatten(), (R @ a[:, i]))) - np.cross(omega_l[:, i], np.cross(omega_l[:, i], l[:, i]),
                                                    )))) / np.dot(l[:, i], e[:, i])

            nu_l[:, i] = 1 / norm(l[:, i]) ** 2 * np.cross(l[:, i], (a_p[:3] + np.cross(a_p[3:6],
                                                                                        (R @ a[:, i]) + np.cross(
                                                                                        v_p[3:6], np.cross(v_p[3:6],
                                                                                        (R @ a[:,i]),),),) - a_q[i] * e[:, i]))
            a_l[:, i] = a_q[i] * e[:, i] + 0.5 * (np.cross(nu_l[:, i], l[:, i]) + np.cross(omega_l[:, i],
                                                                                                   np.cross(
                                                                                                       omega_l[:, i],
                                                                                                       l[:, i])))

        U = np.zeros((6, 1))
        for i in range(6):
            U[i] = np.dot((m_l * a_l[:, i] - m_l * g), e[:, i]) + m_sl * a_q[i] - m_sl * np.dot(
                g, e[:, i])

        W = np.zeros((3, 6))
        I_lbi = np.zeros((3, 3))
        for i in range(6):
            I_lbi = L2B(l[:, i], l_length[0][i], I_l, m_l)
        W[:, i] = 0.5 * (np.cross(l[:, i], m_l * g - m_l * a_l[:, i]) - I_lbi @ nu_l[:, i] - np.cross(
            omega_l[:, i], (I_lbi @ omega_l[:, i])))

        sum1 = np.zeros((3, 1))
        sum2 = np.zeros((3, 1))
        for i in range(6):
            sum1 += (np.cross(e[:, i], W[:, i])/ np.dot(l[:, i], e[:, i])).reshape((3,1))
            # print((R @ (a[:, 0].reshape(3,1) - r_cmp_p.reshape(3,1))))
            # print(np.squeeze((R @ (a[:, 0].reshape(3,1) - r_cmp_p.reshape(3,1)))))
            sum2 += np.cross(np.squeeze((R @ (a[:, 0].reshape(3,1) - r_cmp_p.reshape(3,1)))),
                             (np.cross(e[:, i], W[:, i]) / np.dot(l[:, i], e[:, i]))).reshape((3,1))
        I_po = R @ Ipc @ R.T
        F_L = np.vstack((-F_load.reshape(-1,1) + m_p * a_c.reshape(-1,1) - m_p * g.reshape(-1,1) + sum1,
                         -M_load.reshape(-1,1) + I_po @ a_p[3:6].reshape(-1,1) + np.cross(v_p[3:6], np.squeeze(I_po @ v_p[3:6].reshape(-1,1))).reshape(-1,1) + sum2))

        F_A = np.hstack((F_A, U + inv(G_for_FL) @ F_L))

    result = {'F_A': F_A}
    return result


def L2B(l_i, l_length, I_l, m_l):
    lx_i = np.cross(np.array([1, 0, 0]), (l_i / l_length))
    R_l = np.vstack([lx_i, 1 / l_length * np.cross(l_i, lx_i), l_i / l_length]).T

    I_lbi = R_l @ (I_l + m_l / 4 * np.diag([l_length ** 2, l_length ** 2, 0])) @ R_l.T
    return I_lbi

def vel_limit_gener():
    vel_max_min = np.array([400, 400, 400, np.pi / 6, np.pi / 6, np.pi / 6])
    return vel_max_min

def my_fitnessfcn(x, init_data):
    # 首先将x转成之前代码里的符号，并做一些初始化
    dl_min = init_data['dl_min']
    refpoints = init_data['refPoints']
    delta_rho = init_data['delta_rho']
    dz = init_data['dz']
    guide_len = init_data['guide_len']
    switch_pm = init_data['switch_pm']
    F_load = init_data['F_load']
    M_load = init_data['M_load']
    rot_vec = init_data['rot_vecs']
    vel_limit = init_data['vel_limit']
    
    # Get workspace method: 'PRBD' or 'PABD', default to 'PRBD'
    ws_method = init_data.get('ws_method', 'PRBD')
    
    [a, c, init_height, l_len, e] = generRbtPara(x)
    
    # 工作空间要考虑与框架的干涉情况
    # Select workspace calculation method based on ws_method
    if ws_method == 'PABD':
        edge_k_init = init_data['edge_k_init']
        ws_points, ws_limit = ws_facePoints_PABD(a, c, l_len, e, switch_pm, init_height, dz, guide_len, dl_min, edge_k_init, rot_vec)
    else:
        # Default to PRBD (ws_facePoints_Ver4)
        ws_points, ws_limit = ws_facePoints_Ver4(a, c, l_len, e, switch_pm, init_height, dz, guide_len, dl_min, delta_rho, rot_vec)
    size_ws_points = ws_points.shape[0]
    print(f"  [Workspace] Method: {ws_method}, Points: {size_ws_points}, Limit: {ws_limit}")
    if size_ws_points < 84:   #点云至少两层
        return 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20
    stepsize = int(np.ceil(size_ws_points / 500))
    testPoint = ws_points[::stepsize, :] / 1000
    # 计算两个点云的最大最小值
    p2_x = np.array([np.min(refpoints[:, 0]), np.max(refpoints[:, 0])])
    p2_y = np.array([np.min(refpoints[:, 1]), np.max(refpoints[:, 1])])
    p2_z = np.array([np.min(refpoints[:, 2]), np.max(refpoints[:, 2])])
    tp_x = np.array([np.min(testPoint[:, 0]), np.max(testPoint[:, 0])])
    tp_y = np.array([np.min(testPoint[:, 1]), np.max(testPoint[:, 1])])
    tp_z = np.array([np.min(testPoint[:, 2]), np.max(testPoint[:, 2])])
    tp_temp = (tp_z[1]-tp_z[0])/10+tp_z[0]
    relaMove = np.array([(p2_x[0]+p2_x[1]-tp_x[0]-tp_x[1])/2, (p2_y[0]+p2_y[1]-tp_y[0]-tp_y[1])/2, p2_z[0]-tp_temp])
    refpoints -= relaMove
    testPoints_extended = extendPoints(testPoint, 1000, MODE)
    if np.shape(testPoints_extended)[0] == 0:
        print("disp(ERR::testPoints_extended is emptyddd)")
        return 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20

    if np.shape(ws_points)[0] >= 3:
        # f1 = chamfer_distance(refpoints, testPoints_extended)  # Assuming chamferDistance is defined
        f1, f2 = ws_inside_boundaries(refpoints, testPoints_extended)  # Assuming ws_insideBoundaries is defined
    else:
        # return 1e20, 1e20, 1e20, 1e20, 1e20,1e20, 1e20
        f1,f2 = 1e20, 1e20
    testPoints_otherPerformance = testPoints_extended[::5, :]
    size_testPoints_otherPerformance = testPoints_otherPerformance.shape[0]

    # Fill the missing dimensions of testPoints_otherPerformance
    testPoints_otherPerformance = np.hstack([
        testPoints_otherPerformance,
        np.pi / 6 * np.random.rand(size_testPoints_otherPerformance, 3) - np.pi / 12
    ])

    # Calculate speed, error, and stiffness
    f3, f4, f5 = other_performance(testPoints_otherPerformance, a, c, l_len, e, switch_pm, init_height, guide_len, dl_min, vel_limit)

    motion1 = {'x': testPoints_otherPerformance, 'v': np.zeros((size_testPoints_otherPerformance, 6)),
              'a': np.zeros((size_testPoints_otherPerformance, 6))}

    param = paramInitialize(a/1000, c/1000, e, init_height/1000,switch_pm)

    try:
        F_A_all = np.empty((6, 0))
        for i in range(M_load.shape[1]):
            result1 = robot_dynamics(param, motion1, F_load, M_load[:, i])
            F_A_all = np.hstack((F_A_all, result1['F_A']))
        f6 = np.max(np.abs(F_A_all))
        max_FA = []
        for i in range(6):
            sort_FA = np.sort(np.abs(F_A_all[i, :]))
            size_sort_FA = sort_FA.shape[0]
            max_FA.append(sort_FA[int(round(size_sort_FA*0.9))-1])
        f7 = np.max(max_FA)
    except:
        # 这个是输出错误的具体原因
        print(e)
        print(sys.exc_info())
        print("Input x:", x)

        # 以下两步都是输出错误的具体位置，报错行号位置在第几行
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        print('\n', '>>>' * 20)
        # return 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20
        f6, f7 = 1e20, 1e20
    print(f'f1:{f1},\nf2:{f2},\nf3:{f3},\nf4:{f4}，\nf5:{f5},\nf6:{f6},\nf7:{f7}')
    return f1, f2, f3, f4, f5, f6, f7

    # 主函数，依据不同的 idx 返回对应的初始化数据
def generInitData(refPoints, idx, ws_method='PRBD'):
    if idx == 1:
        return generInitData_simu(refPoints, ws_method=ws_method)
    elif idx == 2:
        return generInitData_assembly(refPoints, ws_method=ws_method)
    elif idx == 3:
        return generInitData_medical(refPoints, ws_method=ws_method)
    else:
        raise ValueError("Invalid idx value. Should be 1, 2, or 3.")

# 驾驶仿真数据初始化
def generInitData_simu(refPoints, ws_method='PRBD'):
    init_data = {}
    init_data['refPoints'] = refPoints
    init_data['dl_min'] = 20  # 判断连杆是否干涉的最小距离
    init_data['delta_rho'] = 10  # 工作空间沿半径探索的步长
    init_data['dz'] = 50  # 工作空间z轴方向的增长步长
    init_data['edge_k_init'] = 50  # 工作空间PABD法点阵间距
    init_data['guide_len'] = 1000  # 导轨长度
    init_data['switch_pm'] = np.array([-1, -1, -1, -1, -1, -1])  # 用来判断反解中是＋还是﹣，1*6，默认值为-1
    init_data['F_load'] = np.array([0, 0, 800])
    init_data['M_load'] = np.array([[0, 0, 0], [120, 0, 0], [0, 120, 0], [0, 0, 20]]).T
    init_data['rot_vecs'] = np.array([
        [0, 0, 0],
        [np.pi / 6, 0, 0],
        [0, np.pi / 6, 0],
        [0, 0, np.pi / 3],
        [-np.pi / 6, 0, 0],
        [0, -np.pi / 6, 0],
        [0, 0, -np.pi / 3]
    ])
    init_data['vel_limit'] = np.array([1.55, 0.77, 0.35, 0.97, 0.32, 0.32])
    init_data['ws_method'] = ws_method  # 'PRBD' or 'PABD'
    return init_data

# 装配机器人数据初始化
def generInitData_assembly(refPoints, ws_method='PRBD'):
    init_data = {}
    init_data['refPoints'] = refPoints
    init_data['dl_min'] = 8
    init_data['delta_rho'] = 8
    init_data['dz'] = 20
    init_data['edge_k_init'] = 20  # 工作空间PABD法点阵间距
    init_data['guide_len'] = 600  # 导轨长度
    init_data['switch_pm'] = np.array([-1, -1, -1, -1, -1, -1])  # 用来判断反解中是＋还是﹣，1*6，默认值为-1
    init_data['F_load'] = np.array([100, 100, 400])
    init_data['M_load'] = np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 10]]).T
    init_data['rot_vecs'] = np.array([
        [0, 0, 0],
        [np.pi / 4, 0, 0],
        [0, np.pi / 4, 0],
        [0, 0, np.pi / 3],
        [-np.pi / 4, 0, 0],
        [0, -np.pi / 4, 0],
        [0, 0, -np.pi / 3]
    ])
    init_data['vel_limit'] = np.array([0.4, 0.4, 0.4, 2*np.pi/3,2*np.pi/3,2*np.pi/3])
    init_data['ws_method'] = ws_method  # 'PRBD' or 'PABD'
    return init_data

# 医疗机器人数据初始化
def generInitData_medical(refPoints, ws_method='PRBD'):
    init_data = {}
    init_data['refPoints'] = refPoints
    init_data['dl_min'] = 0.5
    init_data['delta_rho'] = 0.2
    init_data['dz'] = 2
    init_data['edge_k_init'] = 1  # 工作空间PABD法点阵间距
    init_data['guide_len'] = 150  # 导轨长度
    init_data['switch_pm'] = np.array([-1, -1, -1, -1, -1, -1])  # 用来判断反解中是＋还是﹣，1*6，默认值为-1
    init_data['F_load'] = np.array([2, 2, 10])
    init_data['M_load'] = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.2]]).T
    init_data['rot_vecs'] = np.array([
        [0, 0, 0],
        [0.175, 0, 0],
        [0, 0.175, 0],
        [0, 0, np.pi / 2],
        [-0.175, 0, 0],
        [0, -0.175, 0],
        [0, 0, -np.pi / 2]
    ])
    init_data['vel_limit'] = np.array( [0.02, 0.02, 0.2, 0.175,0.175,np.pi/2])
    init_data['ws_method'] = ws_method  # 'PRBD' or 'PABD'
    return init_data

def generRbtPara(x):
    a = generA(x)
    c = generC(x)

    init_height = x[6]

    l_len = np.sqrt(init_height ** 2 + np.sum((a - c) ** 2, axis=0))

    e = np.tile(np.array([0, 0, 1], dtype=float), (6, 1)).T  # (3, 6) matrix
    e_angle = np.vstack((
        np.arctan2(c[1, :], c[0, :]),
        x[7:13],  # Indices in Python are 0-based, so x(8:13) corresponds to x[7:13]
        np.zeros([1,6])
    ))

    for i_e in range(6):
        R_e = RotMatrix(e_angle[:, i_e], 3)
        e_ie = e[:, i_e].reshape(3,1)
        e[:, i_e] = np.dot(R_e, e_ie).flatten()

    return a, c, init_height, l_len, e

def generA(x):
    r = x[14]  # Radius
    alpha_0 = (x[13] - np.pi / 10) / (2 / 3* np.pi - 2 / 10 * np.pi)  * 2 - 1  # Normalize angle to [-1, 1]
    alpha_1 = (np.sin(alpha_0 * np.pi / 2) + 1) / 2 * (2 / 3 - 1 / 10 - 1 / 10) * np.pi + np.pi / 10
    alpha = alpha_1 / 2  # Half angle
    theta = np.array(
        [-np.pi / 6 + alpha, np.pi / 2 - alpha, np.pi / 2 + alpha, 7 * np.pi / 6 - alpha, 7 * np.pi / 6 + alpha,
         -np.pi / 6 - alpha])

    x_dir = r * np.cos(theta)
    y_dir = r * np.sin(theta)

    a = np.zeros((3, 6))
    a[0, :] = x_dir  # x-direction components
    a[1, :] = y_dir  # y-direction components

    return a

def generC(x):
    r = x[:6]  # Radii from x(1) to x(6)
    theta = x[15:21]  # Angles from x(16) to x(21)
    x_dir = r * np.cos(theta)
    y_dir = r * np.sin(theta)

    c = np.zeros((3, 6))
    c[0, :] = x_dir  # x-direction components
    c[1, :] = y_dir  # y-direction components

    return c


def generRefPoints(idx):
    if idx == 1:
        return generRefPoints_simu()
    elif idx == 2:
        return generRefPoints_assembly()
    elif idx == 3:
        return generRefPoints_medical()


# 1. 驾驶仿真平台
def generRefPoints_simu():
    num_points = 5000  # 生成的点数

    # 生成球体内的均匀随机点
    theta = 2 * np.pi * np.random.rand(num_points)  # 随机生成角度theta
    phi = np.arccos(2 * np.random.rand(num_points) - 1)  # 随机生成角度phi
    r = np.random.rand(num_points)
    # 将球坐标转换为笛卡尔坐标系
    x = 400 * r * np.sin(phi) * np.cos(theta)
    y = 200 * r * np.sin(phi) * np.sin(theta)
    z = 200 * r * np.cos(phi)

    refPoints0 = np.vstack((x, y, z)).T/1000  # 返回一个 num_points x 3 的点云
    refPoints = extendPoints(refPoints0,1000, MODE);
    return refPoints


# 2. 工业装配机器人
def generRefPoints_assembly():
    num_points = 5000  # 生成的点数
    side_length = 300  # 立方体的边长，范围为[-200, 200]

    # 在 [-200, 200] 范围内生成均匀分布的随机点
    x = side_length * np.random.rand(num_points) - side_length / 2  # x 范围 [-200, 200]
    y = side_length * np.random.rand(num_points) - side_length / 2  # y 范围 [-200, 200]
    z = side_length * np.random.rand(num_points) - side_length / 2  # z 范围 [-200, 200]

    refPoints0 = np.vstack((x, y, z)).T/1000  # 返回一个 num_points x 3 的点云
    refPoints = extendPoints(refPoints0,1000, MODE);
    return refPoints


# 3. 医疗机器人
def generRefPoints_medical():
    num_points = 5000  # 生成的点数
    side_length_x = 20  # 立方体的边长，范围为[-10, 10]
    side_length_y = 20  # 立方体的边长，范围为[-10, 10]
    side_length_z = 100  # 立方体的边长，范围为[-100, 100]

    # 在 [-10, 10] 和 [-100, 100] 范围内生成均匀分布的随机点
    x = side_length_x * np.random.rand(num_points) - side_length_x / 2  # x 范围 [-10, 10]
    y = side_length_y * np.random.rand(num_points) - side_length_y / 2  # y 范围 [-10, 10]
    z = side_length_z * np.random.rand(num_points) - side_length_z / 2  # z 范围 [-100, 100]

    refPoints0 = np.vstack((x, y, z)).T/1000  # 返回一个 num_points x 3 的点云
    refPoints = extendPoints(refPoints0,1000,MODE);
    return refPoints

def generGACon(idx):
    if idx == 1:
        return generGACon_simu()
    elif idx == 2:
        return generGACon_assembly()
    elif idx == 3:
        return generGACon_medical()


def generGACon_simu():
    """
    Generates constraints and initial population for the simulation problem.
    """
    epsilon = 0.0175  # Ensure at least 1 degree separation between linkages
    idxA_l = [16, 17, 18, 19, 20, 21]  # Indices for smaller elements in linear inequality
    idxA_u = [17, 18, 19, 20, 21, 16]  # Indices for larger elements in linear inequality
    len_idx = len(idxA_u)

    A = np.zeros((len_idx, 21))
    b = np.zeros(len_idx)

    for i in range(len_idx):
        A[i, idxA_l[i] - 1] = 1
        A[i, idxA_u[i] - 1] = -1
        b[i] = -epsilon

    b[-1] = 2 * np.pi - epsilon  # Last constraint value

    Aeq = np.array([])  # No equality constraints
    beq = np.array([])

    # Define lower and upper bounds
    lb = np.array([600, 600, 600, 600, 600, 600, 500, -np.pi / 18, -np.pi / 18, -np.pi / 18,
                   -np.pi / 18, -np.pi / 18, -np.pi / 18, np.pi / 10, 320, -np.pi / 2 + np.pi / 18,0,0,0,0,0])
    ub = np.array([1200, 1200, 1200, 1200, 1200, 1200, 1500, np.pi / 4, np.pi / 4, np.pi / 4,
                   np.pi / 4, np.pi / 4, np.pi / 4, np.pi * (2 / 3 - 1 / 10), 600, np.pi / 2 - np.pi / 18, 2*np.pi, 2*np.pi, 2*np.pi,2*np.pi,5*np.pi/2])

    # Define initial population (individual solutions)
    initialPopulation = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1500, 0, 0, 0, 0, 0, 0,
                                  np.pi * (1 / 9), 400, 0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3,
                                  5 * np.pi / 3])

    return A, b, Aeq, beq, lb, ub, initialPopulation


def generGACon_assembly():
    """
    Generates constraints and initial population for the assembly problem.
    """
    epsilon = 0.0175  # Ensure at least 1 degree separation between linkages
    idxA_l = [16, 17, 18, 19, 20, 21]  # Indices for smaller elements in linear inequality
    idxA_u = [17, 18, 19, 20, 21, 16]  # Indices for larger elements in linear inequality
    len_idx = len(idxA_u)

    A = np.zeros((len_idx, 21))
    b = np.zeros(len_idx)

    for i in range(len_idx):
        A[i, idxA_l[i] - 1] = 1
        A[i, idxA_u[i] - 1] = -1
        b[i] = -epsilon

    b[-1] = 2 * np.pi - epsilon  # Last constraint value

    Aeq = np.array([])  # No equality constraints
    beq = np.array([])

    # Define lower and upper bounds
    lb = np.array([150,  150,    150,    150,    150,    150,    300,    -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, np.pi/10,          35,    -np.pi/2+np.pi/18, 0,0,0,0,0])
    ub = np.array([400,  400,    400,    400,    400,    400,    600,    np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi*(2/3-1/10),  100,    np.pi/2-np.pi/18, 2*np.pi, 2*np.pi, 2*np.pi,2*np.pi,5*np.pi/2])

    # Define initial population (individual solutions)
    initialPopulation = np.array([300,300,300,300,300,300,500,0,0,0,0,0,0,np.pi*(1/9),80,0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3])

    return A, b, Aeq, beq, lb, ub, initialPopulation


def generGACon_medical():
    """
    Generates constraints and initial population for the medical problem.
    """
    epsilon = 0.0175  # Ensure at least 1 degree separation between linkages
    idxA_l = [16, 17, 18, 19, 20, 21]  # Indices for smaller elements in linear inequality
    idxA_u = [17, 18, 19, 20, 21, 16]  # Indices for larger elements in linear inequality
    len_idx = len(idxA_u)

    A = np.zeros((len_idx, 21))
    b = np.zeros(len_idx)

    for i in range(len_idx):
        A[i, idxA_l[i] - 1] = 1
        A[i, idxA_u[i] - 1] = -1
        b[i] = -epsilon

    b[-1] = 2 * np.pi - epsilon  # Last constraint value

    Aeq = np.array([])  # No equality constraints
    beq = np.array([])
    # Define lower and upper bounds
    lb = np.array([50,  50,    50,    50,    50,    50,    80,   -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, -np.pi/18, np.pi/10,     8,  -np.pi/2+np.pi/18,0,0,0,0,0])
    ub = np.array([80,  80,    80,    80,    80,    80,    150,    np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi/4,   np.pi*(2/3-1/10), 20,   np.pi/2-np.pi/18, 2*np.pi, 2*np.pi, 2*np.pi,2*np.pi,5*np.pi/2])

    # Define initial population (individual solutions)
    initialPopulation = np.array([60,60,60,60,60,60,120,0,0,0,0,0,0,np.pi*(1/9),15,0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3])

    return A, b, Aeq, beq, lb, ub, initialPopulation

if __name__ == '__main__':
    idx = 2
    state_size = 21
    Global_Seed = 42

    np.random.seed(Global_Seed)
    _, _, _, _, _, _, x = generGACon(idx)
    refpoints = generRefPoints(idx)
    
    # ============================================================
    # 选择工作空间计算方法: 'PRBD' 或 'PABD'
    # ============================================================
    # PRBD: 极坐标扫描方法 (原 ws_facePoints_Ver4)
    # PABD: 极坐标自适应边界检测方法 (新增)
    # ============================================================
    ws_method = 'PABD'  # Change to 'PABD' to use PABD method
    
    init_Data = generInitData(refpoints, idx, ws_method=ws_method)
    
    print(f"Using workspace method: {ws_method}")
    print(f"edge_k_init (for PABD): {init_Data['edge_k_init']}")
    print(f"delta_rho (for PRBD): {init_Data['delta_rho']}")
    print("-" * 50)
    
#     x =np.array([1000.0366,1000.0366,1000.0366,1000.0366,1000.0366,1000.0366,1500.061,1.5979942e-05,1.5979942e-05,1.5979942e-05,1.5979942e-05,1.5979942e-05,1.5979942e-05,0.34914896,400.00244,8.5226355e-05,1.0473893,2.0943952,3.1417844,4.1887903,5.2364674
# ])
    #x = np.array([305.166009344783,300.337307567119,300.296405060805,299.809808426451,298.800439743394,308.847919219047,361.217707931598,0.235581414470418,0.204572102002028,0.0994363256224157,0.191203943384271,0.285818619030984,0.216546400978996,1.06110862585271,95.4402190751212,0.355260277502043,1.39048155240878,2.49765963991051,3.37207188179681,4.49959698890156,6.01113202636867])
    
    # start_time = time.time()
    # f1, f2, f3, f4, f5, f6, f7 = my_fitnessfcn(x, init_Data)
    # total_time = time.time() - start_time
    # print(f'f1:{f1},\nf2:{f2},\nf3:{f3},\nf4:{f4}，\nf5:{f5},\nf6:{f6},\nf7:{f7}')
    # print(f'[Time] Total: {total_time:.3f}s')
    
     # 第一次运行（包含初始化开销）
    print("Run 1 (with init overhead):")
    f1, f2, f3, f4, f5, f6, f7 = my_fitnessfcn(x, init_Data)
    
    # 后续运行（实际运行时间）
    print("\nRun 2:")
    start = time.time()
    f1, f2, f3, f4, f5, f6, f7 = my_fitnessfcn(x, init_Data)
    print(f"Time: {time.time() - start:.3f}s")
    
    print("\nRun 3:")
    start = time.time()
    f1, f2, f3, f4, f5, f6, f7 = my_fitnessfcn(x, init_Data)
    print(f"Time: {time.time() - start:.3f}s")
