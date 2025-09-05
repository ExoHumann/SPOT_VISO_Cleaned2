# models/axis.py
import math
import os
import webbrowser
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

import logging
logger = logging.getLogger(__name__)

class Axis:
    
    def __init__(
        self,
        stations: Optional[List[float]] = None,
        x_coords: Optional[List[float]] = None,
        y_coords: Optional[List[float]] = None,
        z_coords: Optional[List[float]] = None,
        units: str = 'm',
    ):
        if any(v is None for v in (stations, x_coords, y_coords, z_coords)):
            raise TypeError("Axis requires stations and x/y/z coords.")
        s = np.asarray(stations, dtype=float)
        x = np.asarray(x_coords, dtype=float)
        y = np.asarray(y_coords, dtype=float)
        z = np.asarray(z_coords, dtype=float)
        if not (len(s) == len(x) == len(y) == len(z)):
            raise ValueError("Array lengths differ.")
        factor = 1.0 if units.lower() == 'mm' else 1000.0
        s_mm = s * factor
        x_mm = x * factor
        y_mm = y * factor
        z_mm = z * factor
        order = np.argsort(s_mm)
        self.stations = s_mm[order]
        self.x_coords = x_mm[order]
        self.y_coords = y_mm[order]
        self.z_coords = z_mm[order]
        self.validate()

    def validate(self):
        """Validate axis data."""
        if len(self.stations) < 2:
            raise ValueError("Axis requires at least 2 points.")

    def __len__(self):
        return len(self.stations)

    def __repr__(self) -> str:
        n = len(self)
        rng = (float(self.stations[0]), float(self.stations[-1])) if n else (None, None)
        return f"Axis(n={n}, stations_mm=[{rng[0]}, {rng[1]}])"
    
    @lru_cache(maxsize=64)
    def frames_for_stations(self, stations_mm_tuple):
        stations_mm = np.asarray(stations_mm_tuple, float)
        s = np.asarray(self.stations, float)
        x = np.asarray(self.x_coords, float)
        y = np.asarray(self.y_coords, float)
        z = np.asarray(self.z_coords, float)
        Ax = np.interp(stations_mm, s, x)
        Ay = np.interp(stations_mm, s, y)
        Az = np.interp(stations_mm, s, z)
        idx = np.searchsorted(s, stations_mm, side='right') - 1
        idx = np.clip(idx, 0, len(s) - 2)
        dx = x[idx + 1] - x[idx]; dy = y[idx + 1] - y[idx]; dz = z[idx + 1] - z[idx]
        L = np.sqrt(dx*dx + dy*dy + dz*dz); L[L == 0.0] = 1.0
        U = np.stack([dx/L, dy/L, dz/L], axis=1)
        A = np.stack([Ax, Ay, Az], axis=1)
        return A, U
    
    # @lru_cache(maxsize=64)
    # def frames_for_stations(self, stations_mm_tuple: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    #     stations_mm = np.asarray(stations_mm_tuple, float)
    #     s = self.stations
    #     x = self.x_coords
    #     y = self.y_coords
    #     z = self.z_coords
    #     Ax = np.interp(stations_mm, s, x)
    #     Ay = np.interp(stations_mm, s, y)
    #     Az = np.interp(stations_mm, s, z)
    #     idx = np.searchsorted(s, stations_mm, side='right') - 1
    #     idx = np.clip(idx, 0, len(s) - 2)
    #     dx = x[idx + 1] - x[idx]
    #     dy = y[idx + 1] - y[idx]
    #     dz = z[idx + 1] - z[idx]
    #     L = np.sqrt(dx**2 + dy**2 + dz**2)
    #     L[L == 0.0] = 1.0
    #     U = np.stack([dx/L, dy/L, dz/L], axis=1)
    #     A = np.stack([Ax, Ay, Az], axis=1)
    #     return A, U
    
    
   
    def _rotate_vecs_about_axis(V, U, theta):
        """Rotate each row of V about corresponding U by angle theta (radians). V,U:(S,3), theta:(S,)"""
        theta = np.asarray(theta, float).reshape(-1, 1)
        c = np.cos(theta); s = np.sin(theta)
        udv = (U * V).sum(axis=1, keepdims=True)
        return V * c + np.cross(U, V) * s + U * udv * (1.0 - c)

  
    def _rotate_180_about_axis(P, A, U):
        R = P - A[:, None, :]
        dot = np.einsum('sni,si->sn', R, U)
        Rnew = 2.0*dot[:, :, None]*U[:, None, :] - R
        return A[:, None, :] + Rnew

    
    def _section_basis_from_tangent(U):
        U = np.asarray(U, float)
        up = np.array([0.0, 0.0, 1.0])
        dot_up = (U * up[None, :]).sum(axis=1)
        near_up = np.abs(dot_up) > 0.9
        ref = np.tile(up, (len(U), 1))
        ref[near_up] = np.array([1.0, 0.0, 0.0])
        Y = np.cross(ref, U)
        Yn = np.linalg.norm(Y, axis=1, keepdims=True); Yn[Yn == 0.0] = 1.0
        Y /= Yn
        Z = np.cross(U, Y)
        Zn = np.linalg.norm(Z, axis=1, keepdims=True); Zn[Zn == 0.0] = 1.0
        Z /= Zn
        mask = Z[:, 2] < 0.0
        if np.any(mask):
            Y[mask] *= -1.0
            Z[mask] *= -1.0
        return Y, Z

    def embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=0.0):
        stations_mm = np.asarray(stations_mm, float)
        X_mm = np.asarray(X_mm, float)
        Y_mm = np.asarray(Y_mm, float)

        # cached frames (lru_cache on the Axis instance)
        A, U = axis.frames_for_stations(tuple(np.round(stations_mm, 6)))

        # basis in the normal plane
        Yb, Zb = Axis._section_basis_from_tangent(U)

        # optional twist
        if np.any(np.asarray(twist_deg) != 0.0):
            theta = np.deg2rad(np.asarray(twist_deg, float)).reshape(len(stations_mm))
            Yb = Axis._rotate_vecs_about_axis(Yb, U, theta)
            Zb = Axis._rotate_vecs_about_axis(Zb, U, theta)

        # assemble + legacy 180° flip
        P = (A[:, None, :] +
             X_mm[:, :, None] * Yb[:, None, :] +
             Y_mm[:, :, None] * Zb[:, None, :])
        P = Axis._rotate_180_about_axis(P, A, U)
        return P

    def get_segment_for_station(self, station):
        """
        Find the segment containing the station and compute interpolation factor.
        
        Args:
            station (float): Station value in mm.
            
        Returns:
            tuple: (i, t) where i is the index of the segment's start point,
                   and t is the interpolation factor (0 <= t <= 1).
                   Returns (None, None) if station is out of range.
        """
        for i in range(len(self.stations) - 1):
            if self.stations[i] <= station <= self.stations[i + 1]:
                t = (station - self.stations[i]) / (self.stations[i + 1] - self.stations[i]) if self.stations[i + 1] != self.stations[i] else 0
                return i, t
        print(f"Warning: Station {station} mm is outside the range [{self.stations[0]}, {self.stations[-1]}] mm")
        return None, None

    def get_position_at_station(self, station):
        """
        Interpolate global position (X, Y, Z) at the given station in mm.
        
        Args:
            station (float): Station value in mm.
            
        Returns:
            tuple: (X, Y, Z) global coordinates in mm, or (0, 0, 0) if out of range.
        """
        i, t = self.get_segment_for_station(station)
        if i is None:
            return 0.0, 0.0, 0.0
        x = self.x_coords[i] + t * (self.x_coords[i + 1] - self.x_coords[i])
        y = self.y_coords[i] + t * (self.y_coords[i + 1] - self.y_coords[i])
        z = self.z_coords[i] + t * (self.z_coords[i + 1] - self.z_coords[i])
        return x, y, z

    def get_plane_basis_at_station(self, station):
        """
        Compute orthonormal basis vectors for the plane perpendicular to the axis segment at the interpolated position.
        
        Args:
            station (float): Station value in mm.
            
        Returns:
            tuple: (u, v, tangent) where u and v are orthonormal vectors in the plane,
                and tangent is the segment's direction vector (unitless).
                Returns (None, None, None) if out of range or invalid segment.
        """
        i, t = self.get_segment_for_station(station)
        if i is None:
            return None, None, None

        pos0 = np.array([self.x_coords[i], self.y_coords[i], self.z_coords[i]])
        pos1 = np.array([self.x_coords[i + 1], self.y_coords[i + 1], self.z_coords[i + 1]])
        position = pos0 + t * (pos1 - pos0)
        tangent = pos1 - pos0
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            print(f"Warning: Zero-length segment at station {station} mm")
            return None, None, None
        tangent = tangent / tangent_norm

        # Use a reference vector perpendicular to the tangent, ensure v is positive Z
        ref_vector = np.array([0, 0, 1]) if abs(np.dot(tangent, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
        u = np.cross(tangent, ref_vector)
        u = u / np.linalg.norm(u)
        v = np.cross(u, tangent)  # Ensure v aligns with positive Z preference
        v = v / np.linalg.norm(v)
        if v[2] < 0:
            v = -v  # Flip v to ensure positive Z component

        return u, v, tangent

    def transform_points(self, list_dicts_eval_points, stations, rotation_angle=0):
        """
        Transform lists of points at given stations to global coordinates in mm, with optional rotation around the longitudinal axis.
        
        Args:
            list_dicts_eval_points (list): List of lists of dictionaries with 'id', 'x', 'y', and possibly other keys,
                                        where 'x' and 'y' are in mm.
            stations (list): List of station values in mm corresponding to each point list.
            rotation_angle (float, optional): Rotation angle in degrees around the longitudinal axis (default: 0).
            
        Returns:
            list: List of lists of dictionaries with all original keys plus 'x_global', 'y_global', 'z_global' keys,
                where 'x_global', 'y_global', 'z_global' are in mm.
        """
        if len(list_dicts_eval_points) != len(stations):
            print(f"Warning: Number of point sets ({len(list_dicts_eval_points)}) does not match number of stations ({len(stations)})")
            return []

        transformed_points = []
        for station, points in zip(stations, list_dicts_eval_points):
            station_points = []
            for point in points:
                try:
                    global_x, global_y, global_z = self.transform_to_global(station, point['x'], point['y'], rotation_angle)
                    # Create a new dictionary that copies all original keys and adds global coordinates
                    transformed_point = dict(point)  # Shallow copy of the original point dictionary
                    transformed_point.update({
                        'x_global': float(global_x),  # Global X coordinate (in mm)
                        'y_global': float(global_y),  # Global Y coordinate (in mm)
                        'z_global': float(global_z)   # Global Z coordinate (in mm)
                    })
                    station_points.append(transformed_point)
                except (KeyError, TypeError) as e:
                    print(f"Warning: Invalid point format at station {station} mm: {point}. Error: {e}")
                    continue
            transformed_points.append(station_points)
        return transformed_points

    def transform_to_global(self, station, local_x, local_y, rotation_angle=0):
        """
        Transform local (x, y) coordinates to global (X, Y, Z) coordinates in mm,
        ensuring the plane is perpendicular to the axis segment at the station, with optional rotation around the longitudinal axis.
        
        Args:
            station (float): Station value in mm.
            local_x (float): Local x coordinate in mm (maps to width direction in the plane).
            local_y (float): Local y coordinate in mm (maps to height direction in the plane).
            rotation_angle (float, optional): Rotation angle in degrees around the longitudinal axis (default: 0).
            
        Returns:
            tuple: (X, Y, Z) global coordinates in mm, or (0, 0, 0) if transformation fails.
        """
        # Get position and plane basis
        origin_x, origin_y, origin_z = self.get_position_at_station(station)
        u, v, tangent = self.get_plane_basis_at_station(station)
        if u is None or v is None:
            return 0.0, 0.0, 0.0

        # Apply rotation to local coordinates
        theta = math.radians(rotation_angle)  # Convert degrees to radians
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        # 2D rotation matrix: [cos θ, -sin θ; sin θ, cos θ]
        x_rotated = local_x * cos_theta - local_y * sin_theta
        y_rotated = local_x * sin_theta + local_y * cos_theta

        # Debug basis and rotation
        print(f"Station {station} mm: tangent={tangent}, u={u}, v={v}, rotation_angle={rotation_angle} deg")
        print(f"Local coords: ({local_x}, {local_y}) -> Rotated local coords: ({x_rotated}, {y_rotated})")

        # Transform rotated local coordinates to global
        global_coords = (
            origin_x + x_rotated * u[0] + y_rotated * v[0],
            origin_y + x_rotated * u[1] + y_rotated * v[1],
            origin_z + x_rotated * u[2] + y_rotated * v[2]
        )
        return global_coords

    # @lru_cache(maxsize=64)
    # def frames_for_stations(self, stations_mm_tuple: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    #     stations_mm = np.asarray(stations_mm_tuple, float)
    #     s = self.stations
    #     x = self.x_coords
    #     y = self.y_coords
    #     z = self.z_coords
    #     Ax = np.interp(stations_mm, s, x)
    #     Ay = np.interp(stations_mm, s, y)
    #     Az = np.interp(stations_mm, s, z)
    #     idx = np.searchsorted(s, stations_mm, side='right') - 1
    #     idx = np.clip(idx, 0, len(s) - 2)
    #     dx = x[idx + 1] - x[idx]
    #     dy = y[idx + 1] - y[idx]
    #     dz = z[idx + 1] - z[idx]
    #     L = np.sqrt(dx**2 + dy**2 + dz**2)
    #     L[L == 0.0] = 1.0
    #     U = np.stack([dx/L, dy/L, dz/L], axis=1)
    #     A = np.stack([Ax, Ay, Az], axis=1)
    #     return A, U

    # def transform_points(self, list_dicts_eval_points, stations, rotation_angle=0):
    #     """Transform points (from original)."""
    #     if len(list_dicts_eval_points) != len(stations):
    #         logger.warning("Point sets and stations mismatch.")
    #         return []
    #     transformed_points = []
    #     for station, points in zip(stations, list_dicts_eval_points):
    #         station_points = []
    #         for point in points:
    #             try:
    #                 global_x, global_y, global_z = self.transform_to_global(station, point['x'], point['y'], rotation_angle)
    #                 transformed_point = dict(point)
    #                 transformed_point.update({
    #                     'x_global': float(global_x),
    #                     'y_global': float(global_y),
    #                     'z_global': float(global_z)
    #                 })
    #                 station_points.append(transformed_point)
    #             except (KeyError, TypeError) as e:
    #                 logger.warning(f"Invalid point at station {station}: {e}")
    #                 continue
    #         transformed_points.append(station_points)
    #     return transformed_points

    # def transform_to_global(self, station, local_x, local_y, rotation_angle=0):
    #     """Transform local to global (from original)."""
    #     origin_x, origin_y, origin_z = self.get_position_at_station(station)
    #     u, v, tangent = self.get_plane_basis_at_station(station)
    #     if u is None or v is None:
    #         return 0.0, 0.0, 0.0
    #     theta = math.radians(rotation_angle)
    #     cos_theta = math.cos(theta)
    #     sin_theta = math.sin(theta)
    #     x_rotated = local_x * cos_theta - local_y * sin_theta
    #     y_rotated = local_x * sin_theta + local_y * cos_theta
    #     global_coords = (
    #         origin_x + x_rotated * u[0] + y_rotated * v[0],
    #         origin_y + x_rotated * u[1] + y_rotated * v[1],
    #         origin_z + x_rotated * u[2] + y_rotated * v[2]
    #     )
    #     return global_coords

    # def get_position_at_station(self, station):
    #     """Get position at station (from original)."""
    #     i, t = self.get_segment_for_station(station)
    #     if i is None:
    #         return 0.0, 0.0, 0.0
    #     x = self.x_coords[i] + t * (self.x_coords[i + 1] - self.x_coords[i])
    #     y = self.y_coords[i] + t * (self.y_coords[i + 1] - self.y_coords[i])
    #     z = self.z_coords[i] + t * (self.z_coords[i + 1] - self.z_coords[i])
    #     return x, y, z

    # def get_plane_basis_at_station(self, station):
    #     """Get basis at station (from original)."""
    #     i, t = self.get_segment_for_station(station)
    #     if i is None:
    #         return None, None, None
    #     pos0 = np.array([self.x_coords[i], self.y_coords[i], self.z_coords[i]])
    #     pos1 = np.array([self.x_coords[i + 1], self.y_coords[i + 1], self.z_coords[i + 1]])
    #     tangent = pos1 - pos0
    #     tangent_norm = np.linalg.norm(tangent)
    #     if tangent_norm == 0:
    #         return None, None, None
    #     tangent = tangent / tangent_norm
    #     ref_vector = np.array([0, 0, 1]) if abs(np.dot(tangent, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
    #     u = np.cross(tangent, ref_vector)
    #     u = u / np.linalg.norm(u)
    #     v = np.cross(u, tangent)
    #     v = v / np.linalg.norm(v)
    #     if v[2] < 0:
    #         v = -v
    #     return u, v, tangent
    
    # @staticmethod
    # def embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=0.0):
    #     stations_mm = np.asarray(stations_mm, float)
    #     X_mm = np.asarray(X_mm, float)
    #     Y_mm = np.asarray(Y_mm, float)

    #     # cached frames (lru_cache on the Axis instance)
    #     A, U = axis.frames_for_stations(tuple(np.round(stations_mm, 6)))

    #     # basis in the normal plane
    #     Yb, Zb = Axis._section_basis_from_tangent(U)

    #     # optional twist
    #     if np.any(np.asarray(twist_deg) != 0.0):
    #         theta = np.deg2rad(np.asarray(twist_deg, float)).reshape(len(stations_mm))
    #         Yb = Axis._rotate_vecs_about_axis(Yb, U, theta)
    #         Zb = Axis._rotate_vecs_about_axis(Zb, U, theta)

    #     # assemble + legacy 180° flip
    #     P = (A[:, None, :] +
    #          X_mm[:, :, None] * Yb[:, None, :] +
    #          Y_mm[:, :, None] * Zb[:, None, :])
    #     P = Axis._rotate_180_about_axis(P, A, U)
    #     return P

    def get_segment_for_station(self, station):
        """Get segment for station (from original)."""
        for i in range(len(self.stations) - 1):
            if self.stations[i] <= station <= self.stations[i + 1]:
                t = (station - self.stations[i]) / (self.stations[i + 1] - self.stations[i]) if self.stations[i + 1] != self.stations[i] else 0
                return i, t
        return None, None

if __name__ == "__main__":
    test_stations = [0, 100, 200]
    test_x = [0, 100, 200]
    test_y = [0, 100, 100]
    test_z = [0, 100, 200]
    axis = Axis(test_stations, test_x, test_y, test_z, units='m')
    assert len(axis) == 3
    logger.info("axis.py test passed.")