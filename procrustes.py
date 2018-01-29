# -*- coding: utf-8 -*-
"""
Created on Dec 18 17:12 2017

@author: Denis Tome'
"""

import numpy as np
from numpy.linalg import svd, norm


def procrustes_rot(m1, m2):
    """
    Perform procrustes analysis on a subset of the joints to identify the rotation
    :param m1: gt pose to be aligned to, in the format 3 x N_POINTS
    :param m2: predicted pose to be aligned, in the format 3 x N_POINTS
    :return: rotation matrix 3 x 3
    """
    U, _, V = svd(m1.dot(m2.T))
    return U.dot(V)


def procrustes(m1, m2):
    """
    Perform procrustes analysis on a subset of the joints
    :param m1: gt pose to be aligned to, in the format 3 x N_POINTS
    :param m2: predicted pose to be aligned, in the format 3 x N_POINTS
    :return: rotation matrix 3 x 3
             scale
             translation
    """
    rot_matrix = procrustes_rot(m1, m2)
    rotated_pose = np.dot(rot_matrix, m2)

    gt_mean_vals = np.mean(m1, axis=1)[:, np.newaxis]
    pred_mean_vals = np.mean(rotated_pose, axis=1)[:, np.newaxis]

    scale = norm(m1 - gt_mean_vals, 'fro') / norm(m2 - pred_mean_vals, 'fro')
    translation = gt_mean_vals - scale * pred_mean_vals
    return rot_matrix, scale, translation


def align_data(gt_3d, pred):
    """
    Procrustes analysis on the poses and return aligned pose
    :param gt_3d: sub-set of joints to use for alignment (3 x SET_LANDMARKS)
    :param pred: pose returned by LiftingFromTheDeep module (3 x SET_LANDMARKS)
    :return: aligned pose (3 x SET_LANDMARKS)
    """
    rot, s, t = procrustes(gt_3d, pred)
    new_pose = np.dot(rot, pred)*s + t
    return new_pose
