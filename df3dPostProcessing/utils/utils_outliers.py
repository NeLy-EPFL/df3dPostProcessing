import itertools

import numpy as np
import scipy.interpolate
#import deepfly.signal_util


def find_outliers(camNet):
    """
    This function determines which 3D points were not correctly
    estimated by df3d.
    It uses the segment length and an absolute threshold to
    determine if a joint position is an outlier or not.

    Parameters:
    -----------
    camNet : camNet object from df3d

    Returns:
    --------
    outliers: dict
        A dictionary that holds the `image_ids` (frame numbers)
        `joint_ids` of each outlier.
        Additionally, it holds the `median_lengths` of each leg segment,
        the `start_indices` and `stop_indices` in joint number used to
        compute the segments.
    """
    start_indices = np.array([0, 1, 2, 3,
                              5, 6, 7, 8,
                              10, 11, 12, 13,
                              19, 20, 21, 22,
                              24, 25, 26, 27,
                              29, 30, 31, 32, ],
                             dtype=np.int)
    stop_indices = start_indices + 1
    lengths = np.linalg.norm(
        camNet.points3d[:, start_indices, :] - camNet.points3d[:, stop_indices, :], axis=2)
    median_lengths = np.median(lengths, axis=0)
    length_outliers = (
        lengths > median_lengths *
        1.4) | (
        lengths < median_lengths *
        0.4)
    outlier_mask = np.zeros(camNet.points3d.shape, dtype=np.bool)
    for i, mask_offset in enumerate([0, 5, 10, 19, 24, 29]):
        claw_outliers = np.where(
            length_outliers[:, i * 4 + 3] & ~length_outliers[:, i * 4 + 2])[0]
        tarsus_outliers = np.where(
            length_outliers[:, i * 4 + 3] & length_outliers[:, i * 4 + 2])[0]
        tibia_outliers = np.where(
            length_outliers[:, i * 4 + 2] & length_outliers[:, i * 4 + 1])[0]
        femur_outliers = np.where(
            length_outliers[:, i * 4 + 1] & length_outliers[:, i * 4 + 0])[0]
        outlier_mask[femur_outliers, 1 + mask_offset] = True
        outlier_mask[tibia_outliers, 2 + mask_offset] = True
        outlier_mask[tarsus_outliers, 3 + mask_offset] = True
        outlier_mask[claw_outliers, 4 + mask_offset] = True

    outlier_mask = np.logical_or(outlier_mask, np.abs(camNet.points3d) > 5)

    outlier_image_ids = np.where(outlier_mask)[0]
    outlier_joint_ids = np.where(outlier_mask)[1]
    return {
        "image_ids": outlier_image_ids,
        "joint_ids": outlier_joint_ids,
        "median_lengths": median_lengths,
        "start_indices": start_indices,
        "stop_indices": stop_indices}


def _triangluate_specific_cameras(camNet, cam_id_list, img_id, j_id):
    from deepfly.cv_util import triangulate_linear
    cam_list_iter = list()
    points2d_iter = list()
    for cam in [camNet.cam_list[cam_idx] for cam_idx in cam_id_list]:
        cam_list_iter.append(cam)
        points2d_iter.append(cam[img_id, j_id, :])
    return triangulate_linear(cam_list_iter, points2d_iter)


def interpolate_remaining_outliers(camNet, outliers):
    for joint_id in np.unique(outliers["joint_ids"]):
        joint_mask = outliers["joint_ids"] == joint_id
        frames = np.unique(outliers["image_ids"][joint_mask])
        non_outlier_frames = np.array(
            [i for i in range(camNet.points3d.shape[0]) if i not in frames])
        non_outlier_values = camNet.points3d[non_outlier_frames, joint_id]
        interp_func = scipy.interpolate.PchipInterpolator(
            non_outlier_frames, non_outlier_values, axis=0)
        camNet.points3d[frames, joint_id] = interp_func(frames)
    return camNet


def correct_outliers(camNet):
    outliers = find_outliers(camNet)
    median_lengths = outliers["median_lengths"]
    start_indices = outliers["start_indices"]
    stop_indices = outliers["stop_indices"]

    for img_id, joint_id in zip(outliers["image_ids"], outliers["joint_ids"]):
        reprojection_errors = list()
        segment_length_diff = list()
        points_using_2_cams = list()
        # Select cameras based on which side the joint is on
        all_cam_ids = [0, 1, 2] if joint_id < 19 else [4, 5, 6]
        for subset_cam_ids in itertools.combinations(all_cam_ids, 2):
            points3d_using_2_cams = _triangluate_specific_cameras(
                camNet, subset_cam_ids, img_id, joint_id)

            new_diff = 0
            median_index = np.where(stop_indices == joint_id)[0]
            if len(median_index) > 0:
                new_diff += np.linalg.norm(points3d_using_2_cams - \
                                           camNet.points3d[img_id, joint_id - 1]) - median_lengths[median_index]
            median_index = np.where(start_indices == joint_id)[0]
            if len(median_index) > 0:
                new_diff += np.linalg.norm(points3d_using_2_cams - \
                                           camNet.points3d[img_id, joint_id + 1]) - median_lengths[median_index]
            segment_length_diff.append(new_diff)

            def reprojection_error_function(cam_id): return camNet.cam_list[cam_id].project(
                points3d_using_2_cams) - camNet.cam_list[cam_id].points2d[img_id, joint_id]
            reprojection_error = np.mean(
                [reprojection_error_function(cam_id) for cam_id in subset_cam_ids])
            reprojection_errors.append(reprojection_error)
            points_using_2_cams.append(points3d_using_2_cams)

        best_cam_tuple_index = np.argmin(segment_length_diff)

        old_diff = 0
        new_diff = 0
        median_index = np.where(stop_indices == joint_id)[0]
        if len(median_index) > 0:
            old_diff += np.linalg.norm(camNet.points3d[img_id,
                                                       joint_id] - camNet.points3d[img_id,
                                                                                   joint_id - 1]) - median_lengths[median_index]
            new_diff += np.linalg.norm(points_using_2_cams[best_cam_tuple_index] -
                                       camNet.points3d[img_id, joint_id - 1]) - median_lengths[median_index]
        median_index = np.where(start_indices == joint_id)[0]
        if len(median_index) > 0:
            old_diff += np.linalg.norm(camNet.points3d[img_id,
                                                       joint_id] - camNet.points3d[img_id,
                                                                                   joint_id + 1]) - median_lengths[median_index]
            new_diff += np.linalg.norm(points_using_2_cams[best_cam_tuple_index] -
                                       camNet.points3d[img_id, joint_id + 1]) - median_lengths[median_index]

        if new_diff < old_diff:
            camNet.points3d[img_id,
                            joint_id] = points_using_2_cams[best_cam_tuple_index]

    outliers = find_outliers(camNet)
    camNet = interpolate_remaining_outliers(camNet, outliers)

    camNet.points3d = deepfly.signal_util.filter_batch(
        camNet.points3d, freq=100)
    return camNet
