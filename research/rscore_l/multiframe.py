import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .geometry import surface_targets
from .prepare import save_json


def prepare_multiframe(data):
    rows = json.loads((data / 'manifest.json').read_text())['train']
    poses = np.load(data / 'train/poses.npy')
    version = 'multiframe-fill-v1'
    report_path = data / 'proc/multiframe_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['version'] == version and all(len(list((data / 'proc' / ('multiframe_' + kind)).glob('*.npz'))) == len(rows)
                for kind in ('training_features', 'features_train')):
            return

    @lru_cache(maxsize=24)
    def cloud(index):
        path = Path(rows[index]['geometry_path'])
        points = np.load(path) if path.exists() else np.empty((0, 3), np.float32)
        return points[np.isfinite(points).all(1)]

    @lru_cache(maxsize=24)
    def tree(index):
        return cKDTree(cloud(index))

    statistics = []
    for index, row in enumerate(tqdm(rows, desc='Multiframe coordinate supervision')):
        distances = np.linalg.norm(poses[:, :3, 3] - poses[index, :3, 3], axis=1)
        allowed = np.array([r['session_id'] == row['session_id'] for r in rows]) & (distances <= 5.)
        neighbors = np.flatnonzero(allowed)
        neighbors = neighbors[np.argsort(distances[neighbors], kind='stable')[:9]]
        world = np.concatenate([cloud(int(j)) for j in neighbors])
        if len(world):
            _, unique = np.unique(np.floor(world / .05).astype(np.int64), axis=0, return_index=True)
            world = world[np.sort(unique)]
        for kind in ('training_features', 'features_train'):
            destination = data / 'proc' / ('multiframe_' + kind)
            destination.mkdir(exist_ok=True)
            path = destination / (row['frame_id'] + '.npz')
            if path.exists():
                saved = dict(np.load(path))
                if str(saved.get('supervision_version', '')) != version:
                    raise RuntimeError('Incompatible multiframe cache: ' + str(path))
                statistics.append(dict(frame=row['frame_id'], kind=kind, single=int(saved['single_valid_count']),
                    multi=int(saved['geometry_valid'].sum()), total=len(saved['geometry_valid']), neighbors=len(neighbors)))
                continue
            feature = dict(np.load(data / 'proc' / kind / path.name))
            original = dict(np.load(data / 'proc' / ('geometry_' + kind) / path.name))
            combined = surface_targets(world, feature['uv'], feature['K'], poses[index], feature['image_size_hw'])
            candidate = np.flatnonzero(combined['geometry_valid'] & ~original['geometry_valid'])
            votes = np.zeros(len(candidate), np.int32)
            for j in neighbors:
                distance, _ = tree(int(j)).query(combined['xyz_target_world'][candidate], distance_upper_bound=.2)
                votes += np.isfinite(distance)
            added = candidate[votes >= 2]
            result = {key: value.copy() for key, value in original.items()}
            for key in ('xyz_target_world', 'geometry_valid', 'geometry_quality', 'sigma_parallel_m', 'sigma_perpendicular_m', 'surface_id'):
                result[key][added] = combined[key][added]
            result.update(supervision_version=np.array(version), support_train_indices=neighbors,
                single_valid_count=np.array(original['geometry_valid'].sum()), added_cross_scan_votes=votes[votes >= 2])
            np.savez_compressed(path, **result)
            statistics.append(dict(frame=row['frame_id'], kind=kind, single=int(original['geometry_valid'].sum()),
                multi=int(result['geometry_valid'].sum()), total=len(result['geometry_valid']), neighbors=len(neighbors)))
        save_json(data / 'proc/multiframe_progress.json', dict(completed=index+1, total=len(rows)))
    totals = {}
    for kind in ('training_features', 'features_train'):
        subset = [item for item in statistics if item['kind'] == kind]
        totals[kind] = {key: sum(item[key] for item in subset) for key in ('single', 'multi', 'total')}
        totals[kind]['single_fraction'] = totals[kind]['single'] / totals[kind]['total']
        totals[kind]['multi_fraction'] = totals[kind]['multi'] / totals[kind]['total']
    save_json(report_path, dict(version=version, totals=totals, frames=statistics,
        missing_source_scans=sum(not Path(row['geometry_path']).exists() for row in rows),
        method='Preserve all single-scan labels; fill unlabeled points from up to 9 same-session training scans within 5m, 0.05m voxel deduplication, same ray-plane and z-buffer checks, >=2 scans within 0.2m of each added target',
        independent_ground_truth=False, static_consistency='Cross-scan proximity is weak evidence, not proof of staticness',
        controls='Same features, PCA, LiDAR graph, Node2Vec, sampling, initialization, loss, 10000 iterations and evaluation as single-frame lidar variant',
        validation_and_test_scans_used=False))


def compare_multiframe(root):
    report = json.loads((root / 'data/proc/multiframe_report.json').read_text())
    results = {}
    for variant in ('glace', 'lidar', 'lidar-multiframe'):
        results[variant] = {split: json.loads((root / 'evaluation' / variant / split / 'summary.json').read_text())
            for split in ('val', 'test')}
    save_json(root / 'multiframe_comparison.json', dict(coverage=report, results=results))
    lines = ['# 单帧与多帧坐标监督对比', '', '固定训练 907 张、验证 303 张、开发测试 148 张；单帧与多帧使用相同 LiDAR 共视图、Node2Vec、特征缓存和 10,000 次迭代预算。', '',
        '| 特征集 | 单帧有效标签 | 多帧有效标签 | 总点数 |', '|---|---:|---:|---:|']
    for kind, value in report['totals'].items():
        lines.append(f"| {kind} | {value['single']} ({value['single_fraction']:.2%}) | {value['multi']} ({value['multi_fraction']:.2%}) | {value['total']} |")
    for split in ('val', 'test'):
        lines.extend(['', f'## {split}', '', '| 方法 | 视觉均值 m / ° | 视觉中位数 m / ° | 视觉成功数 <1m、2° |', '|---|---:|---:|---:|'])
        for variant, values in results.items():
            camera = values[split]['camera']
            lines.append(f"| {variant} | {camera['mean'][0]:.3f} / {camera['mean'][1]:.3f} | {camera['median'][0]:.3f} / {camera['median'][1]:.3f} | {camera['success_1m_2deg']} / {camera['frames']} |")
    lines.extend(['', '## 测试集融合', '', '| 方法 | 均值 m / ° | 成功数 <1m、2° | 挽救 / 损害 |', '|---|---:|---:|---:|'])
    for variant, values in results.items():
        test = values['test']
        fused = test['fused']
        lines.append(f"| {variant} | {fused['mean'][0]:.3f} / {fused['mean'][1]:.3f} | {fused['success_1m_2deg']} / 148 | {test['rescue']} / {test['damage']} |")
    lines.extend(['', '多帧仅补充原来没有标签的点，保留单帧标签；新增目标需至少两帧在 0.2m 内支持。空间一致性不是独立静态真值，覆盖率增加不证明三维标签精度提高。', '',
        '这项对比尚不包含训练后的可靠性头。测试集曾用于开发，GLACE 为已有缓存；R-SCoRe 保留 10 个检索假设，GLACE 为一个假设，不能视作相同总推理计算量。'])
    (root / 'multiframe_comparison.md').write_text('\n'.join(lines) + '\n')


def audit_multiframe(data):
    rows = json.loads((data / 'manifest.json').read_text())['train']
    poses = np.load(data / 'train/poses.npy')
    added_count = 0
    for i, row in enumerate(rows):
        for kind in ('training_features', 'features_train'):
            name = row['frame_id'] + '.npz'
            original = dict(np.load(data / 'proc' / ('geometry_' + kind) / name))
            result = dict(np.load(data / 'proc' / ('multiframe_' + kind) / name))
            valid = original['geometry_valid']
            assert result['geometry_valid'][valid].all()
            for key in ('xyz_target_world', 'geometry_quality', 'sigma_parallel_m', 'sigma_perpendicular_m'):
                np.testing.assert_array_equal(original[key][valid], result[key][valid])
                assert np.isfinite(result[key][result['geometry_valid']]).all()
            assert (result['added_cross_scan_votes'] >= 2).all()
            support = result['support_train_indices']
            assert all(rows[int(j)]['session_id'] == row['session_id'] for j in support)
            assert (np.linalg.norm(poses[support, :3, 3] - poses[i, :3, 3], axis=1) <= 5.00001).all()
            assert len(support) <= 9
            added_count += int((result['geometry_valid'] & ~valid).sum())
    save_json(data / 'proc/multiframe_audit.json', dict(frames=len(rows), added_across_both_caches=added_count,
        original_targets_preserved=True, finite_labels=True, same_training_session_only=True, maximum_neighbor_distance_m=5))
