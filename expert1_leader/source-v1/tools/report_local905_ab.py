import argparse
import json
from pathlib import Path

from tools.compare_local905 import compare


def read_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if report['subset'] != 'test' or report['frames'] != 313:
        raise ValueError(f'Expected fixed 313-frame test report: {path}')
    return report


def metrics(report):
    return {
        'frames': report['frames'],
        'successful_frames': report['successful_frames'],
        'failed_frames': report['failed_frames'],
        'mean_mpe_m': report['all_frame_mpe_mean_m'],
        'median_mpe_m': report['all_frame_mpe_median_m'],
        'p90_mpe_m': report['all_frame_mpe_p90_m'],
        'mean_moe_deg': report['all_frame_moe_mean_deg'],
        'median_moe_deg': report['all_frame_moe_median_deg'],
        'p90_moe_deg': report['all_frame_moe_p90_deg'],
        'rotation_gt_10_deg': report['all_frame_rotation_gt_10_deg'],
        'rotation_gt_90_deg': report['all_frame_rotation_gt_90_deg'],
        'predictions_sha256': report['predictions_sha256'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar', type=Path, required=True)
    parser.add_argument('--A', type=Path, required=True)
    parser.add_argument('--B', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = {
        'lidar': read_report(args.lidar),
        'A': read_report(args.A),
        'B': read_report(args.B),
    }
    frame_lists = [[row['scan'] for row in report['rows']] for report in reports.values()]
    if frame_lists[0] != frame_lists[1] or frame_lists[0] != frame_lists[2]:
        raise ValueError('A/B and LiDAR reports have different test frames')
    result = {
        'protocol': 'local905_ab_fixed_313_v1',
        'metrics': {name: metrics(report) for name, report in reports.items()},
        'A_minus_lidar': compare(reports['A'], reports['lidar']),
        'B_minus_lidar': compare(reports['B'], reports['lidar']),
        'A_minus_B': compare(reports['A'], reports['B']),
    }
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
