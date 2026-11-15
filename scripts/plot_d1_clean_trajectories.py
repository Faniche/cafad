"""plot_d1_clean_trajectories.py | 2026-11-13 | 绘制 D1 注入前全部航班三维航迹 SVG | Codex."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import duckdb
import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


DEFAULT_INPUT = Path('data/0_preprocess/preprocessed_data_d1/3#20261112_163241#42/data/observations.parquet')
DEFAULT_OUTPUT = Path('tmp/d1_clean_all_flights_3d_40cm.svg')


def build_parser() -> argparse.ArgumentParser:
    """构造只读绘图参数，默认指向最新 D1 预处理产物。"""
    parser = argparse.ArgumentParser(description='将注入前 D1 的全部航班按时间连接为三维 SVG 航迹图。')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='预处理 observations.parquet 路径。')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='输出 SVG 路径。')
    parser.add_argument('--max-points-per-flight', type=int, default=2_000,
                        help='单航班展示上限；超过时等间隔抽样，仍按时间连线。')
    parser.add_argument('--line-width', type=float, default=0.18, help='SVG 航迹线宽（pt）。')
    parser.add_argument('--alpha', type=float, default=0.24, help='SVG 航迹不透明度。')
    return parser


def _flight_paths(input_path: Path, max_points_per_flight: int) -> Iterable[np.ndarray]:
    """按 flight_id、time_ns 流式读取经纬度，并为每条航迹受控抽样。"""
    if max_points_per_flight < 2:
        raise ValueError('--max-points-per-flight 至少为 2')
    query = '''
        SELECT flight_id, time_ns, longitude_deg, latitude_deg, geometric_altitude_m
        FROM read_parquet(?)
        WHERE longitude_deg IS NOT NULL AND latitude_deg IS NOT NULL AND geometric_altitude_m IS NOT NULL
        ORDER BY flight_id, time_ns
    '''
    with duckdb.connect(':memory:') as database:
        reader = database.execute(query, [str(input_path)]).to_arrow_reader(batch_size=250_000)
        current_flight: str | None = None
        points: list[tuple[float, float, float]] = []
        for batch in reader:
            frame = batch.to_pandas()
            for flight_id, longitude, latitude, altitude in frame[
                    ['flight_id', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m']].itertuples(
                    index=False, name=None):
                flight = str(flight_id)
                if current_flight is not None and flight != current_flight:
                    yield _downsample(points, max_points_per_flight)
                    points = []
                current_flight = flight
                points.append((float(longitude), float(latitude), float(altitude) / 1_000.0))
        if points:
            yield _downsample(points, max_points_per_flight)


def _downsample(points: list[tuple[float, float, float]], maximum: int) -> np.ndarray:
    """等间隔保留端点，避免百万级原始顶点生成难以打开的 SVG。"""
    values = np.asarray(points, dtype=float)
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices]


def render(input_path: Path, output_path: Path, max_points_per_flight: int, line_width: float, alpha: float) -> dict[str, int]:
    """以固定 40×40 cm 画布输出全航班的经度、纬度和几何高度连续航迹。"""
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f'找不到预处理观测表：{resolved_input}')
    paths = list(_flight_paths(resolved_input, max_points_per_flight))
    if not paths:
        raise ValueError('观测表中没有有效经纬度航迹')
    segments = [path for path in paths if len(path) >= 2]
    all_points = np.concatenate(segments)
    figure = plt.figure(figsize=(40 / 2.54, 40 / 2.54), constrained_layout=True)
    axis = figure.add_subplot(projection='3d')
    axis.add_collection3d(Line3DCollection(segments, colors='#1769aa', linewidths=line_width, alpha=alpha))
    longitude_min, latitude_min, altitude_min = np.min(all_points, axis=0)
    longitude_max, latitude_max, altitude_max = np.max(all_points, axis=0)
    longitude_padding = max((longitude_max - longitude_min) * 0.015, 0.02)
    latitude_padding = max((latitude_max - latitude_min) * 0.015, 0.02)
    axis.set_xlim(longitude_min - longitude_padding, longitude_max + longitude_padding)
    axis.set_ylim(latitude_min - latitude_padding, latitude_max + latitude_padding)
    axis.set_zlim(max(0.0, altitude_min), altitude_max * 1.015)
    axis.set_box_aspect((longitude_max - longitude_min, latitude_max - latitude_min, altitude_max - altitude_min))
    axis.view_init(elev=28, azim=-62)
    axis.set_xlabel('Longitude (deg)')
    axis.set_ylabel('Latitude (deg)')
    axis.set_zlabel('Geometric altitude (km)')
    axis.set_title(f'D1 clean trajectories: {len(segments):,} flights (pre-injection)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format='svg', facecolor='white')
    plt.close(figure)
    return {'flight_count': len(segments), 'display_point_count': int(sum(len(path) for path in segments))}


def main() -> None:
    """运行独立绘图任务并报告 SVG 输出摘要。"""
    args = build_parser().parse_args()
    summary = render(args.input, args.output, args.max_points_per_flight, args.line_width, args.alpha)
    print(f'已生成 {args.output}；航班={summary["flight_count"]}；展示点={summary["display_point_count"]}')


if __name__ == '__main__':
    main()
