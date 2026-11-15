"""Audit original UMN field definitions and observed discrete status values.

Only the UMN directory is read. Evidence is from its dictionaries, embedded units,
and the three author-provided FlightCode archives; undocumented fields stay explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.utils.logging_utils import setup_project_logging, close_project_logging


CORE = {
    'time': ('时间戳/该记录时钟', 's'), 'h': ('气压高度；旧字典标为 AGL，保留源参考面', 'm'),
    'h_msl': ('海平面参考高度；具体融合/换算方法依版本核对', 'm'),
    'ias': ('指示空速，不等于地速', 'm/s'), 'Ps': ('静压', 'kPa'), 'Pd': ('动压/差压', 'kPa'),
    'Pd_aoa': ('迎角通道差压', 'kPa'), 'Pd_aos': ('侧滑角通道差压', 'kPa'),
    'Ps_2': ('五孔探针静压', 'kPa'), 'Pt': ('五孔探针中心总压', 'kPa'),
    'Pt_a1': ('五孔探针迎角端口1总压', 'kPa'), 'Pt_a2': ('五孔探针迎角端口2总压', 'kPa'),
    'Pt_b1': ('五孔探针侧滑端口1总压', 'kPa'), 'Pt_b2': ('五孔探针侧滑端口2总压', 'kPa'),
    'aoa': ('迎角 alpha', 'rad'), 'aos': ('侧滑角 beta', 'rad'),
    'p': ('机体X轴角速度/滚转角速度', 'rad/s'), 'q': ('机体Y轴角速度/俯仰角速度', 'rad/s'),
    'r': ('机体Z轴角速度/偏航角速度', 'rad/s'),
    'ax': ('机体X轴加速度', 'm/s²'), 'ay': ('机体Y轴加速度', 'm/s²'), 'az': ('机体Z轴加速度', 'm/s²'),
    'hx': ('X轴磁场', 'Gauss'), 'hy': ('Y轴磁场', 'Gauss'), 'hz': ('Z轴磁场', 'Gauss'),
    'lat': ('GPS大地纬度', 'deg'), 'lon': ('GPS大地经度', 'deg'), 'alt': ('GPS WGS84参考高度', 'm'),
    'vn': ('GPS北向速度', 'm/s'), 've': ('GPS东向速度', 'm/s'), 'vd': ('GPS向下速度；爬升方向与其相反', 'm/s'),
    'navlat': ('导航滤波纬度；对应 FlightCode 为 rad，旧PDF的deg标注不可直接通用', 'rad（该源码版本）'),
    'navlon': ('导航滤波经度；对应 FlightCode 为 rad，旧PDF的deg标注不可直接通用', 'rad（该源码版本）'),
    'navalt': ('导航滤波 WGS84参考高度估计', 'm'),
    'navvn': ('导航滤波北向速度估计', 'm/s'), 'navve': ('导航滤波东向速度估计', 'm/s'), 'navvd': ('导航滤波向下速度估计', 'm/s'),
    'phi': ('Euler滚转角；源码如此定义，旧PDF有phi/psi互换错误', 'rad'),
    'theta': ('Euler俯仰角', 'rad'), 'the': ('Euler俯仰角，旧字段拼写', 'rad'),
    'psi': ('Euler偏航角，不等于GPS地面航迹角；旧PDF有phi/psi互换错误', 'rad'),
    'pitch_ref': ('俯仰角指令/参考值', 'rad'), 'roll_ref': ('滚转角指令/参考值', 'rad'),
    'phi_cmd': ('滚转角指令', 'rad'), 'theta_cmd': ('俯仰角指令', 'rad'), 'psi_cmd': ('偏航角指令', 'rad'),
    'p_cmd': ('滚转角速度指令', 'rad/s'), 'q_cmd': ('俯仰角速度指令', 'rad/s'), 'r_cmd': ('偏航角速度指令', 'rad/s'),
    'ias_cmd': ('指示空速指令', 'm/s'), 'h_cmd': ('高度指令，参考面依版本', 'm'),
    'gndtrk_cmd': ('相对真北的地面航迹角指令', 'rad'), 'gamma_cmd': ('航迹倾角指令', 'rad'),
    'aoa_cmd': ('迎角指令', 'rad'), 'aos_cmd': ('侧滑角指令', 'rad'),
    'GPS_TOW': ('GPS周内秒；不等于完整UTC日期时间', 's'),
    'mode': ('飞控工作模式：0=dump data；1=manual control；2=autopilot control', '枚举'),
    'run_num': ('自动驾驶接通次数计数，不是飞行阶段标签', '计数'),
    'satVisible': ('参与定位解算/可见卫星数，具体口径依接收机', '计数'),
    'navValid': ('GPS解算有效标记/位域；0=有效，非0需按驱动定义', '状态/位域'),
    'imuStatus': ('IMU读取状态；对应errdefs，不是正常/异常真值', '枚举'),
    'imu_status': ('IMU读取状态，旧字段拼写；解码需对照记录固件版本', '枚举'),
    'gpsStatus': ('GPS读取状态；对应errdefs，不等同于navValid', '枚举'),
    'gps_status': ('GPS读取状态，旧字段拼写；解码需对照记录固件版本', '枚举'),
    'adStatus': ('气压传感器状态位域；AMS_pressure驱动低4位为各压力通道读取失败位', '位域'),
    'navStatus': ('导航滤波状态；对应版本errdefs的导航状态', '枚举'),
    'cpuLoad': ('飞控CPU负载', '%'), 'cpuload': ('飞控CPU负载，旧字段拼写', '%'),
    'var1': ('部分FDI试验的副翼故障估计值；不是统一故障真值', 'rad（旧字典）'),
    'var2': ('部分FDI试验的陀螺故障估计值；不是统一故障真值', 'rad/s（旧字典）'),
    'var3': ('部分FDI试验的MRAC是否启用', '布尔/数值标记'),
    'var4': ('部分FDI试验的自适应增益X', '无量纲/试验定义'),
    'var5': ('部分FDI试验的自适应增益Y', '无量纲/试验定义'),
}
SURFACES = {'de': '升降舵', 'de_l': '左升降舵', 'de_r': '右升降舵', 'dr': '方向舵',
            'da_l': '左副翼', 'da_r': '右副翼', 'df_l': '左襟翼', 'df_r': '右襟翼', 'dthr': '油门', 'dg': '起落架/接收机复用开关'}
VIREO = {
    'latitude': ('纬度', 'deg'), 'longitude': ('经度', 'deg'), 'altitude_gps': ('GPS高度', 'm'),
    'altitude_ekf': ('EKF高度估计', 'm'), 'Vn': ('北向速度', 'm/s'), 'Ve': ('东向速度', 'm/s'), 'Vd': ('向下速度', 'm/s'),
    'Phi': ('滚转角', 'deg'), 'Theta': ('俯仰角', 'deg'), 'Psi': ('机体偏航角', 'deg'),
    'pressure': ('气压', 'mbar/hPa'), 'temperature': ('该传感器温度，不能自动视作外界气温', '°C（G2文档）'),
    'airspeed': ('空速；G2原始Air包的单位为kts', 'kt'), 'altitude_baro': ('气压高度，参考面依版本', 'm'),
    'altitude_true': ('GPS/气压混合高度估计，不是无误差真值', 'm'), 'vertical_speed': ('垂直速度', 'ft/min'),
    'wind_dir': ('风来向', 'deg'), 'wind_speed': ('风速', 'kt'), 'pitot_scale_error': ('皮托管比例误差', '无量纲'),
    'unix_time': ('接收机输出Unix时间字段；有效性需单独核验', 's'), 'satellites': ('GPS卫星数', '计数'),
    'horizontal_accuracy': ('水平精度估计', 'm'),
    'vertical_accuracy': ('垂直精度估计', 'm'), 'pdop': ('位置精度衰减因子', '无量纲'),
    'fix_type': ('GPS定位类型：0无定位，1航位推算，2二维定位，3三维定位；字段是否有效仍需核验', '枚举（G2字典）'),
    'status_flag': ('该数据包/模块的状态标记，不能直接套用G1 errdefs', '状态码，需按版本解码'),
    'target_ground_track': ('目标地面航迹角', 'deg'), 'target_roll_angle': ('目标滚转角', 'deg'),
    'target_alt_MSL': ('目标海平面高度', 'ft'), 'target_climb_rate': ('目标爬升率', 'ft/s'),
    'target_theta': ('目标俯仰角', 'deg'), 'target_q': ('目标俯仰角速度', 'rad/s'), 'target_airspeed': ('目标空速', 'kt'),
    'Master_SWT': ('飞控主开关状态；不是阶段标签', '开关码'), 'FlightMode_SWT': ('飞控模式选择开关；不是阶段标签', '开关码'),
    'load_avg': ('内核平均运行队列长度；1.0通常表示CPU饱和，不是百分数', '无量纲'), 'avionics_VCC': ('航电供电电压', 'V'),
    'external_battery': ('外部电池总电压', 'V'), 'external_battery_avg_cell': ('外部电池平均单体电压', 'V/cell'),
    'current_draw': ('电流', 'A'), 'total_current_used': ('累计耗电量', 'mAh'),
}
ERRDEFS = [('got_invalid', '未收到数据'), ('checksum_err', '校验和错误'), ('gps_nolock', 'GPS未锁定'),
           ('data_valid', '数据有效'), ('noPacketHeader', '收到部分数据但未找到包头'), ('incompletePacket', '包不完整'),
           ('TU_only', '导航滤波仅时间更新'), ('gps_aided', '导航滤波有GPS辅助'),
           ('inflated_gps_aided', '使用放大噪声的GPS辅助；仅扩展版本'),
           ('refining', '导航滤波初始化细化；仅扩展版本'), ('MU_ahrs', 'AHRS测量更新；仅扩展版本')]


def classify(field):
    canonical = re.sub(r'^(NAVREPAIRED|variant_\d+)\.', '', field)
    if canonical.startswith('_') or canonical.endswith('._sample_index'):
        return '整理元数据', 'CSV整理时新增的身份/行号/时钟并集或源缺失标记，不是原始飞行状态标签', '整理定义', '整理脚本'
    if canonical in CORE:
        meaning, unit = CORE[canonical]
        return '原始平面字段', meaning, unit, '官方字典/随记录提供的FlightCode；跨版本仍需核对'
    if canonical.startswith('dtc.'):
        leaf = canonical.rsplit('.', 1)[-1]
        if leaf in {'lat', 'lon'}:
            return '官方同步G2', '官方同步EKF纬度/经度估计', 'deg（内嵌units）', 'MAT内嵌units'
        if leaf in {'phi', 'theta', 'psi'}:
            return '官方同步G2', {'phi': '滚转角', 'theta': '俯仰角', 'psi': '偏航角'}[leaf], 'deg（内嵌units）', 'MAT内嵌units'
        if leaf in {'alt_ekf', 'alt_baro'}:
            return '官方同步G2', 'EKF高度估计' if leaf == 'alt_ekf' else '气压AGL高度', 'm', 'MAT内嵌units'
        if leaf in {'Vn', 'Ve', 'Vd', 'airspeed'}:
            return '官方同步G2', '导航速度分量' if leaf != 'airspeed' else '指示空速', 'm/s', 'MAT内嵌units'
        if leaf in {'dynpress', 'staticpress'}:
            return '官方同步G2', '动压' if leaf == 'dynpress' else '静压', 'Pa', 'MAT内嵌units'
        if '.imu.' in canonical and leaf in CORE:
            if leaf in {'hx', 'hy', 'hz'}:
                return '官方同步G2', CORE[leaf][0], '[-]（对应内嵌units）', 'MAT内嵌units'
            return '官方同步G2', *CORE[leaf], 'MAT内嵌units'
        if leaf == 'time':
            return '官方同步G2', '官方同步时钟', 's', 'MAT内嵌time'
        if leaf == 'SWT':
            return '官方同步G2', '遥控开关状态；源同步处理可能使跳变附近出现非整数值', 'two-state switch（内嵌units）', 'MAT内嵌units'
        if '.bird.' in canonical:
            return '官方同步G2', '左右迎角/侧滑探针或风速通道，按字段后缀区分', 'm/s（anem）或deg（alpha/beta）', 'MAT内嵌units'
        if any(part in canonical for part in ('.pilot.', '.actuator.', '.surface.')):
            return '官方同步G2', '驾驶员/执行器/实测舵面通道，具体通道见所属记录units', '归一化量或deg；不能跨版本混用', 'MAT内嵌units'
    if '.' in canonical:
        group, leaf = canonical.rsplit('.', 1)
        if leaf == 'time':
            return '异步G2原始包', f'{group}原始时间戳', 's', 'G2 FlightDataFormat'
        if leaf in VIREO:
            return '异步G2原始包', *VIREO[leaf], 'G2 FlightDataFormat；新增协议字段另标待核实'
        if group == 'imu' and leaf in CORE:
            if leaf in {'hx', 'hy', 'hz'}:
                return '异步G2原始包', CORE[leaf][0], '随磁传感器型号变化，G2文档未统一', 'G2 FlightDataFormat'
            return '异步G2原始包', *CORE[leaf], 'G2 FlightDataFormat'
        if group == 'ekf' and leaf.endswith('_bias'):
            return '异步G2原始包', 'EKF估计的陀螺/加速度计偏置', 'rad/s或m/s²，按传感器区分', 'G2 FlightDataFormat'
        if re.fullmatch(r'chan_[ca]\d+', leaf):
            return '异步G2原始包', '遥控器/执行器编号通道；功能由飞控配置决定', '归一化量或rad，依飞控配置', 'G2 FlightDataFormat'
    base = canonical.removesuffix('_filt')
    if base != canonical and base in CORE:
        return '滤波量', '滤波后的' + CORE[base][0], CORE[base][1], 'FlightCode'
    for suffix, meaning in [('_pos', '实测位置'), ('_in', '遥控器输入指令')]:
        if canonical.endswith(suffix) and canonical[:-len(suffix)] in SURFACES:
            stem = canonical[:-len(suffix)]
            return '舵面与控制', SURFACES[stem] + meaning, '0–1（油门）或rad（舵角）', 'FlightCode globaldefs.h'
    if canonical in SURFACES:
        return '舵面与控制', SURFACES[canonical] + '指令', '0–1（油门）或rad（舵角）', 'FlightCode globaldefs.h'
    if canonical in {'l_alpha', 'r_alpha', 'l_beta', 'r_beta'}:
        return '空速与探针', '左/右探针迎角或侧滑角', 'rad', 'FlightCode globaldefs.h'
    if re.fullmatch(r'[pqr]_bias(?:_.*)?', canonical):
        return '偏置估计', '相应轴陀螺偏置；后缀区分滤波器/试验版本', 'rad/s', 'FlightCode；后缀版本需核对'
    if re.fullmatch(r'a[xyz]_bias(?:_.*)?', canonical):
        return '偏置估计', '相应轴加速度计偏置；旧PDF rad/s为量纲错误', 'm/s²', 'FlightCode；后缀版本需核对'
    if canonical.startswith('etime_'):
        return '运行诊断', canonical[6:] + '模块运行耗时', 's（记录版本需核对）', '名称/任务代码，非飞行阶段'
    if canonical.startswith('jitter') or canonical.startswith('threadJitter'):
        return '运行诊断', '实时任务调度抖动', 'ns（旧字典）；其他版本需核对', '旧字典/字段名'
    if canonical.startswith('cell_'):
        return '专用蜂窝导航', '蜂窝导航试验量：ID=基站标识，RSSI=接收强度，TA=定时提前，lat/lon/alt=估计位置；其余按试验定义', '单位/枚举未在公共字典完整说明', '字段名推断，待对应试验代码核实'
    if canonical.endswith(('Status', '_flag')) or 'Status_' in canonical:
        return '版本专用状态', '模块/试验状态码；不应直接套用其他固件枚举', '需按对应版本解码', '待核实'
    return '版本/试验专用', '公共字典与已检查源码未给出可靠的一一对应解释；保留原名，不能据字段名认定其物理真值含义', '待核实', '未核实，不猜测'


def categorical(field):
    leaf = field.rsplit('.', 1)[-1]
    return (leaf in {'mode', 'run_num', 'var3', 'fix_type', 'navValid', 'Master_SWT', 'FlightMode_SWT', 'SWT'}
            or 'status' in leaf.lower() or leaf.endswith('_flag') or field.startswith('_source_missing.'))


def scan_file(path):
    import numpy as np
    import pandas as pd
    columns = next(csv.reader(Path(path).open(encoding='utf-8-sig')))
    selected = [c for c in columns if categorical(c)]
    counts, nonmissing = {c: Counter() for c in selected}, Counter()
    integers = {c: Counter() for c in selected}
    fractional, invalid_mode = Counter(), Counter()
    capped = set()
    if selected:
        with pd.read_csv(path, usecols=selected, chunksize=100_000) as reader:
            for chunk in reader:
                for column in selected:
                    values = chunk[column].dropna()
                    nonmissing[column] += len(values)
                    is_integer = np.isfinite(values) & (values % 1 == 0)
                    fractional[column] += int((~is_integer).sum())
                    integers[column].update({str(int(value)): int(count) for value, count in values[is_integer].value_counts().items()})
                    if column.rsplit('.', 1)[-1] == 'mode':
                        invalid_mode[column] += int((~values.isin([0, 1, 2])).sum())
                    if column in capped:
                        continue
                    for value, count in values.value_counts().items():
                        number = float(value)
                        key = str(int(number)) if number.is_integer() else repr(number)
                        counts[column][key] += int(count)
                    if len(counts[column]) > 512:
                        capped.add(column)
    return {'file': Path(path).name, 'columns': columns,
            'categorical': {c: {'counts': dict(counts[c]), 'integer_counts': dict(integers[c]),
                                'noninteger_count': fractional[c], 'invalid_mode_count': invalid_mode[c],
                                'nonmissing': nonmissing[c], 'values_capped': c in capped} for c in selected}}


def inspect_mode_sources(root, manifest, records):
    """Compare malformed source HDF5 columns with companion MAT; never alter either."""
    import h5py
    import numpy as np
    from scipy.io import loadmat
    checks = []
    for record in records:
        if not record['categorical'].get('mode', {}).get('invalid_mode_count'):
            continue
        item = next(i for i in manifest['items'] if i['uuid'][:8] in record['file'])
        check = {'file': record['file'], 'title': item['title'], 'source_checks': []}
        for spec in item['files']:
            path = root / spec['relative_path']
            if path.suffix.lower() not in {'.h5', '.hdf5', '.mat'}:
                continue
            entry = {'source': spec['relative_path']}
            try:
                if path.suffix.lower() in {'.h5', '.hdf5'}:
                    with h5py.File(path) as h:
                        arrays = {k: np.asarray(h[k]).reshape(-1) for k in ('mode', 'Pd', 'time', 'lat', 'imuStatus') if k in h}
                    entry['mode_time_lat_all_equal_Pd'] = all(
                        k in arrays and np.array_equal(arrays[k], arrays['Pd'], equal_nan=True)
                        for k in ('mode', 'time', 'lat')) if 'Pd' in arrays else False
                else:
                    arrays = {k: np.asarray(v).reshape(-1) for k, v in loadmat(
                        path, variable_names=['mode', 'Pd', 'time', 'lat', 'imuStatus']).items() if not k.startswith('__')}
                entry['first_values'] = {k: v[:4].tolist() for k, v in arrays.items()}
                if 'mode' in arrays:
                    entry['mode_values'] = np.unique(arrays['mode']).tolist()[:20]
                    entry['mode_valid_012'] = bool(np.isin(arrays['mode'], [0, 1, 2]).all())
            except Exception as exc:
                entry['read_error'] = str(exc)
            check['source_checks'].append(entry)
        checks.append(check)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data/processed/umn')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    logger, log_path = setup_project_logging(console=False, name='umn_field_reference', filename_prefix='umn_field_reference')
    try:
        root = args.data_root.resolve()
        out = root / 'field_reference'
        out.mkdir(exist_ok=True)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        field_files, discrete, records = Counter(), {}, []
        files = sorted((root / 'csv').glob('*.csv'))
        logger.info('开始字段审计：%d份UMN CSV；进程=%d', len(files), args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {executor.submit(scan_file, str(path)): path for path in files}
            for future in as_completed(pending):
                result = future.result()
                records.append(result)
                field_files.update(result['columns'])
                for name, stats in result['categorical'].items():
                    target = discrete.setdefault(name, {'counts': Counter(), 'integer_counts': Counter(),
                                                         'noninteger_count': 0, 'nonmissing': 0, 'files': 0, 'values_capped': False})
                    target['counts'].update(stats['counts'])
                    target['integer_counts'].update(stats['integer_counts'])
                    target['noninteger_count'] += stats['noninteger_count']
                    target['nonmissing'] += stats['nonmissing']
                    target['files'] += 1
                    target['values_capped'] |= stats['values_capped']
                if len(records) % 25 == 0 or len(records) == len(files):
                    logger.info('已审计 %d/%d 份', len(records), len(files))
        # Extract only the explicitly selected author-provided source text as evidence.
        evidence = []
        for item in manifest['items']:
            for spec in item['files']:
                if 'flightcode' not in spec['name'].lower():
                    continue
                with zipfile.ZipFile(root / spec['relative_path']) as archive:
                    for member in archive.namelist():
                        if '/.git/' in member:
                            continue
                        if member.endswith(('globaldefs.h', 'AMS_pressure.c', 'baldr_datalog.h', 'standard_datalog.h')):
                            target = out / 'source_evidence' / item['title'].replace(' ', '_') / Path(member).name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            text = archive.read(member).decode('utf-8', errors='replace').replace('\r\r\n', '\n').replace('\r\n', '\n')
                            target.write_text(text, encoding='utf-8')
                            evidence.append(str(target.relative_to(out)))
        fields = [{'field': field, 'files': field_files[field], 'group': classify(field)[0],
                   'meaning': classify(field)[1], 'unit': classify(field)[2], 'evidence': classify(field)[3]} for field in sorted(field_files)]
        subjects = Counter(tag for item in manifest['items'] for tag in {s['value'] for s in item['metadata'].get('dc.subject', [])})
        quality_checks = inspect_mode_sources(root, manifest, records)
        report = {'scanned_files': len(records), 'distinct_columns': len(fields), 'fields': fields,
                  'categorical': {name: {**stats, 'counts': dict(stats['counts'])} for name, stats in sorted(discrete.items())},
                  'subject_tags': dict(subjects), 'source_evidence': evidence, 'mode_source_quality': quality_checks,
                  'per_file': sorted(records, key=lambda r: r['file']), 'log_file': str(log_path),
                  'counts_note': 'integer_counts和noninteger_count为全量统计；counts遇到高基数字段时仅保留触发上限前的批次，values_capped标明，不能当完整分布。'}
        (out / 'field_reference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        lines = ['# UMN 原始字段与类别/状态取值', '',
                 f'全量检查 {len(records)} 份 CSV，共 {len(fields)} 个不同列名。包含机型/固件差异、变体前缀，以及整理脚本新增的元数据列。', '',
                 '没有统一的逐点飞行阶段真值列。mode 是飞控控制模式；传感器状态是诊断量；馆藏subject是记录级多标签；Sun phase是算法输出。几者不能互换。', '',
                 '## 常用字段速查', '',
                 '下表是典型G1含义；G2异步包使用gps./imu./ekf./air.等前缀，同步数据使用dtc.前缀。完整逐列说明在后面的全部字段表中。', '',
                 '| 字段 | 含义 | 典型G1单位 |', '|---|---|---|',
                 '| time / GPS_TOW | 记录时钟 / GPS周内时间，不能单独当完整UTC日期 | s |',
                 '| lat / lon / alt | GPS纬度、经度、WGS84高度 | deg / deg / m |',
                 '| vn / ve / vd | 北、东、向下速度；爬升时vd通常为负 | m/s |',
                 '| h / ias | 气压高度（参考面依版本）/ 指示空速 | m / m/s |',
                 '| Ps / Pd / Pd_aoa / Pd_aos | 静压 / 动压 / 迎角差压 / 侧滑差压 | kPa |',
                 '| p / q / r | 机体滚转、俯仰、偏航轴角速度 | rad/s |',
                 '| ax / ay / az | 机体三轴加速度 | m/s² |',
                 '| hx / hy / hz | 三轴磁场 | Gauss；G2不统一 |',
                 '| phi / the或theta / psi | 滚转、俯仰、偏航角；旧PDF与随附源码存在冲突 | rad；G2通常deg |',
                 '| navlat / navlon / navalt / navvn / navve / navvd | 导航滤波后的位置和速度 | 源码lat/lon为rad；m；m/s |',
                 '| de / da_l / da_r / dr / df_l / df_r / dthr | 升降舵 / 左右副翼 / 方向舵 / 左右襟翼 / 油门指令 | 舵角rad；油门0–1 |',
                 '| *_in / *_pos / *_cmd / *_bias | 输入 / 实测位置 / 指令 / 偏置估计 | 随对应物理量 |',
                 '| mode / run_num | 飞控模式 / 自动驾驶接通次数 | 枚举 / 计数 |',
                 '| imuStatus / gpsStatus / navStatus / adStatus | IMU / GPS / 导航 / 气压模块诊断 | 状态码或位域 |',
                 '| var1–var5 | 部分FDI试验的故障估计与自适应控制诊断 | 不能当统一故障真值 |', '',
                 '## 数据质量核验', '',
                 f'{len(quality_checks)} 个CSV的mode含非法值。以下逐条比较官方HDF5和同记录MAT；不修改现有数据。详情及前4个样本见JSON的mode_source_quality。', '',
                 '| 文件 | HDF5的mode/time/lat均与Pd完全相同 | 同记录MAT的mode全为0/1/2 |', '|---|---|---|',
                 *[f'| {c["file"]} | {any(s.get("mode_time_lat_all_equal_Pd", False) for s in c["source_checks"])} | {any(s.get("mode_valid_012", False) and s["source"].lower().endswith(".mat") for s in c["source_checks"])} |' for c in quality_checks], '',
                 '这些是源文件语义问题；校验和一致与CSV忠实转换不等于物理字段有效。非法状态值不能作为新类别或通过四舍五入修复。', '',
                 '## 飞控模式（随记录提供的三份FlightCode均有同一0/1/2定义）', '',
                 '| 值 | 符号/含义 |', '|---:|---|', '| 0 | dump data，数据导出/转储模式 |', '| 1 | manual control，人工控制 |', '| 2 | autopilot control，自动驾驶控制 |', '',
                 '## errdefs 状态枚举（不是飞行阶段）', '', '| 值 | 代码名 | 含义 |', '|---:|---|---|']
        for index, (name, meaning) in enumerate(ERRDEFS):
            lines.append(f'| {index} | {name} | {meaning} |')
        lines += ['', 'Baldr随附版本只定义0—7；Fenrir 16/17随附版本扩展到10。该表只用于对应固件的err_type映射，不能套到G2 status_flag、GPS fix_type或其他版本的专用状态列。', '',
                  '## adStatus 位域（AMS_pressure.c版本）', '', '| 位权 | 置位含义 |', '|---:|---|',
                  '| 1 | 静压 Ps 通道读取失败 |', '| 2 | 动压 Pd 通道读取失败 |', '| 4 | 迎角差压 Pd_aoa 通道读取失败 |', '| 8 | 侧滑差压 Pd_aos 通道读取失败 |', '',
                  '数值可按位组合；例如12=4+8。不能把12当作第12类飞行状态，且未安装相应传感器也会影响其解释。实际另见1024，已检查驱动未说明该位，不能强行套用。navValid在对应驱动中0=有效，非0=无效/需解码。', '',
                  '## 数据中实际出现的状态取值', '', '| 字段 | 涉及文件 | 非空记录数 | 实际取值（最多展示40项） |', '|---|---:|---:|---|']
        for name, stats in sorted(discrete.items()):
            values = sorted(stats['counts'], key=lambda x: float(x))
            integer_values = sorted(stats['integer_counts'], key=lambda x: int(x))
            display = ', '.join(integer_values[:40]) or '无整数值'
            if len(integer_values) > 40:
                display += ' …（完整整数取值见JSON integer_counts）'
            if stats['noninteger_count']:
                display += f'；另有{stats["noninteger_count"]:,}个非整数/非有限观测，不能直接视为枚举类别'
            if stats['values_capped']:
                display += '（非整数取值目录已截断）'
            lines.append(f'| {name} | {stats["files"]} | {stats["nonmissing"]:,} | {display} |')
        lines += ['', 'G2 gps.fix_type文档定义：0无定位、1航位推算、2二维、3三维；目前31份文件中该列仅见0，即使部分记录存在有效位置。不能仅据此列一刀切删除GPS点，需核验源导出/接收机协议。', '',
                  'dtc.pilot.SWT的源同步数据包含非整数，须先核验插值和开关端点定义，不能把每个小数当一种模式。status_flag虽仅见0，公共G2字典未完整给出状态码语义。', '',
                  '## 原始记录主题标签', '',
                  'subject可同时包含机型、地点、控制器、试验目的、机动名称。它们不是逐点标签，也不是最终互斥分组；Fault/Fault Injection不等于整条记录每个点异常。', '',
                  ', '.join(f'{name}（{count}）' for name, count in subjects.most_common()), '',
                  '## 单位与文档冲突', '',
                  '- G1典型GPS lat/lon是度；相同源码中的导航 navlat/navlon是弧度，不能混用。',
                  '- G1典型phi/theta或the/psi是滚转/俯仰/偏航（rad）；旧PDF把phi/psi的说明互换。G2 EKF姿态字段在对应units中是度。',
                  '- G1 ias是m/s；G2原始air.airspeed在文档中是kts；Baldr dtc.air.pitot.airspeed内嵌units为m/s。',
                  '- 旧PDF把加速度计偏置写为rad/s，源码为m/s²；以对应版本为准。',
                  '- h/alt_baro不能自动视为与GPS WGS84高度或ADS-B标准气压高度同一参考面。', '',
                  '## 全部字段', '', '| 字段 | 文件数 | 类别 | 含义 | 单位/范围 | 依据与限制 |', '|---|---:|---|---|---|---|']
        for field in fields:
            lines.append('| ' + ' | '.join(str(field[k]).replace('|', '/') for k in ('field', 'files', 'group', 'meaning', 'unit', 'evidence')) + ' |')
        lines += ['', '## 随数据提供的源代码证据', ''] + [f'- [{path}]({path})' for path in evidence]
        (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        if quality_checks:
            logger.warning('%d份CSV的mode含非法值，源HDF5/MAT核验结果见mode_source_quality；未修改数据', len(quality_checks))
        logger.info('字段审计完成：%d文件，%d列名，%d个状态字段；输出=%s', len(records), len(fields), len(discrete), out)
    except Exception:
        logger.exception('字段审计失败')
        raise
    finally:
        close_project_logging(logger)


if __name__ == '__main__':
    main()
