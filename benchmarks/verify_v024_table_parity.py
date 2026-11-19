#!/usr/bin/env python3
"""Verify frozen main tables and every numeric cell in the v0.2.4 audit tables.

Run from any directory. Only the Python standard library and aggregate ledgers are needed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.5 * 1e-4 + 1e-9

ADNI_MODELS = (
    'DeepFuzzyTorch', 'FUML', 'QMF', 'DrFuse', 'DDEF', 'XGBoost', 'CatBoost', 'TabPFN',
    'FUML_Faithful', 'RFUML_Faithful', 'QMF_Faithful', 'MUSE_BatchGraph',
    'XGBoost_ModDrop', 'CatBoost_ModDrop', 'TabPFN_ModDrop',
)
PATHOLOGY_REGIMES = (
    ('PLIP inversion', 'plip_invert_neg2x'),
    ('Single-expert inversion', 'single_fm_invert_macro'),
    ('Dual-expert noise', 'dual_fm_gauss_noise'),
    ('Clean OOD', 'clean_ood'),
    ('Temperature', 'temp_scale_miscalibration'),
    ('Dead expert', 'single_fm_dead_zero_macro'),
)
PATHOLOGY_METHODS = (
    'PFS_TTA_Full', 'Tent_TTA', 'SAR_TTA', 'EATA_TTA', 'Abl_LinearIFS_q1',
    'Abl_wo_DiscordanceReg', 'Abl_wo_SourceAnchor',
)
TCGA_MODELS = (
    ('B1a', 'B1a_all_cohorts', ('FuzzyRough-20', 'OWA-FuzzyRough-20', 'mRMR-20',
        'Lasso-20', 'MI-20', 'PCA-20', 'All-1000', 'CoxPH_Inductive20')),
    ('B1b', 'B1b_nonlung', ('FuzzyRough-20', 'OWA-FuzzyRough-20', 'mRMR-20',
        'Lasso-20', 'MI-20', 'All-1000', 'CoxPH_Inductive20')),
    ('B1c', 'B1c_within_cohort_z', ('FuzzyRough-20', 'OWA-FuzzyRough-20', 'mRMR-20',
        'Lasso-20', 'All-1000', 'CoxPH_Inductive20')),
)


def _tt(value: str) -> str:
    return r'\texttt{' + value.replace('_', r'\_') + '}'


def _table_rows(source: str, label: str, section: int, section_count: int) -> list[list[str]]:
    """Extract top-level ampersand-separated cells and row breaks from one table."""
    pattern = re.compile(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', re.S)
    tables = [match.group(0) for match in pattern.finditer(source)
              if re.search(r'\\label\{' + re.escape(label) + r'\}', match.group(0))]
    if len(tables) != 1:
        raise ValueError(f'{label}: expected one table, found {len(tables)}')
    segments = re.findall(r'\\toprule(.*?)\\bottomrule', tables[0], re.S)
    if len(segments) != section_count:
        raise ValueError(f'{label}: expected {section_count} tabular bodies, found {len(segments)}')
    segment = segments[section].replace(r'\midrule', '')
    rows: list[list[str]] = []
    cells: list[str] = []
    buf = ''
    depth = 0
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == '\\' and i + 1 < len(segment):
            nxt = segment[i + 1]
            if nxt == '\\' and depth == 0:
                cells.append(buf.strip())
                rows.append(cells)
                cells, buf = [], ''
                i += 2
                continue
            if nxt in '{}&%':
                buf += segment[i:i + 2]
                i += 2
                continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        if char == '&' and depth == 0:
            cells.append(buf.strip())
            buf = ''
        else:
            buf += char
        i += 1
    if depth != 0 or buf.strip() or cells:
        raise ValueError(f'{label}: incomplete or unbalanced table row')
    return rows


def _normal(text: str) -> str:
    return ' '.join(text.split())


def _check_table(source: str, label: str, header: list[str],
                 expected_rows: list[tuple[list[str], list[list[float]]]],
                 section: int = 0, section_count: int = 1) -> tuple[int, int]:
    rows = _table_rows(source, label, section, section_count)
    if not rows or [_normal(cell) for cell in rows[0]] != header:
        raise ValueError(f'{label}: unexpected column headings')
    if len(rows) - 1 != len(expected_rows):
        raise ValueError(f'{label}: expected {len(expected_rows)} rows, found {len(rows) - 1}')
    numeric_cells = 0
    numeric_values = 0
    number = re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])')
    for index, (row, (names, expected_cells)) in enumerate(zip(rows[1:], expected_rows), 1):
        if len(row) != len(names) + len(expected_cells):
            raise ValueError(f'{label}: row {index} has an unexpected number of cells')
        if [_normal(cell) for cell in row[:len(names)]] != names:
            raise ValueError(f'{label}: row {index} has unexpected labels: {row[:len(names)]}')
        for column, (cell, expected) in enumerate(zip(row[len(names):], expected_cells), len(names) + 1):
            values = number.findall(cell.replace('--', ' '))
            if len(values) != len(expected):
                raise ValueError(f'{label}: row {index}, column {column}: expected '
                                 f'{len(expected)} values, found {values}')
            if values:
                numeric_cells += 1
            for actual, target in zip(values, expected):
                if not re.fullmatch(r'\d+\.\d{4}', actual):
                    raise ValueError(f'{label}: row {index}, column {column}: '
                                     f'value {actual} is not formatted to four decimals')
                if abs(float(actual) - float(target)) > TOLERANCE:
                    raise ValueError(f'{label}: row {index}, column {column}: '
                                     f'{actual} differs from JSON value {target}')
                numeric_values += 1
    return numeric_cells, numeric_values


def verify_supplementary_tables(source: str, adni: dict, pathology: dict, tcga: dict) -> dict:
    """Validate row identities, all data cells and numeric values against the three ledgers."""
    adni_rows = []
    for name in ADNI_MODELS:
        full = adni['c1_metrics']['full'][name]
        adni_rows.append(([_tt(name)], [
            [full['auroc'], *full['confidence_intervals_95']['auroc']],
            [adni['c1_metrics']['same_visit_bl_924'][name]['auroc']],
            [adni['c1_15_subset_full'][name]['macro_15subset_auroc']],
            [adni['same_visit_bl_924_15_subset'][name]['macro_15subset_auroc']],
            [adni['c2_metrics']['full'][name]['auroc']],
            [adni['deduplicated_discordant_matched_coverage']['models_by_calibration']
                ['rank_inner'][name]['aurc']],
        ]))
    pathology_first = []
    pathology_second = []
    for display, regime in PATHOLOGY_REGIMES:
        for dataset in ('Camelyon17', 'NCT-CRC'):
            result = pathology['datasets'][dataset]['regimes'][regime]
            methods = result.get('methods', result.get('macro_average'))
            expected = [[methods[name]['auroc'], methods[name]['ece']] if name in methods else []
                        for name in PATHOLOGY_METHODS]
            pathology_first.append(([display, dataset], expected[:4]))
            pathology_second.append(([display, dataset], expected[4:]))
    tcga_rows = []
    for display, cohort, names in TCGA_MODELS:
        for name in names:
            result = tcga[cohort]['models'][name]
            stability = result.get('gene_stability_jaccard_5fold')
            mean = stability.get('mean') if isinstance(stability, dict) else None
            tcga_rows.append(([display, _tt(name)], [
                [result['auroc_4mod'], *result['auroc_4mod_ci_95']],
                [result['macro_cohort_auroc']], [result['os_c_index']],
                [result['within_cohort_os_c_index_mean']], [result['rna_only_observed_auroc']],
                [] if mean is None else [mean], [result['selector_fit_time_mean_seconds']],
            ]))
    specifications = (
        ('tab:supp_adni_fidelity_exposure', ['Model', r'Cohort~1 [95\% CI]', 'Same visit',
            'Subsets: full', 'Subsets: visit', r'Cohort~2', 'Inner AURC'], adni_rows),
        ('tab:supp_tcga_inductive_nonlung', ['Audit', 'Selector / model', r'AUROC [95\% CI]',
            'Macro', 'OS C-index', 'Within C-index', 'RNA only', 'Jaccard', 'Seconds / fold'], tcga_rows),
    )
    counts = {label: _check_table(source, label, header, rows)
              for label, header, rows in specifications}
    label = 'tab:supp_pathology_tta_ablation'
    first = _check_table(source, label, ['Shift', 'Dataset', 'PFS-TTA', 'Tent', 'SAR', 'EATA'],
                         pathology_first, section=0, section_count=2)
    second = _check_table(source, label, ['Shift', 'Dataset', 'Linear IFS', 'No discordance',
                                          'No anchor'], pathology_second, section=1, section_count=2)
    counts[label] = (first[0] + second[0], first[1] + second[1])
    return counts


def _parse_main_tables(code: str) -> list[list[list[str]]]:
    """Parse LaTeX tabular bodies matching the canonical parity extractor."""
    result: list[list[list[str]]] = []
    for segment in re.findall(r'\\toprule(.*?)\\bottomrule', code, re.S):
        rows: list[list[str]] = []
        cells: list[str] = []
        buf = ''
        depth = 0
        i = 0
        while i < len(segment):
            x = segment[i]
            if x == '\\' and i + 1 < len(segment):
                nxt = segment[i + 1]
                if nxt == '\\' and depth == 0:
                    cells.append(buf.strip())
                    rows.append(cells)
                    cells = []
                    buf = ''
                    i += 2
                    continue
                if nxt in '{}&%':
                    buf += segment[i:i + 2]
                    i += 2
                    continue
            if x == '{':
                depth += 1
            if x == '}':
                depth -= 1
            if x == '&' and depth == 0:
                cells.append(buf.strip())
                buf = ''
                i += 1
                continue
            buf += x
            i += 1
        result.append([r for r in rows if len(r) > 1])
    return result


def _cell_values(cell: str) -> list[str]:
    clean = cell.replace(',', '').replace('--', ' ')
    clean = re.sub(
        r'\\(?:textbf|texttt|shortstack|textit|cite|text|mathrm|mathbf|mathcal|bm|pm)\b',
        '',
        clean,
    )
    return re.findall(r'(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])[-+]?\d+(?![\w.])', clean)


def _fmt_ci(record: dict, key: str, ci_key: str) -> list:
    return [record[key], *record[ci_key]]


def verify_main_tables(main_source: str, benchmark: dict, runtime: dict) -> dict[str, int]:
    """Verify all frozen main-body tables against benchmark_results.json and runtime_vram_benchmark.json."""
    tables = {
        re.search(r'\\label\{tab:([^}]+)\}', chunk).group(1): chunk
        for chunk in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', main_source, re.S)
        if re.search(r'\\label\{tab:([^}]+)\}', chunk)
    }
    counts: dict[str, int] = {}
    fails: list[tuple] = []
    rows_done: dict[str, int] = {}

    def check(tab: str, row_name: str, col_idx: int | str, cell: str, expected: object) -> None:
        counts[tab] = counts.get(tab, 0) + 1
        toks = _cell_values(cell)
        if expected is None:
            exp_list: list = []
        elif isinstance(expected, list):
            exp_list = [x for x in expected if x is not None]
        else:
            exp_list = [expected]
        if len(toks) != len(exp_list):
            fails.append((tab, row_name, col_idx, toks, exp_list, 'length'))
            return
        for tok, target in zip(toks, exp_list):
            places = len(tok.split('.')[1]) if '.' in tok else 0
            if abs(float(tok) - float(target)) > 0.5 * (10 ** (-places)) + 1e-9:
                fails.append((tab, row_name, col_idx, toks, exp_list, 'value'))
                return

    def runrow(
        tab: str,
        row_name: str,
        row: list[str],
        exps: list[object],
        label_tokens: Sequence[str] | tuple[str, ...] = (),
    ) -> None:
        if len(row) - 1 != len(exps):
            raise ValueError(f'{tab} row {row_name}: expected {len(exps)} cells, found {len(row) - 1}')
        row_label = _normal(row[0].replace('\\midrule', ''))
        for tok in label_tokens:
            if tok not in row_label:
                raise ValueError(f'{tab} row {row_name}: expected label token {tok!r} in {row_label!r}')
        rows_done[tab] = rows_done.get(tab, 0) + 1
        for col_idx, (cell, expected) in enumerate(zip(row[1:], exps), 1):
            check(tab, row_name, col_idx, cell, expected)

    main = benchmark['track1_adni_cohort1']
    rows = _parse_main_tables(tables['adni_main'])[0][1:]
    c1_names = [
        'EarlyConcat', 'LateAvg', 'DeepTSK', 'DempsterShafer', 'GatedAttention', 'EvidentialDL',
        'RCML', 'QMF', 'DrFuse', 'DDEF', 'FUML', 'XGBoost', 'CatBoost', 'TabPFN', 'DeepFuzzyTorch',
        'EarlyConcat_EqLoss', 'DempsterShafer_EqLoss', 'GatedAttention_EqLoss',
        'DeepFuzzyTorch_ColdNoAux', 'DeepFuzzyTorch',
    ]
    c1_tokens = [
        ('EarlyConcat-MLP',), ('LateAvg-Ensemble',), ('DeepTSK-RuleNet',), ('Dempster--Shafer',),
        ('GatedAttention',), ('EvidentialDL',), ('RCML',), ('QMF',), ('DrFuse',), ('DDEF',),
        ('FUML',), ('XGBoost',), ('CatBoost',), ('TabPFN',), ('DeepFuzzyTorch', 'Ours'),
        ('EarlyConcat\\_EqLoss',), ('DempsterShafer\\_EqLoss',), ('GatedAttention\\_EqLoss',),
        ('DeepFuzzyTorch\\_ColdNoAux',), ('DeepFuzzyTorch', 'Standard'),
    ]
    for idx, (row, name, ltoks) in enumerate(zip(rows, c1_names, c1_tokens)):
        iseq = idx >= 15
        d = (main['equalized_supervision_ablation'] if iseq else main['main_table'])[name]
        sub = [d['macro_15subset_auroc']] if iseq else [
            d['macro_15subset_auroc'],
            d['macro_15subset_auroc_eligible'],
        ]
        ex: list[object] = [
            _fmt_ci(d, 'auroc', 'auroc_ci'),
            None if iseq else main['same_visit_bl_924_table'][name]['auroc'],
            None if iseq else main['complete_4mod_854_table'][name]['auroc'],
            d['auprc'], d['acc'], d['f1'], d['brier'], d['ece'], d['aurc'],
            sub,
            d['mrr_14incomplete'],
            None if name == 'DeepFuzzyTorch' else d['p_value_vs_ours'],
            None if iseq or name == 'DeepFuzzyTorch' else d['q_bh_vs_ours'],
        ]
        runrow('adni_main', name, row, ex, ltoks)

    rows = _parse_main_tables(tables['ablation'])[0][1:]
    abl_specs = [
        ('DeepFuzzyTorch', ('DeepFuzzyTorch', 'Full Model')),
        ('Ablation_wo_VacuityMask', ('w/o Vacuity Masking',)),
        ('Ablation_wo_Choquet', ('w/o Non-Additive Choquet',)),
        ('Ablation_wo_ConflictDiscount', ('w/o Conflict Hesitancy Discount',)),
        ('Ablation_wo_FuzzyRough', ('w/o Fuzzy-Rough Gate',)),
        ('Ablation_wo_FuzzyRough_Unreduced320', ('320 ROIs',)),
        ('Ablation_wo_TSKHead', ('w/o White-Box Concept TSK Rule Head',)),
    ]
    for row, (name, ltoks) in zip(rows, abl_specs):
        d = main['main_table'][name]
        diff = d['auroc'] - main['main_table']['DeepFuzzyTorch']['auroc']
        runrow('ablation', name, row, [
            _fmt_ci(d, 'auroc', 'auroc_ci'),
            d['ece'], d['aurc'], d['macro_15subset_auroc'], d['mrr_14incomplete'],
            d['referral_20pct_auroc_g08'], None if name == 'DeepFuzzyTorch' else diff,
        ], ltoks)

    rows = _parse_main_tables(tables['missing_15'])[0][1:]
    modelnames = [
        'EarlyConcat', 'GatedAttention', 'EvidentialDL', 'RCML',
        'QMF', 'DrFuse', 'DDEF', 'FUML', 'DeepFuzzyTorch',
    ]
    subsets = [
        'MRI', 'Plasma', 'FDG', 'Clin', 'MRI+Plasma', 'MRI+FDG', 'MRI+Clin',
        'Plasma+FDG', 'Plasma+Clin', 'FDG+Clin', 'MRI+Plasma+FDG', 'MRI+Plasma+Clin',
        'MRI+FDG+Clin', 'Plasma+FDG+Clin', 'MRI+Plasma+FDG+Clin',
    ]
    for row, subset in zip(rows, subsets):
        d = main['combinatorial_15subsets']['DeepFuzzyTorch']['subsets'][subset]
        ex_sub: list[object] = [d['num_modalities'], d['n_eligible'], d['n_complete_subset']]
        for m_name in modelnames:
            z = main['combinatorial_15subsets'][m_name]['subsets'][subset]
            ex_sub.append([z['auroc'], z['auroc_eligible']])
        ex_sub.append(d['mean_hesitancy'])
        runrow('missing_15', subset, row, ex_sub, (r'\texttt{' + subset + '}',))
    for row, (typ, ltoks) in zip(rows[-2:], [('macro', ('Macro-Average',)), ('mrr', ('Incomplete MRR',))]):
        ex_agg: list[object] = [[1, 4] if typ == 'macro' else [1, 3], None, None]
        for m_name in modelnames:
            d = main['combinatorial_15subsets'][m_name]
            ex_agg.append(
                [d['macro_avg_auroc'], d['macro_avg_auroc_eligible']]
                if typ == 'macro'
                else [d['missing_robustness_retention'], d['missing_robustness_retention_eligible']]
            )
        ex_agg.append(None)
        runrow('missing_15', typ, row, ex_agg, ltoks)

    rows = _parse_main_tables(tables['external_ckd'])[0][1:]
    ext_specs = [
        ('EarlyConcat', ('EarlyConcat-MLP',)),
        ('GatedAttention', ('GatedAttention',)),
        ('EvidentialDL', ('EvidentialDL',)),
        ('RCML', ('RCML',)),
        ('QMF', ('QMF',)),
        ('DrFuse', ('DrFuse',)),
        ('DDEF', ('DDEF',)),
        ('FUML', ('FUML',)),
        ('XGBoost', ('XGBoost',)),
        ('CatBoost', ('CatBoost',)),
        ('TabPFN', ('TabPFN',)),
        ('DeepFuzzyTorch', ('DeepFuzzyTorch', 'Ours')),
    ]
    t2 = benchmark['track2_adni_cohort2_ptau217']
    t3 = benchmark['track3_oasis3_external']
    disc = t2['real_world_discordance_audit']
    ckd = t2['ckd_subgroup']['symmetric_composite_triage']
    for row, (name, ltoks) in zip(rows, ext_specs):
        if name in t3['adni_longitudinal_table']:
            lon = t3['adni_longitudinal_table'][name]
            oas = t3['oasis3_external_table'][name]
            ex_ext: list[object] = [
                _fmt_ci(lon, 'auroc_60m', 'auroc_60m_ci'),
                lon['c_index'],
                [oas['auroc_5seed_mean'], oas['auroc_5seed_std']],
                [oas['cdr05_nommse_auroc'], oas['cdr05_with_mmse_auroc']],
            ]
        else:
            ex_ext = [None] * 4
        d = disc['dedup_discordant_triage'][name]
        ex_ext.append([d['acc_refer20'], d['k_correct_refer20'], d['n_refer20_retained']])
        ex_ext.append([t2['cohort2_metrics'][name]['auroc'], t2['cohort2_strict_90d_metrics'][name]['auroc']])
        if name in ckd:
            ck = ckd[name]
            k_val, n_val = map(int, ck['ckd_post_triage_k_n'].split('/'))
            ex_ext.append([k_val / n_val, k_val, n_val])
        else:
            ex_ext.append(None)
        runrow('external_ckd', name, row, ex_ext, ltoks)

    rows = _parse_main_tables(tables['matched_coverage'])[0][1:]
    mc_specs = [
        ('EarlyConcat', ('EarlyConcat-MLP',)),
        ('GatedAttention', ('GatedAttention',)),
        ('EvidentialDL', ('EvidentialDL',)),
        ('RCML', ('RCML',)),
        ('QMF', ('QMF',)),
        ('DrFuse', ('DrFuse',)),
        ('DDEF', ('DDEF',)),
        ('FUML', ('FUML',)),
        ('XGBoost', ('XGBoost',)),
        ('CatBoost', ('CatBoost',)),
        ('TabPFN', ('TabPFN',)),
        ('DeepFuzzyTorch', ('DeepFuzzyTorch', 'Comp.')),
        ('pure_conflict', ('DeepFuzzyTorch', 'Pure')),
    ]
    mc = disc['matched_coverage_referral']
    cov_keys = ['cov_100pct', 'match_161_81p3pct', 'cov_80pct', 'match_152_76p8pct', 'match_140_70p7pct', 'cov_60pct']
    for row, (name, ltoks) in zip(rows, mc_specs):
        mkey = 'DeepFuzzyTorch' if name == 'pure_conflict' else name
        z = mc['conflict_only_nested_ecdf' if name == 'pure_conflict' else 'composite_nested_ecdf']['models'][mkey]
        if name == 'pure_conflict':
            raw: object = None
        else:
            d = disc['dedup_discordant_triage'][name]
            raw = [d['acc_refer20'], d['k_correct_refer20'], d['n_refer20_retained'], 100 * d['n_refer20_retained'] / 198]
        p_val = None if name in ('DeepFuzzyTorch', 'pure_conflict') else z['aurc_p_vs_ours']
        runrow('matched_coverage', name, row, [raw, [z['aurc'], p_val]] + [z['points'][k]['acc'] for k in cov_keys], ltoks)

    t4 = benchmark['track4_tcga_pancancer']
    hd = t4['highdim_1000gene_reduct']
    rows = _parse_main_tables(tables['tcga_fm'])[0][1:]
    t4_3mod = [
        ('EarlyConcat', ('EarlyConcat-MLP', '3-Mod')),
        ('GatedAttention', ('GatedAttention', '3-Mod')),
        ('EvidentialDL', ('EvidentialDL', '3-Mod')),
        ('RCML', ('RCML', '3-Mod')),
        ('QMF', ('QMF', '3-Mod')),
        ('DrFuse', ('DrFuse', '3-Mod')),
        ('DeepFuzzyTorch', ('DeepFuzzyTorch', '3-Mod')),
    ]
    for row, (name, ltoks) in zip(rows[:7], t4_3mod):
        d = t4['tcga_main_table'][name]
        ex_t4: list[object] = [
            _fmt_ci(d, 'auroc', 'auroc_ci'),
            d['os_c_index'],
            d['macro_7subset_auroc'],
        ] + [t4['tcga_per_cohort']['TCGA-' + ch][name] for ch in ('BRCA', 'COAD', 'KIRC', 'KIRP', 'LUAD', 'LUSC', 'UCEC')]
        runrow('tcga_fm', name + '_3mod', row, ex_t4, ltoks)
    hd_specs = [
        ('DeepFuzzyTorch_4Mod_All1000', ('DeepFuzzyTorch', 'All-1000')),
        ('EarlyConcat_4Mod_All1000', ('EarlyConcat-MLP', 'All-1000')),
        ('GatedAttention_4Mod_Top20', ('GatedAttention', 'K^*=20')),
        ('EarlyConcat_4Mod_Top20', ('EarlyConcat-MLP', 'K^*=20')),
        ('QMF_4Mod_Top20', ('QMF', 'K^*=20')),
        ('DrFuse_4Mod_Top20', ('DrFuse', 'K^*=20')),
        ('RCML_4Mod_Top20', ('RCML', 'K^*=20')),
        ('DempsterShafer_4Mod_Top20', ('DempsterShafer', 'K^*=20')),
        ('EvidentialDL_4Mod_Top20', ('EvidentialDL', 'K^*=20')),
    ]
    for row, (name, ltoks) in zip(rows[7:16], hd_specs):
        d = hd['multimodal_4mod_table'][name]
        runrow('tcga_fm', name, row, [_fmt_ci(d, 'auroc', 'auroc_ci'), d['os_c_index']] + [None] * 8, ltoks)
    sel_specs = [
        ('mRMR-20', ('mRMR-20',)),
        ('Lasso-20', ('Lasso-20',)),
        ('FuzzyRough-20', ('FuzzyRough-20',)),
        ('MI-20', ('MI-20',)),
        ('PCA-20', ('PCA-20',)),
    ]
    for row, (name, ltoks) in zip(rows[16:], sel_specs):
        d = hd['feature_selector_comparison'][name]
        runrow('tcga_fm', name, row, [_fmt_ci(d, 'auroc_4mod', 'auroc_4mod_ci'), d['os_c_index_4mod']] + [None] * 8, ltoks)
        if name != 'PCA-20':
            match = re.search(r'Jacc\.[^\d]*(0\.\d+)', row[0])
            if match is None:
                raise ValueError(f'Missing Jaccard label in tcga_fm row {name}')
            check('tcga_selector_jaccard_labels', name, 'label', match.group(1), d['jaccard_5fold']['mean'])

    rows = _parse_main_tables(tables['tcga_fm'])[1][1:]
    t5_tokens = [('Camelyon17', 'Clean'), ('Camelyon17', 'Corrupted'), ('NCT-CRC', 'Clean'), ('NCT-CRC', 'Corrupted')]
    for row, (name, d), ltoks in zip(rows, benchmark['track5_multi_fm_consensus_tta'].items(), t5_tokens):
        ex_t5: list[object] = [
            d[k] for k in ('BestSingleFM', 'MeanPool6FMs', 'MedianPool6FMs', 'Tent_TTA', 'HesitantFuzzy')
        ] + [[d['HesitantFuzzy_PFS_TTA'], *d['HesitantFuzzy_PFS_TTA_ci']]]
        runrow('tcga_pathology', name.replace('\n', ' '), row, ex_t5, ltoks)

    rows = _parse_main_tables(tables['runtime'])[0][1:]
    rt_specs = [
        ('EarlyConcat', ('EarlyConcat-MLP',)),
        ('EvidentialDL', ('EvidentialDL',)),
        ('RCML', ('RCML',)),
        ('QMF', ('QMF',)),
        ('GatedAttention', ('GatedAttention',)),
        ('DrFuse', ('DrFuse',)),
        ('UnifiedDeepFuzzyModel', ('DeepFuzzyTorch',)),
    ]
    for row, (name, ltoks) in zip(rows, rt_specs):
        def pick(key: str, **kw: int) -> dict:
            for z in runtime[key]:
                if z['model'] == name and all(z[k] == v for k, v in kw.items()):
                    return z
            raise ValueError(f'Missing runtime entry {key} for {name} with {kw}')
        n256 = pick('sweep_N_D100_M4', N=256)
        n1024 = pick('sweep_N_D100_M4', N=1024)
        m16 = pick('sweep_M_N256_D100', M_modalities=16)
        runrow('runtime', name, row, [
            n256['n_params'], n256['fwd_bwd_mean_ms'], n256['infer_mean_ms'],
            n256['peak_memory_mib'], n1024['infer_mean_ms'], m16['infer_mean_ms'],
        ], ltoks)

    if fails:
        raise ValueError(f'Main table parity failures ({len(fails)}): {fails[:5]}')

    expected_rows = {
        'adni_main': (20,), 'ablation': (7,), 'missing_15': (17,),
        'external_ckd': (12,), 'matched_coverage': (13,), 'tcga_fm': (21, 4), 'runtime': (7,),
    }
    main_only = main_source.split(r'\appendix', 1)[0]
    main_labels = re.findall(
        r'\\label\{(tab:[^}]+)\}',
        '\n'.join(re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', main_only, re.S)),
    )
    expected_labels = [
        'tab:paradigm_comparison', 'tab:adni_main', 'tab:ablation',
        'tab:missing_15', 'tab:external_ckd', 'tab:matched_coverage', 'tab:tcga_fm', 'tab:runtime',
    ]
    if main_labels != expected_labels:
        raise ValueError(f'Unexpected main table labels or order: {main_labels}')
    for label, row_counts in expected_rows.items():
        actual = tuple(len(rows_seg) - 1 for rows_seg in _parse_main_tables(tables[label]))
        if actual != row_counts:
            raise ValueError(f'Main table {label}: expected row counts {row_counts}, found {actual}')
    return counts


def main() -> None:
    def load(name: str) -> dict:
        return json.loads((ROOT / 'results' / name).read_text(encoding='utf-8'))

    main_source = (ROOT / 'paper/manuscript_Information_Fusion.tex').read_text(encoding='utf-8')
    main_counts = verify_main_tables(
        main_source,
        load('benchmark_results.json'),
        load('runtime_vram_benchmark.json'),
    )
    counts = verify_supplementary_tables(
        main_source,
        load('v024_adni_fusion_exposure_ecdf_audit.json'),
        load('v024_pathology_tta_ablation_audit.json'),
        load('v024_tcga_inductive_nonlung_owa_audit.json'),
    )
    print(f'Main tables verified ({sum(main_counts.values())} cells, 0 failures)')
    for label, (cells, values) in counts.items():
        print(f'{label}: {cells} numeric cells, {values} values verified')
    print('v0.2.4 parity: 0 failures')


if __name__ == '__main__':
    main()
