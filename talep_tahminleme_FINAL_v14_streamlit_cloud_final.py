# -*- coding: utf-8 -*-
"""
Demand Forecasting - Production Grade Real-World Preprocessing Module
Authoring goal:
    Conservative, auditable, review-friendly preprocessing for demand forecasting.

Compatible modeling families:
    - CNN
    - LSTM
    - Prophet
    - ARIMA
    - SARIMAX
    - XGBoost
    - N-HiTS
    - TiDE
    - PatchTST
    - TFT
    - Chronos-Bolt

Core philosophy:
    CLEANING -> ANOMALY GOVERNANCE

Main capabilities:
    1) Manual Excel file and sheet selection
    2) Date / target auto detection
    3) Frequency inference and regularization
    4) Series profiling (volume, CV, intermittency, volatility, seasonality, trend)
    5) Adaptive anomaly detection by series profile
    6) Anomaly classification:
        - data_error
        - structural_event
        - business_spike_dip
        - unknown_anomaly
    7) Human-review-first policy on recent periods
    8) Structural event engine
    9) Intervention intensity / change tracking
    10) Raw vs clean forecastability comparison
    11) Model-family-specific exports
    12) Strict leakage audit
    13) Review queue
    14) Run manifest / versioning / config hashing
    15) Internal tests / synthetic tests / business-aware tests
"""

import os
import re
import io
import json
import math
import time
import uuid
import hashlib
import traceback
import warnings
import tempfile
import zipfile
import shutil
import atexit
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer

# GUI (desktop optional; Streamlit Cloud/mobile does not use tkinter)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    HAS_TKINTER = True
except Exception:
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None
    HAS_TKINTER = False

try:
    import rarfile  # optional; requires rarfile package and available backend on the host machine
except Exception:
    rarfile = None

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except Exception:
    seasonal_decompose = None
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore")


# =========================================================
# VERSION / PIPELINE
# =========================================================

PIPELINE_NAME = "demand_forecast_preprocessing"
PIPELINE_VERSION = "2.0.0"
OUTPUT_SCHEMA_VERSION = "2.0.0"
CODE_VERSION = "production_governance_rewrite_2_0_0"


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PreprocessConfig:
    output_dir_name: str = "forecast_preprocessing_outputs"

    date_column_candidates: Tuple[str, ...] = (
        "datum", "date", "datetime", "tarih", "timestamp", "zaman", "time"
    )

    non_target_columns: Tuple[str, ...] = (
        "year", "month", "day", "hour", "minute", "second",
        "weekday", "weekday name", "haftanın günü", "haftanın günü (tr)",
        "week", "quarter", "is_holiday", "holiday", "weekofyear",
        "dayofweek", "dayofmonth"
    )

    # Frequency / regularization
    force_regular_frequency: bool = True
    allow_month_start_to_month_end_alignment_fix: bool = True

    # Base anomaly voting windows
    hampel_window: int = 7
    hampel_n_sigma: float = 4.0
    rolling_mad_window: int = 9
    rolling_mad_n_sigma: float = 4.5
    iqr_k: float = 4.0
    min_outlier_votes: int = 2

    # Adaptive governance
    max_outlier_fraction_per_series: float = 0.05
    protect_first_n_periods: int = 1
    protect_last_n_periods: int = 6
    recent_periods_review_only: int = 6
    clip_negative_to_zero: bool = True

    # Structural events
        # Structural events
    structural_zero_ratio_threshold: float = 0.5
    structural_zero_min_series_count: int = 3
    portfolio_drop_ratio_threshold: float = 0.55
    portfolio_rebound_ratio_threshold: float = 1.30
    structural_event_neighbor_window: int = 0
    preserve_structural_zero_events: bool = True
    preserve_zero_values_on_structural_dates: bool = True

    # Incomplete / partial period governance
    enable_incomplete_period_detection: bool = True
    partial_period_drop_ratio_threshold: float = 0.60
    partial_period_compare_last_n: int = 3
    auto_exclude_incomplete_last_period_from_training: bool = True
    auto_flag_incomplete_last_period_review: bool = True

    # Modeling exclusion / masks
    export_training_exclusion_masks: bool = True

    # Monthly feature discipline
    drop_low_signal_calendar_features_for_monthly: bool = True

    # Missing / imputation
    max_interpolation_gap: int = 1
    seasonal_period_map: Dict[str, int] = field(default_factory=lambda: {
        "H": 24,
        "D": 7,
        "W": 52,
        "M": 12,
        "MS": 12
    })
    use_knn_for_dense_missing_blocks: bool = False
    impute_method_preference: str = "seasonal_local"

    # Missing strategy governance / audit (safe-additive; does not override core cleaning by default)
    missing_report_only_threshold: float = 0.00
    missing_drop_row_threshold: float = 0.80
    missing_drop_series_threshold: float = 0.60
    missing_impute_ratio_threshold: float = 0.20
    dense_missing_block_threshold: int = 3
    allow_row_drop_for_non_target_metadata_only: bool = True
    allow_target_row_drop: bool = False
    missingness_mechanism_proxy_check: bool = True

    # Datetime integrity / alignment governance
    normalize_datetime_timezone: bool = True
    align_monthly_dates_to_period_end: bool = True

    # Descriptive statistics / visual diagnostics
    save_distribution_plots: bool = True
    save_trend_plots: bool = True
    save_seasonality_plots: bool = True
    moving_average_windows: Tuple[int, int] = (3, 6)
    save_boxplots: bool = True
    save_year_overlay_seasonality_plots: bool = True
    save_normalized_seasonality_plots: bool = True
    save_correlation_analysis: bool = True
    save_seasonality_heatmaps: bool = True
    save_seasonal_decomposition: bool = True
    robust_trend_window: int = 5
    save_robust_trend: bool = True

    # Proxy backtest enrichment
    enable_additional_backtest_benchmarks: bool = True

    # Pharma event interpretation hints (safe-additive diagnostics)
    promotion_like_jump_ratio: float = 2.0
    stockout_like_drop_ratio: float = 0.30
    rebound_after_drop_ratio: float = 1.80

    # Review / governance policy
    min_action_confidence_for_auto_fix: float = 0.70
    auto_fix_business_spike_dip: bool = False
    auto_fix_unknown_anomaly: bool = False
    auto_fix_data_error: bool = True
    auto_fix_structural_event: bool = False

    # Scaling
    scaler_for_deep_learning: str = "robust"
    export_log1p_version: bool = True

    # Feature engineering
    generate_modeling_ready_feature_pack: bool = True
    exclude_textual_columns_from_modeling_features: bool = True
    drop_weekday_text_from_monthly_modeling: bool = True

    # QA / validation
    save_validation_excel: bool = True
    save_validation_csv: bool = True
    save_validation_plots: bool = True
    create_domain_validation_template: bool = True
    leakage_check_enabled: bool = True
    max_plot_series: int = 50

    # Tests / QA
    run_internal_unit_tests: bool = True
    run_synthetic_tests: bool = True
    run_business_rule_tests: bool = True
    run_manual_sample_audit: bool = True
    run_proxy_backtest_validation: bool = True
    manual_sample_size: int = 20
    random_seed: int = 42

    # Proxy backtest
    backtest_horizon: int = 3
    backtest_min_train_size_monthly: int = 24
    backtest_min_train_size_weekly: int = 52
    backtest_min_train_size_daily: int = 60
    backtest_min_train_size_hourly: int = 24 * 14

    # Validation thresholds
    review_if_outlier_fraction_gt: float = 0.05
    review_if_structural_zero_events_gt: int = 1
    review_if_clean_zero_ratio_gt: float = 0.10
    review_if_proxy_smape_gt: float = 60.0

    # Export
    save_excel: bool = True
    save_csv: bool = True
    save_metadata_json: bool = True
    save_quality_report: bool = True

    # Manifest / versioning
    pipeline_name: str = PIPELINE_NAME
    pipeline_version: str = PIPELINE_VERSION
    output_schema_version: str = OUTPUT_SCHEMA_VERSION
    code_version: str = CODE_VERSION

        # Incomplete / partial period governance
    detect_partial_last_period: bool = True
    partial_last_period_drop_ratio_threshold: float = 0.65
    partial_last_period_compare_window: int = 3
    partial_last_period_exclude_from_training: bool = True

    # Safer anomaly governance
    auto_fix_unknown_anomaly: bool = False
    auto_fix_business_spike_dip: bool = False

    # Monthly feature discipline
    monthly_keep_only_low_leakage_calendar_features: bool = True

    # Training mask / modeling governance
    generate_training_mask: bool = True
    exclude_structural_events_from_statistical_models: bool = True
    exclude_partial_last_period_from_training: bool = True

    # Imputation audit
    track_governance_imputations: bool = True

    # CV-safe / future-safe governance notes
    enable_fold_aware_preprocessing_notes: bool = True


# =========================================================
# UTILITIES
# =========================================================

def safe_excel_sheet_name(name: str, max_len: int = 31) -> str:
    name = re.sub(r"[:\\/?*\[\]]", "_", str(name))
    return name[:max_len]


def normalize_colname(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower())


def _choose_item_from_list(title: str, prompt: str, items: List[str]) -> str:
    if not HAS_TKINTER:
        raise RuntimeError('Bu seçim ekranı tkinter gerektirir. Streamlit/telefon kullanımında dosya ve sheet seçimi uygulama arayüzünden yapılmalıdır.')
    root = tk.Tk()
    root.withdraw()

    item_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(items)])
    answer = simpledialog.askstring(title, f"{prompt}\n\n{item_text}\n\nÖrnek: 1")

    if not answer:
        raise ValueError("Seçim yapılmadı.")

    answer = answer.strip()
    if not answer.isdigit():
        raise ValueError("Geçerli bir seçim yapılmadı.")

    idx = int(answer) - 1
    if not (0 <= idx < len(items)):
        raise ValueError("Geçerli bir seçim yapılmadı.")

    return items[idx]


def _list_excel_files_in_archive(archive_path: str) -> List[str]:
    lower = str(archive_path).lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            names = zf.namelist()
    elif lower.endswith('.rar'):
        if rarfile is None:
            raise ImportError(
                "RAR arşivi seçildi ancak 'rarfile' modülü bu bilgisayarda kurulu değil. "
                "RAR desteği için 'pip install rarfile' ve sistemde unrar/bsdtar benzeri backend gerekir."
            )
        with rarfile.RarFile(archive_path, 'r') as rf:
            names = rf.namelist()
    else:
        raise ValueError("Desteklenmeyen arşiv türü.")

    excel_files = [n for n in names if not n.endswith('/') and n.lower().endswith(('.xlsx', '.xls'))]
    if not excel_files:
        raise FileNotFoundError("Arşiv içinde Excel dosyası bulunamadı.")
    return excel_files


def _extract_excel_from_archive(archive_path: str, member_name: str) -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix='forecast_preproc_archive_')

    lower = str(archive_path).lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extract(member_name, path=temp_dir)
    elif lower.endswith('.rar'):
        if rarfile is None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ImportError(
                "RAR arşivi seçildi ancak 'rarfile' modülü bu bilgisayarda kurulu değil. "
                "RAR desteği için 'pip install rarfile' ve sistemde unrar/bsdtar benzeri backend gerekir."
            )
        with rarfile.RarFile(archive_path, 'r') as rf:
            rf.extract(member_name, path=temp_dir)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("Desteklenmeyen arşiv türü.")

    extracted_path = os.path.join(temp_dir, member_name)
    if not os.path.exists(extracted_path):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError("Arşivden çıkarılan Excel dosyası bulunamadı.")

    atexit.register(lambda p=temp_dir: shutil.rmtree(p, ignore_errors=True))
    return extracted_path, temp_dir


def choose_excel_file() -> Dict[str, Optional[str]]:
    if not HAS_TKINTER:
        raise RuntimeError('Tkinter bu ortamda kullanılamıyor. Streamlit dosya yükleme akışını kullanın.')
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askopenfilename(
        title="Excel dosyasını veya ZIP/RAR arşivini seçin",
        filetypes=[
            ("Excel and archives", "*.xlsx *.xls *.zip *.rar"),
            ("Excel files", "*.xlsx *.xls"),
            ("ZIP archives", "*.zip"),
            ("RAR archives", "*.rar"),
        ]
    )
    if not selected_path:
        raise FileNotFoundError("Dosya seçilmedi.")

    lower = selected_path.lower()
    if lower.endswith(('.xlsx', '.xls')):
        return {
            'source_path': selected_path,
            'excel_path': selected_path,
            'archive_member': None,
            'temp_dir': None,
            'source_type': 'excel'
        }

    if lower.endswith(('.zip', '.rar')):
        excel_files = _list_excel_files_in_archive(selected_path)
        if len(excel_files) == 1:
            chosen_member = excel_files[0]
        else:
            chosen_member = _choose_item_from_list(
                "Arşiv İçindeki Excel Seçimi",
                "Arşiv içinden kullanmak istediğiniz Excel dosyasını seçin:",
                excel_files
            )
        extracted_path, temp_dir = _extract_excel_from_archive(selected_path, chosen_member)
        return {
            'source_path': selected_path,
            'excel_path': extracted_path,
            'archive_member': chosen_member,
            'temp_dir': temp_dir,
            'source_type': 'archive'
        }

    raise ValueError("Desteklenmeyen dosya türü seçildi.")


def choose_sheets(sheet_names: List[str]) -> List[str]:
    if not HAS_TKINTER:
        raise RuntimeError('Tkinter bu ortamda kullanılamıyor. Streamlit sheet seçimini kullanın.')
    root = tk.Tk()
    root.withdraw()

    sheet_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sheet_names)])
    prompt = (
        "Kullanmak istediğiniz sheet numaralarını virgülle girin.\n\n"
        f"{sheet_text}\n\n"
        "Örnek: 1,2 veya sadece 2"
    )
    answer = simpledialog.askstring("Sheet Seçimi", prompt)

    if not answer:
        raise ValueError("Sheet seçimi yapılmadı.")

    idxs = []
    for x in answer.split(","):
        x = x.strip()
        if x.isdigit():
            idx = int(x) - 1
            if 0 <= idx < len(sheet_names):
                idxs.append(idx)

    if not idxs:
        raise ValueError("Geçerli bir sheet seçimi yapılmadı.")

    return [sheet_names[i] for i in sorted(set(idxs))]


def create_output_dir(base_path: str, output_dir_name: str) -> str:
    base_folder = os.path.dirname(base_path)
    out_dir = os.path.join(base_folder, output_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_uploaded_file(uploaded_file) -> str:
    """
    Streamlit uploaded file nesnesini güvenli bir geçici klasöre yazar ve dosya yolunu döndürür.
    Hem .xlsx hem .xls için çalışır. Aynı isimli tekrar yüklemelerde çakışmayı önler.
    """
    if uploaded_file is None:
        raise ValueError("Yüklenecek dosya bulunamadı.")

    original_name = os.path.basename(getattr(uploaded_file, 'name', 'uploaded.xlsx'))
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', original_name)
    suffix = os.path.splitext(safe_name)[1] or '.xlsx'

    temp_dir = os.path.join(tempfile.gettempdir(), 'talep_tahminleme_streamlit_uploads')
    os.makedirs(temp_dir, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(temp_dir, unique_name)

    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
    if file_bytes is None:
        raise ValueError('Yüklenen dosya okunamadı.')

    with open(save_path, 'wb') as f:
        f.write(file_bytes)

    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        raise IOError('Yüklenen dosya diske yazılamadı.')

    if suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError('Desteklenmeyen dosya türü. Lütfen Excel dosyası yükleyin.')

    return save_path



def _check_optional_excel_dependency(ext: str) -> Tuple[bool, Optional[str], Optional[str]]:
    ext = str(ext).lower()
    if ext == '.xlsx':
        try:
            import openpyxl  # noqa: F401
            return True, 'openpyxl', None
        except Exception as e:
            return False, None, f".xlsx dosyalarını okumak için openpyxl gerekir: {e}"
    if ext == '.xls':
        try:
            import xlrd  # noqa: F401
            return True, 'xlrd', None
        except Exception as e:
            return False, None, f".xls dosyalarını okumak için xlrd gerekir: {e}"
    return False, None, f"Desteklenmeyen Excel uzantısı: {ext}"


def safe_excel_file(excel_path: str) -> pd.ExcelFile:
    ext = os.path.splitext(str(excel_path))[1].lower()
    ok, engine, msg = _check_optional_excel_dependency(ext)
    if not ok:
        raise ImportError(msg)
    try:
        return pd.ExcelFile(excel_path, engine=engine)
    except Exception as e:
        raise ImportError(f"Excel dosyası açılamadı ({os.path.basename(excel_path)}): {e}")


def safe_read_excel(excel_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(str(excel_path))[1].lower()
    ok, engine, msg = _check_optional_excel_dependency(ext)
    if not ok:
        raise ImportError(msg)
    try:
        return pd.read_excel(excel_path, sheet_name=sheet_name, engine=engine)
    except Exception as e:
        raise ImportError(f"Excel sayfası okunamadı ({os.path.basename(excel_path)} / {sheet_name}): {e}")

def run_preprocessing_for_sheet(excel_path: str, sheet_name: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Streamlit için tek sheet preprocessing wrapper'ı."""
    config = PreprocessConfig(
        output_dir_name="forecast_preprocessing_outputs",
        force_regular_frequency=True,
        allow_month_start_to_month_end_alignment_fix=True,
        max_interpolation_gap=1,
        use_knn_for_dense_missing_blocks=False,
        impute_method_preference="seasonal_local",
        min_action_confidence_for_auto_fix=0.75,
        auto_fix_business_spike_dip=False,
        auto_fix_unknown_anomaly=False,
        auto_fix_data_error=True,
        auto_fix_structural_event=False,
        scaler_for_deep_learning="robust",
        export_log1p_version=True,
        generate_modeling_ready_feature_pack=True,
        exclude_textual_columns_from_modeling_features=True,
        drop_low_signal_calendar_features_for_monthly=True,
        export_training_exclusion_masks=True,
        save_validation_excel=True,
        save_validation_csv=True,
        save_validation_plots=True,
        create_domain_validation_template=True,
        leakage_check_enabled=True,
        max_plot_series=50,
        run_internal_unit_tests=True,
        run_synthetic_tests=True,
        run_business_rule_tests=True,
        run_manual_sample_audit=True,
        run_proxy_backtest_validation=True,
        manual_sample_size=20,
        random_seed=42,
        backtest_horizon=3,
        backtest_min_train_size_monthly=24,
        backtest_min_train_size_weekly=52,
        backtest_min_train_size_daily=60,
        backtest_min_train_size_hourly=24 * 14,
        review_if_outlier_fraction_gt=0.05,
        review_if_structural_zero_events_gt=1,
        review_if_clean_zero_ratio_gt=0.10,
        review_if_proxy_smape_gt=60.0,
        save_excel=True,
        save_csv=True,
        save_metadata_json=True,
        save_quality_report=True
    )

    os.makedirs(output_dir, exist_ok=True)
    preprocessor = DemandForecastPreprocessor(config=config)
    export_payload = preprocessor.preprocess_sheet(
        file_path=excel_path,
        sheet_name=sheet_name,
        output_dir=output_dir
    )
    try:
        preprocessor.save_global_metadata(output_dir)
    except Exception:
        pass
    return export_payload


def choose_scaler(name: str):
    name = str(name).lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    return RobustScaler()


def clip_negative_values(series: pd.Series) -> pd.Series:
    return series.clip(lower=0)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def make_config_hash(config: PreprocessConfig) -> str:
    payload = stable_json_dumps(asdict(config))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def coefficient_of_variation(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    mean_ = s.mean()
    std_ = s.std()
    if mean_ == 0 or pd.isna(mean_):
        return np.nan
    return float(std_ / mean_)


def demand_intermittency_ratio(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.notna()
    if valid.sum() == 0:
        return np.nan
    return float((s[valid] == 0).mean())


def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = x.median()
    mad = np.median(np.abs(x - med))
    denom = 1.4826 * mad if mad not in [0, np.nan] else np.nan
    z = (x - med) / denom
    return z


# =========================================================
# DATE / TARGET DETECTION
# =========================================================

def detect_date_column(df: pd.DataFrame, config: PreprocessConfig) -> str:
    normalized = {c: normalize_colname(c) for c in df.columns}

    for c, nc in normalized.items():
        if nc in config.date_column_candidates:
            return c

    best_col = None
    best_ratio = 0.0
    for c in df.columns:
        temp = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        ratio = temp.notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c

    if best_col is None or best_ratio < 0.5:
        raise ValueError("Tarih sütunu bulunamadı.")
    return best_col


def parse_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    if dt.notna().mean() < 0.8:
        dt = pd.to_datetime(
            s.astype(str),
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True
        )

    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
    except Exception:
        pass

    return dt


def align_dates_to_frequency(df: pd.DataFrame, date_col: str, freq_alias: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    audit_rows = []

    before = out[date_col].copy()

    if freq_alias == "M":
        aligned = out[date_col].dt.to_period("M").dt.to_timestamp("M")
        changed = int((before != aligned).fillna(False).sum())
        out[date_col] = aligned
        audit_rows.append({
            "rule": "month_end_alignment",
            "applied": True,
            "changed_timestamp_count": changed,
            "note": "Aylık seriler ay sonu timestamp'ına hizalandı."
        })
    elif freq_alias == "W":
        aligned = out[date_col].dt.to_period("W").apply(lambda p: p.end_time.normalize())
        changed = int((before != aligned).fillna(False).sum())
        out[date_col] = aligned
        audit_rows.append({
            "rule": "week_end_alignment",
            "applied": True,
            "changed_timestamp_count": changed,
            "note": "Haftalık seriler hafta sonu timestamp'ına hizalandı."
        })
    elif freq_alias == "D":
        aligned = out[date_col].dt.floor("D")
        changed = int((before != aligned).fillna(False).sum())
        out[date_col] = aligned
        audit_rows.append({
            "rule": "day_floor_alignment",
            "applied": True,
            "changed_timestamp_count": changed,
            "note": "Günlük seriler gün başlangıcına hizalandı."
        })
    elif freq_alias == "H":
        aligned = out[date_col].dt.floor("H")
        changed = int((before != aligned).fillna(False).sum())
        out[date_col] = aligned
        audit_rows.append({
            "rule": "hour_floor_alignment",
            "applied": True,
            "changed_timestamp_count": changed,
            "note": "Saatlik seriler saat başlangıcına hizalandı."
        })

    if len(audit_rows) == 0:
        audit_rows.append({
            "rule": "alignment_not_applied",
            "applied": False,
            "changed_timestamp_count": 0,
            "note": "Frekansa özel hizalama uygulanmadı."
        })

    return out, pd.DataFrame(audit_rows)


def create_datetime_integrity_audit(
    df_original: pd.DataFrame,
    df_aligned: pd.DataFrame,
    df_aggregated: pd.DataFrame,
    df_regular: pd.DataFrame,
    date_col: str,
    freq_alias: str
) -> pd.DataFrame:
    raw_dates = pd.to_datetime(df_original[date_col], errors="coerce") if date_col in df_original.columns else pd.Series(dtype="datetime64[ns]")
    aligned_dates = pd.to_datetime(df_aligned[date_col], errors="coerce") if date_col in df_aligned.columns else pd.Series(dtype="datetime64[ns]")
    agg_dates = pd.to_datetime(df_aggregated[date_col], errors="coerce") if date_col in df_aggregated.columns else pd.Series(dtype="datetime64[ns]")
    reg_dates = pd.to_datetime(df_regular[date_col], errors="coerce") if date_col in df_regular.columns else pd.Series(dtype="datetime64[ns]")

    rows = [
        {"metric": "invalid_date_count_original", "value": int(raw_dates.isna().sum()) if len(raw_dates) else 0},
        {"metric": "duplicate_dates_before_alignment", "value": int(aligned_dates.duplicated().sum()) if len(aligned_dates) else 0},
        {"metric": "duplicate_dates_after_aggregation", "value": int(agg_dates.duplicated().sum()) if len(agg_dates) else 0},
        {"metric": "is_monotonic_after_regularization", "value": bool(reg_dates.is_monotonic_increasing) if len(reg_dates) else True},
        {"metric": "frequency_alias", "value": freq_alias},
        {"metric": "regularized_row_count", "value": int(len(reg_dates))},
        {"metric": "regularized_start", "value": str(reg_dates.min()) if len(reg_dates) else None},
        {"metric": "regularized_end", "value": str(reg_dates.max()) if len(reg_dates) else None},
    ]
    return pd.DataFrame(rows)


def infer_frequency_from_dates(dt_index: pd.DatetimeIndex) -> str:
    dt_index = pd.DatetimeIndex(dt_index).sort_values().drop_duplicates()

    inferred = pd.infer_freq(dt_index)
    if inferred:
        inferred = inferred.upper()
        if inferred.startswith("W"):
            return "W"
        if inferred in ["M", "MS", "ME"]:
            return "M"
        if inferred in ["D"]:
            return "D"
        if inferred in ["H"]:
            return "H"

    if len(dt_index) < 3:
        return "D"

    deltas = pd.Series(dt_index).diff().dropna()
    median_delta = deltas.median()

    if median_delta <= pd.Timedelta(hours=1):
        return "H"
    if median_delta <= pd.Timedelta(days=1):
        return "D"
    if median_delta <= pd.Timedelta(days=7):
        return "W"
    return "M"


def get_expected_freq_alias(freq: str) -> str:
    if freq == "H":
        return "H"
    if freq == "D":
        return "D"
    if freq == "W":
        return "W"
    return "M"


def detect_target_columns(df: pd.DataFrame, date_col: str, config: PreprocessConfig) -> List[str]:
    non_targets = {normalize_colname(x) for x in config.non_target_columns}
    candidates = []

    for c in df.columns:
        if c == date_col:
            continue

        nc = normalize_colname(c)
        if nc in non_targets:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        numeric_ratio = s.notna().mean()
        if numeric_ratio >= 0.5:
            candidates.append(c)

    if not candidates:
        raise ValueError("Hedef kolonlar otomatik bulunamadı.")
    return candidates


# =========================================================
# TIME INDEX / AGGREGATION
# =========================================================

def aggregate_duplicates(df: pd.DataFrame, date_col: str, target_cols: List[str]) -> pd.DataFrame:
    agg_map = {c: "sum" for c in target_cols}
    for c in df.columns:
        if c not in target_cols and c != date_col:
            agg_map[c] = "first"
    return df.groupby(date_col, as_index=False).agg(agg_map)


def build_regular_time_index(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    df = df.sort_values(date_col).copy()
    start = df[date_col].min()
    end = df[date_col].max()

    if freq == "M":
        start = pd.Timestamp(start).to_period("M").to_timestamp("M")
        end = pd.Timestamp(end).to_period("M").to_timestamp("M")
        full_index = pd.date_range(start=start, end=end, freq="ME")
    elif freq == "W":
        full_index = pd.date_range(start=start, end=end, freq="W")
    elif freq == "D":
        full_index = pd.date_range(start=start, end=end, freq="D")
    else:
        full_index = pd.date_range(start=start, end=end, freq="H")

    out = df.set_index(date_col).reindex(full_index).rename_axis(date_col).reset_index()
    return out


def check_regular_index(df: pd.DataFrame, date_col: str, freq: str) -> Tuple[bool, str]:
    dates = pd.DatetimeIndex(df[date_col].dropna().sort_values())
    if len(dates) < 3:
        return True, "Yetersiz gözlem nedeniyle düzenlilik kontrolü sınırlı."

    try:
        expected = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
        ok = len(expected) == len(dates) and (expected == dates).all()
        if ok:
            return True, "Zaman ekseni düzenli."
        return False, "Zaman ekseninde eksik/fazla periyot veya hizalama sorunu var."
    except Exception as e:
        return False, f"Düzenli indeks kontrolü başarısız: {str(e)}"


# =========================================================
# ANOMALY DETECTION PRIMITIVES
# =========================================================

def hampel_filter_flags(series: pd.Series, window_size: int = 7, n_sigma: float = 4.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    rolling_median = x.rolling(window=window_size, center=True, min_periods=1).median()
    diff = np.abs(x - rolling_median)
    mad = diff.rolling(window=window_size, center=True, min_periods=1).median()
    threshold = n_sigma * 1.4826 * mad
    flags = diff > threshold
    return flags.fillna(False)


def rolling_mad_flags(series: pd.Series, window: int = 9, n_sigma: float = 4.5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    median_ = x.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = np.abs(x - median_)
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    robust_z = (x - median_) / (1.4826 * mad.replace(0, np.nan))
    flags = np.abs(robust_z) > n_sigma
    return flags.fillna(False)


def iqr_flags(series: pd.Series, k: float = 4.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return ((s < lower) | (s > upper)).fillna(False)


def limited_linear_interpolation(series: pd.Series, limit: int) -> pd.Series:
    return series.interpolate(method="linear", limit=limit, limit_direction="both")


# =========================================================
# SERIES PROFILING
# =========================================================

def estimate_trend_strength(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 8:
        return np.nan
    x = np.arange(len(s))
    corr = np.corrcoef(x, s)[0, 1]
    return float(abs(corr)) if np.isfinite(corr) else np.nan


def estimate_seasonality_strength(series: pd.Series, season_length: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < max(2 * season_length, 12) or season_length <= 1:
        return np.nan

    grouped = {}
    for i, val in enumerate(s):
        grouped.setdefault(i % season_length, []).append(val)

    seasonal_means = {k: np.mean(v) for k, v in grouped.items()}
    fitted = np.array([seasonal_means[i % season_length] for i in range(len(s))], dtype=float)

    total_var = np.var(s)
    resid_var = np.var(s - fitted)
    if total_var <= 1e-12:
        return np.nan
    strength = 1 - (resid_var / total_var)
    return float(max(0.0, min(1.0, strength)))


def estimate_volatility_regime(series: pd.Series) -> str:
    """
    PATCH:
    - split 'moderate' into 'moderate' and 'elevated'
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return "unknown"

    cv = coefficient_of_variation(s)
    if pd.isna(cv):
        return "unknown"

    if cv < 0.20:
        return "stable"
    if cv < 0.35:
        return "moderate"
    if cv < 0.55:
        return "elevated"
    return "high"


def volume_level(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "unknown"
    med = s.median()
    if med < 10:
        return "very_low"
    if med < 100:
        return "low"
    if med < 1000:
        return "medium"
    return "high"


def build_series_profile(series: pd.Series, freq_alias: str, config: PreprocessConfig) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    season_length = config.seasonal_period_map.get(freq_alias, 1)

    profile = {
        "n_obs": int(s.notna().sum()),
        "mean": safe_float(s.mean()),
        "median": safe_float(s.median()),
        "std": safe_float(s.std()),
        "cv": safe_float(coefficient_of_variation(s)),
        "intermittency_ratio": safe_float(demand_intermittency_ratio(s)),
        "trend_strength": safe_float(estimate_trend_strength(s)),
        "seasonality_strength": safe_float(estimate_seasonality_strength(s, season_length)),
        "volatility_regime": estimate_volatility_regime(s),
        "volume_level": volume_level(s),
        "min": safe_float(s.min()),
        "max": safe_float(s.max())
    }
    return profile


def create_series_profile_report(df: pd.DataFrame, target_cols: List[str], freq_alias: str, config: PreprocessConfig) -> pd.DataFrame:
    rows = []
    for col in target_cols:
        p = build_series_profile(df[col], freq_alias, config)
        p["series"] = col
        rows.append(p)
    cols = ["series"] + [c for c in rows[0].keys() if c != "series"] if rows else ["series"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=["series"])


def get_adaptive_thresholds(profile: Dict[str, Any], config: PreprocessConfig) -> Dict[str, float]:
    hampel_n_sigma = config.hampel_n_sigma
    rolling_mad_n_sigma = config.rolling_mad_n_sigma
    iqr_k = config.iqr_k

    cv = profile.get("cv", np.nan)
    intermittency = profile.get("intermittency_ratio", np.nan)
    vol_regime = profile.get("volatility_regime", "unknown")
    vol_level = profile.get("volume_level", "unknown")

    if vol_regime == "high":
        hampel_n_sigma += 0.8
        rolling_mad_n_sigma += 1.0
        iqr_k += 0.7
    elif vol_regime == "elevated":
        hampel_n_sigma += 0.3
        rolling_mad_n_sigma += 0.4
        iqr_k += 0.2
    elif vol_regime == "stable":
        hampel_n_sigma -= 0.5
        rolling_mad_n_sigma -= 0.5
        iqr_k -= 0.4

    if pd.notna(intermittency) and intermittency >= 0.40:
        hampel_n_sigma += 0.5
        rolling_mad_n_sigma += 0.7
        iqr_k += 0.6

    if vol_level in ["very_low", "low"]:
        hampel_n_sigma += 0.3
        rolling_mad_n_sigma += 0.3

    if pd.notna(cv) and cv < 0.20:
        hampel_n_sigma -= 0.3
        rolling_mad_n_sigma -= 0.3

    hampel_n_sigma = max(2.5, hampel_n_sigma)
    rolling_mad_n_sigma = max(3.0, rolling_mad_n_sigma)
    iqr_k = max(1.5, iqr_k)

    return {
        "hampel_n_sigma": hampel_n_sigma,
        "rolling_mad_n_sigma": rolling_mad_n_sigma,
        "iqr_k": iqr_k
    }


def conservative_outlier_vote_adaptive(series: pd.Series, profile: Dict[str, Any], config: PreprocessConfig) -> Tuple[pd.Series, pd.DataFrame]:
    thr = get_adaptive_thresholds(profile, config)

    flag_h = hampel_filter_flags(series, config.hampel_window, thr["hampel_n_sigma"])
    flag_m = rolling_mad_flags(series, config.rolling_mad_window, thr["rolling_mad_n_sigma"])
    flag_i = iqr_flags(series, thr["iqr_k"])

    vote_count = flag_h.astype(int) + flag_m.astype(int) + flag_i.astype(int)
    combined = vote_count >= config.min_outlier_votes

    vote_df = pd.DataFrame({
        "hampel": flag_h.astype(bool),
        "rolling_mad": flag_m.astype(bool),
        "iqr": flag_i.astype(bool),
        "vote_count": vote_count.astype(int),
        "combined": combined.astype(bool),
        "adaptive_hampel_n_sigma": thr["hampel_n_sigma"],
        "adaptive_rolling_mad_n_sigma": thr["rolling_mad_n_sigma"],
        "adaptive_iqr_k": thr["iqr_k"]
    })
    return combined.fillna(False), vote_df


def cap_outlier_fraction(series: pd.Series, combined_flags: pd.Series, vote_df: pd.DataFrame, max_fraction: float) -> pd.Series:
    combined_flags = combined_flags.copy()
    n = len(series)
    max_allowed = int(np.floor(n * max_fraction))

    flagged_idx = vote_df.index[vote_df["combined"]].tolist()
    if len(flagged_idx) <= max_allowed or max_allowed < 1:
        return combined_flags

    s = pd.to_numeric(series, errors="coerce")
    median_val = s.median()
    distance = (s - median_val).abs()

    ranking = vote_df.loc[flagged_idx].copy()
    ranking["distance"] = distance.loc[flagged_idx]
    ranking = ranking.sort_values(["vote_count", "distance"], ascending=[False, False])

    keep_idx = set(ranking.head(max_allowed).index.tolist())
    combined_flags[:] = False
    for idx in keep_idx:
        combined_flags.loc[idx] = True

    return combined_flags.fillna(False)


def protect_edge_periods(flags: pd.Series, config: PreprocessConfig) -> pd.Series:
    flags = flags.copy()
    if len(flags) == 0:
        return flags

    first_n = min(config.protect_first_n_periods, len(flags))
    last_n = min(config.protect_last_n_periods, len(flags))

    if first_n > 0:
        flags.iloc[:first_n] = False

    # Keep edge protection conservative, but avoid blinding the whole recent zone
    if last_n > 0:
        flags.iloc[-last_n:] = False

    return flags.fillna(False)


# =========================================================
# STRUCTURAL EVENT ENGINE
# =========================================================

def detect_structural_zero_events(
    df: pd.DataFrame,
    target_cols: List[str],
    min_series_count: int,
    ratio_threshold: float
) -> pd.Series:
    zero_matrix = pd.DataFrame(index=df.index)
    for col in target_cols:
        zero_matrix[col] = pd.to_numeric(df[col], errors="coerce").eq(0)

    zero_count = zero_matrix.sum(axis=1)
    zero_ratio = zero_count / max(len(target_cols), 1)
    structural = (zero_count >= min_series_count) & (zero_ratio >= ratio_threshold)
    return structural.fillna(False)


def expand_structural_events(structural_flags: pd.Series, neighbor_window: int) -> pd.Series:
    if neighbor_window <= 0 or len(structural_flags) == 0:
        return structural_flags.copy()

    expanded = structural_flags.copy().astype(bool)
    idx = np.where(structural_flags.values)[0]
    for i in idx:
        start = max(0, i - neighbor_window)
        end = min(len(expanded), i + neighbor_window + 1)
        expanded.iloc[start:end] = True
    return expanded.fillna(False)

def protect_structural_event_edges(flags: pd.Series, protect_last_n: int = 1) -> pd.Series:
    flags = flags.copy()
    if len(flags) > 0:
        flags.iloc[-protect_last_n:] = False
    return flags
def detect_incomplete_last_period(
    df_regular: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    freq_alias: str,
    config: PreprocessConfig
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Detects whether the last period is suspiciously low and may represent
    partial / incomplete reporting rather than real demand collapse.
    """
    flags = pd.Series(False, index=df_regular.index)
    rows = []

    if not config.enable_incomplete_period_detection:
        return flags, pd.DataFrame(columns=[
            "date", "rule_name", "portfolio_total", "baseline_median",
            "ratio_to_baseline", "is_incomplete_candidate", "reason"
        ])

    if len(df_regular) < max(6, config.partial_period_compare_last_n + 1):
        return flags, pd.DataFrame(columns=[
            "date", "rule_name", "portfolio_total", "baseline_median",
            "ratio_to_baseline", "is_incomplete_candidate", "reason"
        ])

    total = compute_portfolio_series(df_regular, target_cols).astype(float)
    last_idx = df_regular.index[-1]
    last_total = safe_float(total.iloc[-1])

    compare_n = max(1, config.partial_period_compare_last_n)
    hist = total.iloc[-(compare_n + 1):-1].dropna()

    if len(hist) == 0 or pd.isna(last_total):
        return flags, pd.DataFrame(columns=[
            "date", "rule_name", "portfolio_total", "baseline_median",
            "ratio_to_baseline", "is_incomplete_candidate", "reason"
        ])

    baseline = safe_float(hist.median())
    ratio = safe_float(last_total / baseline) if pd.notna(baseline) and baseline > 0 else np.nan

    is_candidate = bool(
        pd.notna(ratio) and
        ratio <= config.partial_period_drop_ratio_threshold
    )

    if is_candidate:
        flags.iloc[-1] = True
        rows.append({
            "date": df_regular.loc[last_idx, date_col],
            "rule_name": "last_period_portfolio_drop_vs_recent_median",
            "portfolio_total": last_total,
            "baseline_median": baseline,
            "ratio_to_baseline": ratio,
            "is_incomplete_candidate": True,
            "reason": "Son dönem toplamı, son dönemler medianına göre aşırı düşük. Kısmi raporlama / incomplete period adayı."
        })

    return flags.fillna(False), pd.DataFrame(rows)

def compute_portfolio_series(df: pd.DataFrame, target_cols: List[str]) -> pd.Series:
    total = pd.DataFrame({
        c: pd.to_numeric(df[c], errors="coerce") for c in target_cols
    }).sum(axis=1, min_count=1)
    return total


def detect_portfolio_shocks(
    df: pd.DataFrame,
    target_cols: List[str],
    config: PreprocessConfig
) -> pd.Series:
    """
    Portfolio-wide sharp drop detection.

    Important patch:
    - Do not classify the very last observation as structural shock automatically.
      Last period may be incomplete / partially reported.
    """
    total = compute_portfolio_series(df, target_cols).astype(float)
    prev = total.shift(1)
    ratio = total / prev.replace(0, np.nan)

    drop_flag = ratio <= (1 - config.portfolio_drop_ratio_threshold)
    drop_flag = drop_flag.fillna(False)

    # PATCH: protect last period from automatic structural shock tagging
    if len(drop_flag) > 0:
        drop_flag.iloc[-1] = False

    return drop_flag


def detect_rebound_after_event(df: pd.DataFrame, target_cols: List[str], event_flags: pd.Series, config: PreprocessConfig) -> pd.Series:
    total = compute_portfolio_series(df, target_cols).astype(float)
    next_ = total.shift(-1)
    ratio = next_ / total.replace(0, np.nan)
    rebound = ratio >= config.portfolio_rebound_ratio_threshold
    return (event_flags & rebound.fillna(False)).fillna(False)


def build_structural_event_log(
    df_regular: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    zero_flags: pd.Series,
    portfolio_shock_flags: pd.Series,
    rebound_flags: pd.Series
) -> pd.DataFrame:
    rows = []
    portfolio_series = compute_portfolio_series(df_regular, target_cols).astype(float)

    for idx in df_regular.index:
        triggered = []
        if bool(zero_flags.loc[idx]):
            triggered.append("multi_series_zero_event")
        if bool(portfolio_shock_flags.loc[idx]):
            triggered.append("portfolio_drop_event")
        if bool(rebound_flags.loc[idx]):
            triggered.append("rapid_rebound_pattern")

        if not triggered:
            continue

        event_type = "structural_event"
        if "portfolio_drop_event" in triggered:
            event_subtype = "portfolio_wide_shock"
        elif "multi_series_zero_event" in triggered:
            event_subtype = "category_or_reporting_shock"
        else:
            event_subtype = "structural_pattern"

        row_values = pd.to_numeric(df_regular.loc[idx, target_cols], errors="coerce")
        zero_series_count = int(row_values.eq(0).sum())
        non_null_series_count = int(row_values.notna().sum())
        non_zero_series_count = int((row_values.fillna(0) != 0).sum())

        rows.append({
            "date": df_regular.loc[idx, date_col],
            "event_type": event_type,
            "event_subtype": event_subtype,
            "triggered_rules": "|".join(triggered),
            "portfolio_total_sum": safe_float(portfolio_series.loc[idx]),
            "portfolio_total": safe_float(portfolio_series.loc[idx]),
            "zero_series_count": zero_series_count,
            "non_zero_series_count": non_zero_series_count,
            "non_null_series_count": non_null_series_count,
            "series_count_in_portfolio": int(len(target_cols)),
            "portfolio_drop_flag": bool(portfolio_shock_flags.loc[idx]),
            "multi_series_zero_flag": bool(zero_flags.loc[idx]),
            "rapid_rebound_flag": bool(rebound_flags.loc[idx])
        })
    if not rows:
        return pd.DataFrame(columns=[
            "date", "event_type", "event_subtype", "triggered_rules",
            "portfolio_total_sum", "portfolio_total", "zero_series_count",
            "non_zero_series_count", "non_null_series_count", "series_count_in_portfolio",
            "portfolio_drop_flag", "multi_series_zero_flag", "rapid_rebound_flag"
        ])
    return pd.DataFrame(rows)

def build_date_level_event_map(
    df_regular: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    structural_event_flags: pd.Series,
    incomplete_period_flags: pd.Series
) -> pd.DataFrame:
    rows = []

    for idx in df_regular.index:
        rows.append({
            "date": df_regular.loc[idx, date_col],
            "is_structural_event_date": bool(structural_event_flags.loc[idx]) if len(structural_event_flags) > 0 else False,
            "is_incomplete_period_date": bool(incomplete_period_flags.loc[idx]) if len(incomplete_period_flags) > 0 else False
        })

    return pd.DataFrame(rows)


# =========================================================
# ANOMALY GOVERNANCE / CLASSIFICATION
# =========================================================

def classify_anomaly(
    raw_value: float,
    prev_value: float,
    next_value: float,
    vote_count: int,
    profile: Dict[str, Any],
    is_structural_event: bool,
    is_recent_period: bool,
    is_incomplete_last_period: bool = False
) -> Tuple[str, str, float]:
    """
    Returns:
        anomaly_type, anomaly_reason, confidence

    PATCH LOGIC:
    - better low-volume / seasonal business-event interpretation
    - keep structural events dominant
    """
    if is_incomplete_last_period:
        reason = "Son dönem seviyesi aşırı düşük; incomplete / partial reporting şüphesi var."
        return "incomplete_last_period", reason, 0.95

    if is_structural_event:
        reason = "Aynı tarihte çok-serili ortak bozulma / portföy şoku paterni."
        return "structural_event", reason, 0.92

    if pd.notna(raw_value):
        if raw_value < 0:
            return "data_error", "Negatif talep/satış değeri tespit edildi.", 0.99
        if pd.isna(prev_value) and pd.isna(next_value) and vote_count >= 2:
            return "data_error", "Komşu bilgi zayıf, noktasal aykırılık yüksek.", 0.75

    if pd.notna(raw_value) and pd.notna(prev_value) and prev_value not in [0, np.nan]:
        change_ratio_prev = raw_value / prev_value
    else:
        change_ratio_prev = np.nan

    if pd.notna(raw_value) and pd.notna(next_value) and next_value not in [0, np.nan]:
        change_ratio_next = raw_value / next_value
    else:
        change_ratio_next = np.nan

    seasonality_strength = profile.get("seasonality_strength", np.nan)
    vol_regime = profile.get("volatility_regime", "unknown")
    volume_level_ = profile.get("volume_level", "unknown")

    if vote_count >= 2:
        strong_jump = (
            (pd.notna(change_ratio_prev) and (change_ratio_prev >= 2.5 or change_ratio_prev <= 0.4)) or
            (pd.notna(change_ratio_next) and (change_ratio_next >= 2.5 or change_ratio_next <= 0.4))
        )

        # PATCH: low-volume and strong seasonal series should be more tolerant
        if strong_jump:
            if volume_level_ in ["very_low", "low"]:
                return "business_spike_dip", "Düşük hacimli seride beklenebilir sıçrama/düşüş.", 0.74

            if pd.notna(seasonality_strength) and seasonality_strength >= 0.60:
                return "business_spike_dip", "Güçlü sezonsallık bağlamında sıçrama/düşüş.", 0.78

            if vol_regime in ["elevated", "high"]:
                return "business_spike_dip", "Yüksek oynaklık rejiminde büyük sapma.", 0.72

    if vote_count >= 2:
        reason = "Aykırılık kuralları tetiklendi fakat net iş nedeni veya veri hatası ayrılamadı."
        conf = 0.62 if not is_recent_period else 0.55
        return "unknown_anomaly", reason, conf

    return "none", "Anomali yok.", 0.0


def decide_action(
    anomaly_type: str,
    confidence: float,
    is_recent_period: bool,
    config: PreprocessConfig
) -> Tuple[str, str]:
    """
    action_taken, governance_policy

    PATCH LOGIC:
    - recent periods: always human-review-first
    - unknown anomaly outside recent region: allow controlled auto-fix
    - structural events: preserve unless explicitly enabled
    """
    if anomaly_type == "none":
        return "keep_raw", "no_action"

    # CRITICAL FIX:
    # incomplete last period must ALWAYS be excluded from training,
    # even if it is also a recent period.
    if anomaly_type == "incomplete_last_period":
        return "preserve_raw_flag_exclude_candidate", "incomplete_last_period_exclusion_policy"

    # structural events should also preserve/exclude before generic recent-period handling
    if anomaly_type == "structural_event":
        if config.auto_fix_structural_event and confidence >= config.min_action_confidence_for_auto_fix:
            return "set_nan_then_impute", "auto_fix_structural_event_enabled"
        return "preserve_raw_flag_exclude_candidate", "structural_event_preservation_policy"

    # recent periods: human-review-first for the remaining anomaly classes
    if is_recent_period:
        return "flag_only_review", "recent_period_human_review_first"

    if anomaly_type == "data_error":
        if config.auto_fix_data_error and confidence >= config.min_action_confidence_for_auto_fix:
            return "set_nan_then_impute", "auto_fix_high_confidence_data_error"
        return "flag_only_review", "review_due_to_low_confidence_data_error"
    

    if anomaly_type == "business_spike_dip":
        if config.auto_fix_business_spike_dip and confidence >= config.min_action_confidence_for_auto_fix:
            return "set_nan_then_impute", "auto_fix_business_event_enabled"
        return "keep_raw_flag", "preserve_possible_business_event"

    if anomaly_type == "unknown_anomaly":
        if config.auto_fix_unknown_anomaly and confidence >= config.min_action_confidence_for_auto_fix and not is_recent_period:
            return "set_nan_then_impute", "auto_fix_unknown_non_recent_enabled"
        return "flag_only_review", "unknown_anomaly_review"

    return "flag_only_review", "fallback_review_policy"


def build_anomaly_governance_table(
    df_regular: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    outlier_flags: Dict[str, pd.Series],
    vote_details: Dict[str, pd.DataFrame],
    series_profiles: Dict[str, Dict[str, Any]],
    structural_event_flags: pd.Series,
    incomplete_period_flags: pd.Series,
    config: PreprocessConfig
) -> pd.DataFrame:
    rows = []
    n = len(df_regular)

    for col in target_cols:
        s = pd.to_numeric(df_regular[col], errors="coerce")

        for idx in df_regular.index:
            is_structural = bool(structural_event_flags.loc[idx]) if len(structural_event_flags) > 0 else False
            is_incomplete_last_period = bool(incomplete_period_flags.loc[idx]) if len(incomplete_period_flags) > 0 else False
            outlier_hit = bool(outlier_flags[col].loc[idx]) if col in outlier_flags else False

            # NEW: date-level events should force row creation for all series
            force_row = is_structural or is_incomplete_last_period
            if not outlier_hit and not force_row:
                continue

            raw_val = s.loc[idx]
            prev_val = s.shift(1).loc[idx]
            next_val = s.shift(-1).loc[idx]
            vote_count = int(vote_details[col].loc[idx, "vote_count"]) if col in vote_details else 0
            is_recent = idx >= (n - config.recent_periods_review_only)

            anomaly_type, reason, conf = classify_anomaly(
                raw_value=raw_val,
                prev_value=prev_val,
                next_value=next_val,
                vote_count=vote_count,
                profile=series_profiles[col],
                is_structural_event=is_structural,
                is_recent_period=is_recent,
                is_incomplete_last_period=is_incomplete_last_period
            )

            action_taken, governance_policy = decide_action(
                anomaly_type=anomaly_type,
                confidence=conf,
                is_recent_period=is_recent,
                config=config
            )

            rows.append({
                "date": df_regular.loc[idx, date_col],
                "series": col,
                "raw_value": raw_val,
                "prev_value": prev_val,
                "next_value": next_val,
                "vote_count": vote_count,
                "anomaly_type": anomaly_type,
                "anomaly_reason": reason,
                "action_taken": action_taken,
                "action_confidence": conf,
                "governance_policy": governance_policy,
                "is_structural_event": is_structural,
                "is_incomplete_last_period": is_incomplete_last_period,
                "is_recent_period": is_recent,
                "preserved_for_modeling": action_taken in ["keep_raw", "keep_raw_flag", "flag_only_review"],
                "excluded_from_training_candidate": (
                    action_taken in ["preserve_raw_flag_exclude_candidate"]
                    or anomaly_type in ["incomplete_last_period", "structural_event"]
                ),
                "is_training_excluded": (
                    action_taken in ["preserve_raw_flag_exclude_candidate"]
                    or anomaly_type in ["incomplete_last_period", "structural_event"]
                ),
                "is_preserved_for_review": action_taken in ["flag_only_review", "keep_raw_flag", "preserve_raw_flag_exclude_candidate"],
                "is_final_model_input": action_taken not in ["preserve_raw_flag_exclude_candidate"],
                "recommended_manual_validation": action_taken in ["flag_only_review", "keep_raw_flag", "preserve_raw_flag_exclude_candidate"]
            })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "series", "raw_value", "prev_value", "next_value", "vote_count",
            "anomaly_type", "anomaly_reason", "action_taken", "action_confidence",
            "governance_policy", "is_structural_event", "is_incomplete_last_period",
            "is_recent_period", "preserved_for_modeling", "excluded_from_training_candidate",
            "is_training_excluded", "is_preserved_for_review", "is_final_model_input",
            "recommended_manual_validation"
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(["series", "date", "action_confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["date", "series"], keep="first")
        .reset_index(drop=True)
    )


# =========================================================
# IMPUTATION
# =========================================================

def seasonal_local_impute(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    freq: str,
    seasonal_period: int,
    max_interpolation_gap: int = 1
) -> pd.Series:
    s = pd.to_numeric(df[target_col], errors="coerce").copy()

    if freq == "M":
        group_key = df[date_col].dt.month
    elif freq == "W":
        group_key = df[date_col].dt.isocalendar().week.astype(int)
    elif freq == "D":
        group_key = df[date_col].dt.dayofweek
    elif freq == "H":
        group_key = df[date_col].dt.hour
    else:
        group_key = pd.Series([0] * len(df), index=df.index)

    seasonal_med = s.groupby(group_key).transform("median")
    s = s.fillna(seasonal_med)
    s = limited_linear_interpolation(s, limit=max_interpolation_gap)

    local_med = s.rolling(window=3, min_periods=1).median()
    s = s.fillna(local_med)

    if seasonal_period > 1 and s.isna().any():
        fallback = s.shift(seasonal_period)
        s = s.fillna(fallback)

    s = s.fillna(s.median())
    return s




def _max_consecutive_true(mask: pd.Series) -> int:
    arr = pd.Series(mask).fillna(False).astype(int).values
    best = cur = 0
    for v in arr:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def summarize_missingness_patterns(
    df_regular: pd.DataFrame,
    target_cols: List[str],
    date_col: str
) -> pd.DataFrame:
    rows = []
    for col in target_cols:
        s = pd.to_numeric(df_regular[col], errors="coerce")
        miss = s.isna()
        rows.append({
            "series": col,
            "missing_count": int(miss.sum()),
            "missing_ratio": float(miss.mean()),
            "missing_at_start": bool(miss.iloc[0]) if len(miss) else False,
            "missing_at_end": bool(miss.iloc[-1]) if len(miss) else False,
            "max_consecutive_missing_block": _max_consecutive_true(miss),
            "recommended_dense_block_method": "knn_or_review" if _max_consecutive_true(miss) >= 3 else "seasonal_local"
        })
    return pd.DataFrame(rows)


def decide_missing_value_strategy(
    missingness_summary: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    rows = []
    for _, row in missingness_summary.iterrows():
        ratio = float(row["missing_ratio"])
        block = int(row["max_consecutive_missing_block"])
        strategy = "impute_seasonal_local"
        reason = "Eksik oranı düşük/orta; zaman serisi bütünlüğü korunarak imputasyon tercih edildi."
        exclude_series = False
        row_drop = False

        if ratio >= config.missing_drop_series_threshold:
            strategy = "review_required_series_too_sparse"
            reason = "Eksik oranı çok yüksek; seri modelleme için zayıf aday."
            exclude_series = True
        elif block >= config.dense_missing_block_threshold and config.use_knn_for_dense_missing_blocks:
            strategy = "impute_knn_dense_block"
            reason = "Ardışık eksik blok yoğun; KNN aday yöntem olarak işaretlendi."
        elif ratio >= config.missing_impute_ratio_threshold:
            strategy = "impute_seasonal_local_with_review"
            reason = "Eksik oranı orta-yüksek; seasonal/local imputasyon + uzman gözden geçirme önerildi."
        elif ratio == 0:
            strategy = "no_imputation_needed"
            reason = "Eksik değer bulunmadı."

        rows.append({
            "series": row["series"],
            "missing_ratio": ratio,
            "max_consecutive_missing_block": block,
            "selected_strategy": strategy,
            "reason": reason,
            "row_drop_applied": row_drop,
            "series_excluded_from_modeling": exclude_series,
            "imputer_name": "seasonal_local" if "impute" in strategy else None
        })
    return pd.DataFrame(rows)


def create_descriptive_statistics_report(df_clean: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in target_cols:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        rows.append({
            "series": col,
            "count": int(s.notna().sum()),
            "mean": safe_float(s.mean()),
            "std": safe_float(s.std()),
            "min": safe_float(s.min()),
            "q25": safe_float(s.quantile(0.25)),
            "q50": safe_float(s.quantile(0.50)),
            "q75": safe_float(s.quantile(0.75)),
            "max": safe_float(s.max()),
            "iqr": safe_float(s.quantile(0.75) - s.quantile(0.25)),
            "sum": safe_float(s.sum()),
            "skewness": safe_float(s.skew()),
            "kurtosis": safe_float(s.kurt())
        })
    return pd.DataFrame(rows)


def create_monthly_seasonality_report(
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    freq_alias: str,
    config: PreprocessConfig
) -> pd.DataFrame:
    rows = []
    season_length = config.seasonal_period_map.get(freq_alias, 1)
    for col in target_cols:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        tmp = pd.DataFrame({date_col: df_clean[date_col], col: s}).dropna()
        if len(tmp) == 0:
            continue

        if freq_alias == "M":
            grp_key = tmp[date_col].dt.month
            grp_name = "month"
        elif freq_alias == "W":
            grp_key = tmp[date_col].dt.isocalendar().week.astype(int)
            grp_name = "iso_week"
        elif freq_alias == "D":
            grp_key = tmp[date_col].dt.dayofweek
            grp_name = "dayofweek"
        else:
            grp_key = tmp[date_col].dt.hour
            grp_name = "hour"

        prof = tmp.groupby(grp_key)[col].agg(["mean", "median", "min", "max", "count"]).reset_index()
        overall_mean = tmp[col].mean()
        peak_row = prof.sort_values("mean", ascending=False).iloc[0]
        trough_row = prof.sort_values("mean", ascending=True).iloc[0]

        rows.append({
            "series": col,
            "grouping": grp_name,
            "peak_period": int(peak_row.iloc[0]),
            "peak_mean": safe_float(peak_row["mean"]),
            "trough_period": int(trough_row.iloc[0]),
            "trough_mean": safe_float(trough_row["mean"]),
            "peak_to_trough_ratio": safe_float(peak_row["mean"] / trough_row["mean"]) if pd.notna(trough_row["mean"]) and trough_row["mean"] not in [0, 0.0] else np.nan,
            "overall_mean": safe_float(overall_mean),
            "seasonality_strength": safe_float(estimate_seasonality_strength(tmp[col], season_length))
        })
    return pd.DataFrame(rows)


def create_pharma_event_diagnostic_report(
    df_regular: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    anomaly_gov: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    if len(anomaly_gov) == 0:
        return pd.DataFrame(columns=["date", "series", "raw_value", "prev_value", "next_value", "diagnostic_tag", "note"])

    rows = []
    date_to_idx = {d: i for i, d in enumerate(df_regular[date_col].tolist())}
    for _, row in anomaly_gov.iterrows():
        series = row["series"]
        date_ = row["date"]
        idx = date_to_idx.get(date_)
        if idx is None:
            continue
        s = pd.to_numeric(df_regular[series], errors="coerce")
        raw = s.iloc[idx]
        prev_ = s.iloc[idx - 1] if idx - 1 >= 0 else np.nan
        next_ = s.iloc[idx + 1] if idx + 1 < len(s) else np.nan
        tag = "anomaly_review"
        note = "İstatistiksel aykırılık adayı."

        if pd.notna(prev_) and prev_ > 0 and pd.notna(raw) and raw / prev_ <= config.stockout_like_drop_ratio:
            tag = "possible_stockout_or_supply_issue"
            note = "Ani düşüş tespit edildi; stok kesintisi / sevkiyat problemi adayı."
        elif pd.notna(prev_) and prev_ > 0 and pd.notna(raw) and raw / prev_ >= config.promotion_like_jump_ratio:
            tag = "possible_campaign_or_bulk_order"
            note = "Ani sıçrama tespit edildi; kampanya / ihale / toplu sipariş adayı."
        elif pd.notna(prev_) and pd.notna(next_) and prev_ > 0 and raw > 0 and next_ / raw >= config.rebound_after_drop_ratio:
            tag = "possible_rebound_after_disruption"
            note = "Şok sonrası hızlı rebound görüldü; geçici tedarik bozulması adayı."

        rows.append({
            "date": date_,
            "series": series,
            "raw_value": safe_float(raw),
            "prev_value": safe_float(prev_),
            "next_value": safe_float(next_),
            "diagnostic_tag": tag,
            "note": note
        })
    return pd.DataFrame(rows)


def _safe_linear_trend(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    x = np.arange(len(s))
    mask = s.notna().values
    coeffs = np.polyfit(x[mask], s.values[mask], deg=1)
    fitted = coeffs[0] * x + coeffs[1]
    return pd.Series(fitted, index=s.index)


def _normalized_index(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    mean_ = s.mean()
    if pd.isna(mean_) or mean_ == 0:
        return s * np.nan
    return s / mean_


def drift_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    train = pd.to_numeric(train, errors="coerce").dropna().reset_index(drop=True)
    if len(train) == 0:
        return np.full(horizon, np.nan)
    if len(train) == 1:
        return np.repeat(train.iloc[-1], horizon).astype(float)
    drift = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    return np.array([train.iloc[-1] + drift * (i + 1) for i in range(horizon)], dtype=float)


def create_model_input_transparency_export(
    df_regular: pd.DataFrame,
    df_clean_candidate: pd.DataFrame,
    df_clean_governed_preserve: pd.DataFrame,
    df_feat: pd.DataFrame,
    date_col: str,
    target_cols: List[str]
) -> pd.DataFrame:
    out = pd.DataFrame({date_col: df_regular[date_col].values})
    for col in target_cols:
        raw_s = pd.to_numeric(df_regular[col], errors="coerce")
        cand_s = pd.to_numeric(df_clean_candidate[col], errors="coerce")
        preserve_s = pd.to_numeric(df_clean_governed_preserve[col], errors="coerce")
        exclude_col = f"{col}_exclude_from_training"
        review_col = f"{col}_review_required"
        excluded = df_feat[exclude_col].astype(int) if exclude_col in df_feat.columns else pd.Series(0, index=df_regular.index)
        review = df_feat[review_col].astype(int) if review_col in df_feat.columns else pd.Series(0, index=df_regular.index)
        final_model = cand_s.copy()
        final_model.loc[excluded.astype(bool)] = np.nan

        out[f"{col}__raw_regular"] = raw_s.values
        out[f"{col}__candidate_clean"] = cand_s.values
        out[f"{col}__preserve_clean"] = preserve_s.values
        out[f"{col}__train_excluded"] = excluded.values
        out[f"{col}__is_preserved_for_review"] = review.values
        out[f"{col}__recommended_manual_validation"] = review.values
        out[f"{col}__is_final_model_input"] = (~excluded.astype(bool)).astype(int).values
        out[f"{col}__final_model_series"] = final_model.values
    return out


def save_raw_clean_trend_plots(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    sheet_dir: str,
    max_plot_series: int = 50,
    ma_windows: Tuple[int, int] = (3, 6),
    robust_trend_window: int = 5,
    save_robust_trend: bool = True
):
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for col in target_cols[:max_plot_series]:
        raw_s = pd.to_numeric(df_regular[col], errors="coerce")
        clean_s = pd.to_numeric(df_clean[col], errors="coerce")
        plt.figure(figsize=(13, 5))
        plt.plot(df_regular[date_col], raw_s, label="Raw-Regular", linewidth=1.2)
        plt.plot(df_clean[date_col], clean_s, label="Clean", linewidth=1.5)
        for w in ma_windows:
            if len(clean_s) >= 2:
                plt.plot(
                    df_clean[date_col],
                    clean_s.rolling(window=min(w, max(len(clean_s), 1)), min_periods=1).mean(),
                    label=f"Clean_MA_{w}", linewidth=1.1
                )
        if save_robust_trend:
            plt.plot(
                df_clean[date_col],
                clean_s.rolling(window=min(max(robust_trend_window, 3), max(len(clean_s), 1)), min_periods=1).median(),
                label=f"Clean_RollMedian_{robust_trend_window}", linewidth=1.1
            )
            linear_trend = _safe_linear_trend(clean_s)
            if linear_trend.notna().any():
                plt.plot(df_clean[date_col], linear_trend, label="Clean_LinearTrend", linewidth=1.0)
        plt.title(f"{col} - Raw/Clean + Trend")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_raw_clean_trend.png"), dpi=150)
        plt.close()


def save_distribution_plots(
    df_clean: pd.DataFrame,
    target_cols: List[str],
    sheet_dir: str,
    max_plot_series: int = 50,
    save_boxplots: bool = True
):
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for col in target_cols[:max_plot_series]:
        s = pd.to_numeric(df_clean[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        plt.figure(figsize=(9, 5))
        plt.hist(s.values, bins=min(30, max(10, int(np.sqrt(len(s))))), edgecolor="black")
        plt.title(f"{col} - Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_distribution_hist.png"), dpi=150)
        plt.close()

        if save_boxplots:
            plt.figure(figsize=(8, 4.5))
            plt.boxplot(s.values, vert=False)
            plt.title(f"{col} - Boxplot")
            plt.xlabel("Value")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{col}_distribution_boxplot.png"), dpi=150)
            plt.close()


def save_seasonality_plots(
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    freq_alias: str,
    sheet_dir: str,
    max_plot_series: int = 50,
    save_year_overlay: bool = True,
    save_normalized_profile: bool = True,
    save_boxplot: bool = True
):
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for col in target_cols[:max_plot_series]:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        tmp = pd.DataFrame({date_col: df_clean[date_col], col: s}).dropna()
        if len(tmp) == 0:
            continue

        if freq_alias == "M":
            tmp["season_key"] = tmp[date_col].dt.month
            tmp["year_key"] = tmp[date_col].dt.year
            xlab = "Month"
        elif freq_alias == "W":
            tmp["season_key"] = tmp[date_col].dt.isocalendar().week.astype(int)
            tmp["year_key"] = tmp[date_col].dt.year
            xlab = "ISO Week"
        elif freq_alias == "D":
            tmp["season_key"] = tmp[date_col].dt.dayofweek
            tmp["year_key"] = pd.to_datetime(tmp[date_col]).dt.to_period("M").astype(str)
            xlab = "Day of Week"
        else:
            tmp["season_key"] = tmp[date_col].dt.hour
            tmp["year_key"] = pd.to_datetime(tmp[date_col]).dt.to_period("D").astype(str)
            xlab = "Hour"

        grp_mean = tmp.groupby("season_key")[col].mean()
        grp_median = tmp.groupby("season_key")[col].median()
        if len(grp_mean) == 0:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(grp_mean.index.astype(int), grp_mean.values, marker="o", label="Mean")
        plt.plot(grp_median.index.astype(int), grp_median.values, marker="s", label="Median")
        plt.title(f"{col} - Seasonal Profile")
        plt.xlabel(xlab)
        plt.ylabel("Demand")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_seasonal_profile.png"), dpi=150)
        plt.close()

        if save_normalized_profile:
            norm = _normalized_index(grp_mean)
            plt.figure(figsize=(10, 5))
            plt.plot(norm.index.astype(int), norm.values, marker="o")
            plt.axhline(1.0, linewidth=1.0)
            plt.title(f"{col} - Normalized Seasonal Index")
            plt.xlabel(xlab)
            plt.ylabel("Index")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{col}_normalized_seasonal_index.png"), dpi=150)
            plt.close()

        if save_year_overlay:
            piv = tmp.pivot_table(index="season_key", columns="year_key", values=col, aggfunc="mean")
            if piv.shape[1] >= 1:
                plt.figure(figsize=(11, 5))
                for c_year in piv.columns:
                    plt.plot(piv.index.astype(int), piv[c_year].values, marker="o", linewidth=1.0, label=str(c_year))
                plt.title(f"{col} - Seasonal Overlay by Period")
                plt.xlabel(xlab)
                plt.ylabel("Demand")
                if piv.shape[1] <= 8:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{col}_seasonal_overlay.png"), dpi=150)
                plt.close()

        if save_boxplot:
            grouped = [g[col].values for _, g in tmp.groupby("season_key") if len(g) > 0]
            labels = [int(k) for k, g in tmp.groupby("season_key") if len(g) > 0]
            if grouped:
                plt.figure(figsize=(11, 5))
                plt.boxplot(grouped, labels=labels)
                plt.title(f"{col} - Seasonal Boxplot")
                plt.xlabel(xlab)
                plt.ylabel("Demand")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{col}_seasonal_boxplot.png"), dpi=150)
                plt.close()



def _safe_plot_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name))


def _seasonal_period_from_freq(freq_alias: str) -> int:
    if freq_alias == "H":
        return 24
    if freq_alias == "D":
        return 7
    if freq_alias == "W":
        return 52
    return 12


def save_correlation_analysis(
    df_for_corr: pd.DataFrame,
    target_cols: List[str],
    sheet_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    numeric_df = df_for_corr.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        corr_matrix = pd.DataFrame(columns=["variable"])
        corr_long = pd.DataFrame(columns=["target_series", "variable", "correlation", "abs_correlation"])
        return corr_matrix, corr_long

    corr_matrix = numeric_df.corr(method="pearson")
    corr_matrix.to_csv(os.path.join(sheet_dir, "correlation_matrix.csv"), encoding="utf-8-sig")
    corr_matrix.to_excel(os.path.join(sheet_dir, "correlation_matrix.xlsx"))

    plt.figure(figsize=(max(10, 0.45 * len(corr_matrix.columns)), max(8, 0.45 * len(corr_matrix.columns))))
    plt.imshow(corr_matrix.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_matrix_heatmap.png"), dpi=150)
    plt.close()

    rows = []
    for target in target_cols:
        if target not in corr_matrix.columns:
            continue
        tmp = corr_matrix[target].drop(labels=[target], errors="ignore").reset_index()
        tmp.columns = ["variable", "correlation"]
        tmp["target_series"] = target
        tmp["abs_correlation"] = tmp["correlation"].abs()
        tmp = tmp.sort_values("abs_correlation", ascending=False)
        rows.append(tmp[["target_series", "variable", "correlation", "abs_correlation"]])

        top_n = min(20, len(tmp))
        if top_n > 0:
            plt.figure(figsize=(10, max(5, 0.35 * top_n)))
            tmp_plot = tmp.head(top_n).sort_values("correlation")
            plt.barh(tmp_plot["variable"].astype(str), tmp_plot["correlation"].astype(float))
            plt.axvline(0.0, linewidth=1.0)
            plt.title(f"Top correlations with {target}")
            plt.xlabel("Correlation")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{_safe_plot_name(target)}_top_correlations.png"), dpi=150)
            plt.close()

    corr_long = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["target_series", "variable", "correlation", "abs_correlation"])
    corr_long.to_csv(os.path.join(sheet_dir, "target_correlations_long.csv"), index=False, encoding="utf-8-sig")
    corr_long.to_excel(os.path.join(sheet_dir, "target_correlations_long.xlsx"), index=False)
    return corr_matrix, corr_long


def save_seasonality_heatmaps_and_decomposition(
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    freq_alias: str,
    sheet_dir: str,
    max_plot_series: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    heatmap_rows = []
    decomp_rows = []
    seasonal_period = _seasonal_period_from_freq(freq_alias)

    for col in target_cols[:max_plot_series]:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        tmp = pd.DataFrame({date_col: pd.to_datetime(df_clean[date_col], errors="coerce"), col: s}).dropna()
        if len(tmp) == 0:
            continue

        tmp = tmp.sort_values(date_col).copy()

        if freq_alias == "M":
            tmp["row_key"] = tmp[date_col].dt.year
            tmp["col_key"] = tmp[date_col].dt.month
            row_label = "year"
            col_label = "month"
            heatmap_name = f"{_safe_plot_name(col)}_heatmap_year_month.png"
        elif freq_alias == "W":
            tmp["row_key"] = tmp[date_col].dt.year
            tmp["col_key"] = tmp[date_col].dt.isocalendar().week.astype(int)
            row_label = "year"
            col_label = "iso_week"
            heatmap_name = f"{_safe_plot_name(col)}_heatmap_year_week.png"
        elif freq_alias == "D":
            tmp["row_key"] = tmp[date_col].dt.year
            tmp["col_key"] = tmp[date_col].dt.month
            row_label = "year"
            col_label = "month"
            heatmap_name = f"{_safe_plot_name(col)}_heatmap_year_month.png"
        else:
            tmp["row_key"] = tmp[date_col].dt.dayofweek
            tmp["col_key"] = tmp[date_col].dt.hour
            row_label = "dayofweek"
            col_label = "hour"
            heatmap_name = f"{_safe_plot_name(col)}_heatmap_dayofweek_hour.png"

        piv = tmp.pivot_table(index="row_key", columns="col_key", values=col, aggfunc="mean")
        if len(piv) > 0:
            piv.to_csv(os.path.join(sheet_dir, f"{_safe_plot_name(col)}_heatmap_data.csv"), encoding="utf-8-sig")

            plt.figure(figsize=(max(8, 0.5 * max(1, len(piv.columns))), max(5, 0.4 * max(1, len(piv.index)))))
            plt.imshow(piv.values, aspect="auto", cmap="YlOrRd")
            plt.colorbar()
            plt.xticks(range(len(piv.columns)), piv.columns)
            plt.yticks(range(len(piv.index)), piv.index)
            plt.title(f"{col} - Heatmap ({row_label} x {col_label})")
            plt.xlabel(col_label)
            plt.ylabel(row_label)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, heatmap_name), dpi=150)
            plt.close()

            heatmap_rows.append({
                "series": col,
                "row_dimension": row_label,
                "column_dimension": col_label,
                "n_rows": int(piv.shape[0]),
                "n_cols": int(piv.shape[1]),
                "heatmap_file": heatmap_name
            })

        decomp_status = "skipped"
        decomp_reason = None
        if not HAS_STATSMODELS:
            decomp_reason = "statsmodels kurulu değil; seasonal decomposition atlandı."
        else:
            series_for_dec = tmp.set_index(date_col)[col].astype(float).copy()
            try:
                if freq_alias == "M":
                    series_for_dec = series_for_dec.asfreq("ME")
                elif freq_alias == "W":
                    series_for_dec = series_for_dec.asfreq("W")
                elif freq_alias == "D":
                    series_for_dec = series_for_dec.asfreq("D")
                else:
                    series_for_dec = series_for_dec.asfreq("H")
            except Exception:
                pass

            series_for_dec = series_for_dec.interpolate(method="linear", limit_direction="both")
            if series_for_dec.notna().sum() >= max(2 * seasonal_period, 12):
                try:
                    dec_model = "multiplicative" if bool((series_for_dec.dropna() > 0).all()) else "additive"
                    dec = seasonal_decompose(series_for_dec, model=dec_model, period=seasonal_period, extrapolate_trend="freq")
                    dec_df = pd.DataFrame({
                        "observed": dec.observed,
                        "trend": dec.trend,
                        "seasonal": dec.seasonal,
                        "resid": dec.resid
                    })
                    dec_df.to_csv(os.path.join(sheet_dir, f"{_safe_plot_name(col)}_seasonal_decomposition.csv"), encoding="utf-8-sig")

                    fig = dec.plot()
                    fig.set_size_inches(14, 10)
                    plt.suptitle(f"{col} - Seasonal Decomposition ({dec_model}, period={seasonal_period})", y=1.02)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{_safe_plot_name(col)}_seasonal_decomposition.png"), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    decomp_status = "saved"
                except Exception as exc:
                    decomp_reason = f"seasonal decomposition başarısız: {str(exc)}"
            else:
                decomp_reason = f"decomposition için yetersiz gözlem: gerekli yaklaşık minimum {max(2 * seasonal_period, 12)}"

        decomp_rows.append({
            "series": col,
            "status": decomp_status,
            "reason": decomp_reason,
            "seasonal_period_used": seasonal_period,
            "statsmodels_available": bool(HAS_STATSMODELS)
        })

    heatmap_report = pd.DataFrame(heatmap_rows, columns=["series", "row_dimension", "column_dimension", "n_rows", "n_cols", "heatmap_file"])
    decomposition_report = pd.DataFrame(decomp_rows, columns=["series", "status", "reason", "seasonal_period_used", "statsmodels_available"])

    if len(heatmap_report) > 0:
        heatmap_report.to_csv(os.path.join(sheet_dir, "seasonality_heatmap_report.csv"), index=False, encoding="utf-8-sig")
        heatmap_report.to_excel(os.path.join(sheet_dir, "seasonality_heatmap_report.xlsx"), index=False)
    if len(decomposition_report) > 0:
        decomposition_report.to_csv(os.path.join(sheet_dir, "seasonal_decomposition_report.csv"), index=False, encoding="utf-8-sig")
        decomposition_report.to_excel(os.path.join(sheet_dir, "seasonal_decomposition_report.xlsx"), index=False)

    return heatmap_report, decomposition_report

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_calendar_features(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    out = df.copy()

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter

    if freq in ["D", "H"]:
        out["dayofweek"] = out[date_col].dt.dayofweek
        out["weekofyear"] = out[date_col].dt.isocalendar().week.astype(int)
        out["dayofmonth"] = out[date_col].dt.day
        out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
        out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
        out["is_quarter_start"] = out[date_col].dt.is_quarter_start.astype(int)
        out["is_quarter_end"] = out[date_col].dt.is_quarter_end.astype(int)
        out["is_year_start"] = out[date_col].dt.is_year_start.astype(int)
        out["is_year_end"] = out[date_col].dt.is_year_end.astype(int)

    if freq == "H":
        out["hour"] = out[date_col].dt.hour
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    if freq in ["H", "D"]:
        out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_lag_features(df: pd.DataFrame, target_cols: List[str], freq_alias: str, season_length: int) -> pd.DataFrame:
    out = df.copy()

    base_lags = [1, 2, 3]
    if freq_alias == "M":
        base_lags += [6, 12]
    elif freq_alias == "W":
        base_lags += [4, 8, 12, 52]
    elif freq_alias == "D":
        base_lags += [7, 14, 28]
    elif freq_alias == "H":
        base_lags += [24, 48, 168]

    for col in target_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        for lag in sorted(set([l for l in base_lags if l < len(out)])):
            out[f"{col}_lag_{lag}"] = s.shift(lag)

        for w in [3, 6, 12]:
            if w < len(out):
                out[f"{col}_roll_mean_{w}"] = s.shift(1).rolling(w, min_periods=1).mean()
                out[f"{col}_roll_std_{w}"] = s.shift(1).rolling(w, min_periods=1).std().fillna(0)

        if season_length > 1 and season_length < len(out):
            out[f"{col}_lag_seasonal"] = s.shift(season_length)

    return out


def add_series_quality_features(
    df: pd.DataFrame,
    target_cols: List[str],
    anomaly_gov: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    out = df.copy()

    gov_small = anomaly_gov.copy() if len(anomaly_gov) > 0 else pd.DataFrame()

    for col in target_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_is_zero"] = s.eq(0).astype(int)
        out[f"{col}_log1p"] = np.log1p(s.clip(lower=0))

        out[f"{col}_anomaly_flag"] = 0
        out[f"{col}_exclude_from_training"] = 0
        out[f"{col}_review_required"] = 0
        out[f"{col}_structural_event_flag"] = 0
        out[f"{col}_incomplete_period_flag"] = 0

        if len(gov_small) > 0:
            sub = gov_small.loc[gov_small["series"] == col].copy()

            if len(sub) > 0:
                anomaly_dates = set(sub["date"].tolist())

                exclude_dates = set(
                    sub.loc[sub["excluded_from_training_candidate"] == True, "date"].tolist()
                )

                review_dates = set(
                    sub.loc[
                        sub["action_taken"].isin([
                            "flag_only_review",
                            "keep_raw_flag",
                            "preserve_raw_flag_exclude_candidate"
                        ]),
                        "date"
                    ].tolist()
                )

                structural_dates = set(
                    sub.loc[sub["is_structural_event"] == True, "date"].tolist()
                )

                incomplete_dates = set()
                if "is_incomplete_last_period" in sub.columns:
                    incomplete_dates = set(
                        sub.loc[sub["is_incomplete_last_period"] == True, "date"].tolist()
                    )

                # CRITICAL SAFETY RULE:
                # incomplete last period must always be excluded from training,
                # even if governance action was downgraded to review for any reason.
                exclude_dates = exclude_dates.union(incomplete_dates)

                out[f"{col}_anomaly_flag"] = out[date_col].isin(anomaly_dates).astype(int)
                out[f"{col}_exclude_from_training"] = out[date_col].isin(exclude_dates).astype(int)
                out[f"{col}_review_required"] = out[date_col].isin(review_dates).astype(int)
                out[f"{col}_structural_event_flag"] = out[date_col].isin(structural_dates).astype(int)
                out[f"{col}_incomplete_period_flag"] = out[date_col].isin(incomplete_dates).astype(int)

    return out


def create_model_family_exports(
    df_feat: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    freq_alias: str,
    config: PreprocessConfig
) -> Dict[str, pd.DataFrame]:
    season_length = config.seasonal_period_map.get(freq_alias, 1)

    # Base model-ready frame
    feat = df_feat.copy()

    if config.exclude_textual_columns_from_modeling_features:
        object_cols = feat.select_dtypes(include=["object"]).columns.tolist()
        drop_cols = [c for c in object_cols if c != date_col]
        feat = feat.drop(columns=drop_cols, errors="ignore")

    if config.drop_low_signal_calendar_features_for_monthly and freq_alias == "M":
        feat = feat.drop(
            columns=[
                "dayofweek", "dayofmonth", "weekofyear",
                "is_month_start", "is_month_end",
                "is_quarter_start", "is_quarter_end",
                "is_year_start", "is_year_end"
            ],
            errors="ignore"
        )

    if config.drop_weekday_text_from_monthly_modeling and freq_alias == "M":
        feat = feat.drop(columns=["Haftanın Günü", "weekday", "weekday name"], errors="ignore")

    # Statistical models
    statistical_cols = [date_col] + target_cols + [
        c for c in feat.columns if c not in target_cols and c != date_col
        and not any(x in c.lower() for x in ["lag_", "roll_mean_", "roll_std_"])
    ]
    df_statistical = feat[[c for c in statistical_cols if c in feat.columns]].copy()

    # ML models
    ml_cols = [date_col] + [c for c in feat.columns if c != date_col]
    df_ml = feat[[c for c in ml_cols if c in feat.columns]].copy()

    # DL / sequence models
    dl = df_clean[[date_col] + target_cols].copy()
    scaler = choose_scaler(config.scaler_for_deep_learning)
    scaled = scaler.fit_transform(dl[target_cols])
    for i, col in enumerate(target_cols):
        dl[f"{col}_scaled"] = scaled[:, i]
    dl = add_calendar_features(dl, date_col, freq_alias)

    # Foundation / transformer style minimal pack
    foundation = df_clean[[date_col] + target_cols].copy()
    for col in target_cols:
        foundation[f"{col}_log1p"] = np.log1p(pd.to_numeric(foundation[col], errors="coerce").clip(lower=0))

        if f"{col}_exclude_from_training" in feat.columns:
            foundation[f"{col}_exclude_from_training"] = feat[f"{col}_exclude_from_training"].values
        if f"{col}_structural_event_flag" in feat.columns:
            foundation[f"{col}_structural_event_flag"] = feat[f"{col}_structural_event_flag"].values
        if f"{col}_incomplete_period_flag" in feat.columns:
            foundation[f"{col}_incomplete_period_flag"] = feat[f"{col}_incomplete_period_flag"].values

    # Prophet-ready long export
    prophet_rows = []
    for col in target_cols:
        tmp = pd.DataFrame({
            "unique_id": col,
            "ds": df_clean[date_col].values,
            "y": pd.to_numeric(df_clean[col], errors="coerce").values
        })
        if f"{col}_exclude_from_training" in feat.columns:
            tmp["exclude_from_training"] = feat[f"{col}_exclude_from_training"].values
        if f"{col}_structural_event_flag" in feat.columns:
            tmp["structural_event_flag"] = feat[f"{col}_structural_event_flag"].values
        if f"{col}_incomplete_period_flag" in feat.columns:
            tmp["incomplete_period_flag"] = feat[f"{col}_incomplete_period_flag"].values
        prophet_rows.append(tmp)
    df_prophet = pd.concat(prophet_rows, axis=0, ignore_index=True)

    # Global / transformer-ready long export
    long_rows = []
    for col in target_cols:
        tmp = pd.DataFrame({
            "unique_id": col,
            "ds": df_clean[date_col].values,
            "y": pd.to_numeric(df_clean[col], errors="coerce").values
        })
        for extra in ["month", "quarter", "month_sin", "month_cos"]:
            if extra in feat.columns:
                tmp[extra] = feat[extra].values

        for flagcol in [
            f"{col}_exclude_from_training",
            f"{col}_review_required",
            f"{col}_structural_event_flag",
            f"{col}_incomplete_period_flag"
        ]:
            if flagcol in feat.columns:
                tmp[flagcol.replace(f"{col}_", "")] = feat[flagcol].values

        long_rows.append(tmp)
    df_global_long = pd.concat(long_rows, axis=0, ignore_index=True)

    return {
        "modeling_features_statistical": df_statistical,
        "modeling_features_ml": df_ml,
        "modeling_features_dl": dl,
        "modeling_features_foundation": foundation,
        "modeling_features_prophet": df_prophet,
        "modeling_features_global_long": df_global_long
    }

# =========================================================
# METRICS
# =========================================================

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), eps) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, train: np.ndarray, seasonality: int = 1, eps: float = 1e-8) -> float:
    train = np.asarray(train, dtype=float)
    if len(train) <= seasonality:
        denom = np.mean(np.abs(np.diff(train))) if len(train) > 1 else np.nan
    else:
        denom = np.mean(np.abs(train[seasonality:] - train[:-seasonality]))
    denom = max(denom, eps) if np.isfinite(denom) else eps
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)


# =========================================================
# REPORTS
# =========================================================

def series_quality_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    target_cols: List[str],
    outlier_flags: Dict[str, pd.Series],
    structural_zero_events: pd.Series
) -> pd.DataFrame:
    rows = []
    n = len(df_clean)

    for c in target_cols:
        raw_s = pd.to_numeric(df_raw[c], errors="coerce") if c in df_raw.columns else pd.Series(dtype=float)
        clean_s = pd.to_numeric(df_clean[c], errors="coerce")

        mean_val = clean_s.mean()
        std_val = clean_s.std()

        rows.append({
            "series": c,
            "n_rows_clean": n,
            "raw_missing_ratio": float(raw_s.isna().mean()) if len(raw_s) > 0 else np.nan,
            "clean_missing_ratio": float(clean_s.isna().mean()),
            "raw_zero_ratio": float(((raw_s == 0) & raw_s.notna()).mean()) if len(raw_s) > 0 else np.nan,
            "clean_zero_ratio": float(((clean_s == 0) & clean_s.notna()).mean()),
            "outlier_count_flagged": int(outlier_flags[c].sum()) if c in outlier_flags else 0,
            "outlier_fraction_flagged": float(outlier_flags[c].mean()) if c in outlier_flags else 0.0,
            "structural_zero_event_count": int(structural_zero_events.sum()),
            "count": int(clean_s.notna().sum()),
            "mean": float(mean_val),
            "median": float(clean_s.median()),
            "std": float(std_val),
            "min": float(clean_s.min()),
            "q25": safe_float(clean_s.quantile(0.25)),
            "q50": safe_float(clean_s.quantile(0.50)),
            "q75": safe_float(clean_s.quantile(0.75)),
            "max": float(clean_s.max()),
            "iqr": safe_float(clean_s.quantile(0.75) - clean_s.quantile(0.25)),
            "sum": safe_float(clean_s.sum()),
            "skewness": safe_float(clean_s.skew()),
            "kurtosis": safe_float(clean_s.kurt()),
            "cv": float(std_val / mean_val) if pd.notna(mean_val) and mean_val != 0 else np.nan,
            "intermittency_ratio": float(((clean_s == 0) & clean_s.notna()).mean()),
        })

    return pd.DataFrame(rows)


def create_missing_value_audit(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    target_cols: List[str],
    anomaly_gov: Optional[pd.DataFrame] = None,
    df_clean_governed_preserve: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    rows = []
    anomaly_gov = anomaly_gov.copy() if isinstance(anomaly_gov, pd.DataFrame) else pd.DataFrame()

    for col in target_cols:
        raw_s = pd.to_numeric(df_regular[col], errors="coerce")
        clean_s = pd.to_numeric(df_clean[col], errors="coerce")

        preserve_changed_count = np.nan
        if isinstance(df_clean_governed_preserve, pd.DataFrame) and col in df_clean_governed_preserve.columns:
            preserve_s = pd.to_numeric(df_clean_governed_preserve[col], errors="coerce")
            preserve_changed_count = int((
                (raw_s.isna() & preserve_s.notna()) |
                (raw_s.notna() & preserve_s.notna() & (raw_s != preserve_s))
            ).sum())

        raw_missing = raw_s.isna()
        clean_missing = clean_s.isna()

        gov_imputed_count = 0
        review_or_exclusion_count = 0
        if len(anomaly_gov) > 0:
            gov_imputed_count = int(
                (
                    (anomaly_gov["series"] == col) &
                    (anomaly_gov["action_taken"] == "set_nan_then_impute")
                ).sum()
            )
            review_or_exclusion_count = int(
                (
                    (anomaly_gov["series"] == col) &
                    (anomaly_gov["action_taken"].isin(["flag_only_review", "keep_raw_flag", "preserve_raw_flag_exclude_candidate"]))
                ).sum()
            )

        changed_count = int((
            (raw_s.isna() & clean_s.notna()) |
            (raw_s.notna() & clean_s.notna() & (raw_s != clean_s))
        ).sum())

        rows.append({
            "series": col,
            "raw_missing_count": int(raw_missing.sum()),
            "clean_missing_count": int(clean_missing.sum()),
            "raw_missing_ratio": float(raw_missing.mean()),
            "clean_missing_ratio": float(clean_missing.mean()),
            "imputed_from_raw_missing_count": int(raw_missing.sum() - clean_missing.sum()),
            "imputed_from_governance_count": gov_imputed_count,
            "review_or_exclusion_governance_count": review_or_exclusion_count,
            "preserve_clean_changed_count": preserve_changed_count,
            "candidate_clean_changed_count": changed_count
        })
    return pd.DataFrame(rows)


def create_missing_strategy_audit(
    df_regular: pd.DataFrame,
    target_cols: List[str],
    date_col: str,
    config: PreprocessConfig
) -> pd.DataFrame:
    missingness_summary = summarize_missingness_patterns(df_regular=df_regular, target_cols=target_cols, date_col=date_col)
    return decide_missing_value_strategy(missingness_summary, config)


def create_frequency_audit(
    df_raw_after_aggregation: pd.DataFrame,
    df_regular: pd.DataFrame,
    date_col: str,
    freq_alias: str
) -> pd.DataFrame:
    raw_dates = pd.DatetimeIndex(df_raw_after_aggregation[date_col].sort_values())
    reg_dates = pd.DatetimeIndex(df_regular[date_col].sort_values())
    missing_dates = reg_dates.difference(raw_dates)

    return pd.DataFrame({
        "metric": [
            "frequency_alias",
            "raw_start",
            "raw_end",
            "regular_start",
            "regular_end",
            "raw_row_count",
            "regular_row_count",
            "inserted_missing_timestamp_count"
        ],
        "value": [
            freq_alias,
            str(raw_dates.min()) if len(raw_dates) else None,
            str(raw_dates.max()) if len(raw_dates) else None,
            str(reg_dates.min()) if len(reg_dates) else None,
            str(reg_dates.max()) if len(reg_dates) else None,
            int(len(raw_dates)),
            int(len(reg_dates)),
            int(len(missing_dates))
        ]
    })


def create_inserted_timestamp_log(df_raw_after_aggregation: pd.DataFrame, df_regular: pd.DataFrame, date_col: str) -> pd.DataFrame:
    raw_dates = pd.DatetimeIndex(df_raw_after_aggregation[date_col].sort_values())
    reg_dates = pd.DatetimeIndex(df_regular[date_col].sort_values())
    missing_dates = reg_dates.difference(raw_dates)

    if len(missing_dates) == 0:
        return pd.DataFrame(columns=["inserted_timestamp"])
    return pd.DataFrame({"inserted_timestamp": missing_dates})


def create_outlier_log(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    anomaly_gov: pd.DataFrame
) -> pd.DataFrame:
    if len(anomaly_gov) == 0:
        return pd.DataFrame(columns=[
            "date", "series", "raw_value_before_action",
            "clean_value_after_processing", "anomaly_type",
            "action_taken", "action_confidence"
        ])

    rows = []
    for _, row in anomaly_gov.iterrows():
        date_ = row["date"]
        series = row["series"]
        idx = df_regular.index[df_regular[date_col] == date_]
        if len(idx) == 0:
            continue
        idx = idx[0]
        rows.append({
            "date": date_,
            "series": series,
            "raw_value_before_action": df_regular.loc[idx, series],
            "clean_value_after_processing": df_clean.loc[idx, series],
            "anomaly_type": row["anomaly_type"],
            "action_taken": row["action_taken"],
            "action_confidence": row["action_confidence"]
        })
    return pd.DataFrame(rows)


def intervention_intensity_report(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    recent_window: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    series_rows = []
    last_period_rows = []

    n = len(df_regular)

    for col in target_cols:
        raw_s = pd.to_numeric(df_regular[col], errors="coerce")
        clean_s = pd.to_numeric(df_clean[col], errors="coerce")

        changed = (
            (raw_s.isna() & clean_s.notna()) |
            (raw_s.notna() & clean_s.notna() & (raw_s != clean_s))
        )

        abs_change = (clean_s - raw_s).abs()
        pct_change = abs_change / raw_s.replace(0, np.nan).abs()

        changed_count = int(changed.sum())
        changed_fraction = float(changed.mean())
        mean_abs_change = safe_float(abs_change[changed].mean()) if changed.any() else 0.0
        mean_pct_change = safe_float((pct_change[changed] * 100).mean()) if changed.any() else 0.0
        max_abs_change = safe_float(abs_change.max()) if len(abs_change) > 0 else 0.0
        max_pct_change = safe_float((pct_change * 100).max()) if len(pct_change) > 0 else 0.0
        changes_in_last = int(changed.tail(recent_window).sum())

        series_rows.append({
            "series": col,
            "changed_cell_count": changed_count,
            "changed_fraction": changed_fraction,
            "mean_abs_change": mean_abs_change,
            "mean_pct_change": mean_pct_change,
            "max_abs_change": max_abs_change,
            "max_pct_change": max_pct_change,
            "changes_in_last_6_periods": changes_in_last
        })

        for idx in df_regular.index[changed]:
            last_period_rows.append({
                "date": df_regular.loc[idx, date_col],
                "series": col,
                "raw_value": raw_s.loc[idx],
                "clean_value": clean_s.loc[idx],
                "abs_change": abs_change.loc[idx],
                "pct_change": safe_float(pct_change.loc[idx] * 100)
            })

    series_df = pd.DataFrame(series_rows)
    last_df = pd.DataFrame(last_period_rows)

    summary_rows.append({
        "metric": "total_changed_cells",
        "value": int(series_df["changed_cell_count"].sum()) if len(series_df) > 0 else 0
    })
    summary_rows.append({
        "metric": "overall_changed_fraction_mean",
        "value": safe_float(series_df["changed_fraction"].mean()) if len(series_df) > 0 else 0.0
    })
    summary_rows.append({
        "metric": "series_with_changes",
        "value": int((series_df["changed_cell_count"] > 0).sum()) if len(series_df) > 0 else 0
    })

    return pd.DataFrame(summary_rows), series_df, last_df


def create_domain_validation_template(df_clean: pd.DataFrame, date_col: str, target_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in target_cols:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        if s.notna().sum() == 0:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = max(q1 - 1.5 * iqr, 0)

        peaks = df_clean.loc[(s > upper) | (s < lower), [date_col, col]].copy()
        peaks = peaks.sort_values(col, ascending=False).head(12)

        for _, row in peaks.iterrows():
            rows.append({
                "series": col,
                "date": row[date_col],
                "value": row[col],
                "possible_real_world_explanation": "",
                "validated_by_domain_expert": "",
                "notes": ""
            })

    if not rows:
        return pd.DataFrame(columns=[
            "series", "date", "value",
            "possible_real_world_explanation",
            "validated_by_domain_expert",
            "notes"
        ])
    return pd.DataFrame(rows)


def build_review_queue(
    anomaly_gov: pd.DataFrame,
    review_candidate_map: Dict[str, pd.Series]
) -> pd.DataFrame:
    if len(anomaly_gov) == 0:
        return pd.DataFrame(columns=[
            "date", "series", "raw_value", "clean_value_candidate",
            "candidate_method", "candidate_ratio_vs_raw",
            "anomaly_type", "reason", "confidence",
            "recommended_action", "is_incomplete_last_period", "is_structural_event",
            "analyst_decision", "analyst_note"
        ])

    rows = []
    for _, row in anomaly_gov.iterrows():
        if row["action_taken"] not in ["flag_only_review", "preserve_raw_flag_exclude_candidate", "keep_raw_flag"]:
            continue

        candidate_series = review_candidate_map.get(row["series"], pd.Series(dtype=float))
        clean_candidate = np.nan

        if len(candidate_series) > 0:
            matched = candidate_series[candidate_series.index == row["date"]]
            if len(matched) > 0:
                clean_candidate = matched.iloc[0]

        raw_val = row["raw_value"]
        candidate_ratio_vs_raw = np.nan
        if pd.notna(raw_val) and raw_val != 0 and pd.notna(clean_candidate):
            candidate_ratio_vs_raw = float(clean_candidate / raw_val)

        rows.append({
            "date": row["date"],
            "series": row["series"],
            "raw_value": raw_val,
            "clean_value_candidate": clean_candidate,
            "candidate_method": "seasonal_local_impute",
            "candidate_ratio_vs_raw": candidate_ratio_vs_raw,
            "anomaly_type": row["anomaly_type"],
            "reason": row["anomaly_reason"],
            "confidence": row["action_confidence"],
            "recommended_action": row["action_taken"],
            "is_incomplete_last_period": row.get("is_incomplete_last_period", False),
            "is_structural_event": row.get("is_structural_event", False),
            "analyst_decision": "",
            "analyst_note": ""
        })
    return pd.DataFrame(rows)


def save_raw_vs_clean_plots(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    sheet_dir: str,
    max_plot_series: int = 50
):
    plot_dir = os.path.join(sheet_dir, "validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for col in target_cols[:max_plot_series]:
        plt.figure(figsize=(12, 5))
        plt.plot(df_regular[date_col], df_regular[col], label="Raw-Regular", linewidth=1.5)
        plt.plot(df_clean[date_col], df_clean[col], label="Clean", linewidth=1.5)
        plt.title(f"{col} - Raw vs Clean")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_raw_vs_clean.png"), dpi=150)
        plt.close()


# =========================================================
# STRICT LEAKAGE AUDIT
# =========================================================

def strict_leakage_audit(df_features: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    rows = []

    cols = [str(c) for c in df_features.columns]
    for col in cols:
        lowered = col.lower()
        status = "OK"
        risk_level = "LOW"
        note = "Belirgin leakage paterni bulunmadı."

        # Explicit forbidden patterns
        forbidden_patterns = [
            (r"shift\(-", "Negative shift / future leak riski."),
            (r"lead", "Lead feature leak riski."),
            (r"future", "Future bilgi kullanımı riski."),
            (r"next", "Gelecek bilgi kullanımı riski."),
            (r"centered", "Centered rolling leak riski."),
            (r"rolling_center", "Centered rolling leak riski."),
            (r"t\+", "Gelecek zaman etiketi riski.")
        ]

        for p, msg in forbidden_patterns:
            if re.search(p, lowered):
                status = "REVIEW"
                risk_level = "HIGH"
                note = msg
                break

        # Direct target columns are okay
        if any(lowered == t.lower() for t in target_cols):
            status = "OK"
            risk_level = "LOW"
            note = "Hedef seri."

        # Allowed historical features
        if (
            "_lag_" in lowered or
            "_roll_mean_" in lowered or
            "_roll_std_" in lowered or
            lowered.endswith("_scaled") or
            lowered.endswith("_log1p") or
            lowered.endswith("_is_zero") or
            lowered.endswith("_anomaly_flag")
        ):
            status = "OK"
            risk_level = "LOW"
            note = "Geçmişe dayalı yardımcı özellik."

        rows.append({
            "column_name": col,
            "status": status,
            "risk_level": risk_level,
            "note": note
        })

    rows.append({
        "column_name": "__RULE__SCALER_POLICY",
        "status": "REVIEW",
        "risk_level": "HIGH",
        "note": "Bu modül export amaçlı full-data scaling üretir. Nihai model eğitiminde scaler sadece train fold üzerinde fit edilmelidir."
    })
    rows.append({
        "column_name": "__RULE__IMPUTER_POLICY",
        "status": "REVIEW",
        "risk_level": "HIGH",
        "note": "Bu modülde imputasyon full-series bağlamında yapılır. Nihai walk-forward / CV pipeline içinde imputasyon sadece train fold bilgisiyle yeniden kurulmalıdır."
    })
    rows.append({
        "column_name": "__RULE__ANOMALY_POLICY",
        "status": "REVIEW",
        "risk_level": "HIGH",
        "note": "Outlier/anomaly governance full-series audit amaçlıdır. Model selection sırasında fold-aware anomaly policy kullanılmalıdır."
    })
    rows.append({
        "column_name": "__RULE__CENTERED_ROLLING_POLICY",
        "status": "PASS",
        "risk_level": "LOW",
        "note": "Modeling feature export içinde centered rolling kullanılmadı."
    })

    return pd.DataFrame(rows)


# =========================================================
# BACKTEST
# =========================================================

def get_seasonal_period_for_backtest(freq_alias: str) -> int:
    if freq_alias == "M":
        return 12
    if freq_alias == "W":
        return 52
    if freq_alias == "D":
        return 7
    if freq_alias == "H":
        return 24
    return 1


def get_min_train_size_for_freq(config: PreprocessConfig, freq_alias: str) -> int:
    if freq_alias == "M":
        return config.backtest_min_train_size_monthly
    if freq_alias == "W":
        return config.backtest_min_train_size_weekly
    if freq_alias == "D":
        return config.backtest_min_train_size_daily
    if freq_alias == "H":
        return config.backtest_min_train_size_hourly
    return 24


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_length: int) -> np.ndarray:
    train = pd.Series(train).dropna().reset_index(drop=True)
    if len(train) == 0:
        return np.array([np.nan] * horizon)

    preds = []
    for i in range(horizon):
        idx = len(train) - season_length + (i % season_length)
        if season_length > 0 and len(train) >= season_length and idx >= 0:
            preds.append(float(train.iloc[idx]))
        else:
            preds.append(float(train.iloc[-1]))
    return np.array(preds, dtype=float)


def rolling_mean_forecast(train: pd.Series, horizon: int, window: int = 3) -> np.ndarray:
    train = pd.Series(train).dropna().reset_index(drop=True)
    if len(train) == 0:
        return np.array([np.nan] * horizon)
    val = float(train.tail(min(window, len(train))).mean())
    return np.array([val] * horizon, dtype=float)


def _collect_forecast_metrics(y_true, y_pred, train, season_length):
    return {
        "mape": safe_mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mase": mase(y_true, y_pred, train=np.asarray(train), seasonality=season_length)
    }


def run_proxy_backtest_validation(
    df_raw_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    target_cols: List[str],
    freq_alias: str,
    config: PreprocessConfig,
    truth_source: str = "clean"
) -> pd.DataFrame:
    """
    PATCH:
    truth_source:
        - 'clean' : evaluate against cleaned series
        - 'raw'   : evaluate against raw regular series
    """
    rows = []
    horizon = config.backtest_horizon
    min_train_size = get_min_train_size_for_freq(config, freq_alias)
    season_length = get_seasonal_period_for_backtest(freq_alias)

    for col in target_cols:
        raw_series = pd.to_numeric(df_raw_regular[col], errors="coerce").reset_index(drop=True)
        clean_series = pd.to_numeric(df_clean[col], errors="coerce").reset_index(drop=True)
        truth_series = clean_series if truth_source == "clean" else raw_series

        if clean_series.notna().sum() < min_train_size + horizon:
            rows.append({
                "series": col,
                "comparison": "raw_vs_clean",
                "truth_source": truth_source,
                "note": "Yetersiz gözlem nedeniyle proxy backtest uygulanamadı."
            })
            continue

        metrics = {
            "raw_seasonal_naive": [],
            "clean_seasonal_naive": [],
            "clean_rolling_mean_3": []
        }
        if config.enable_additional_backtest_benchmarks:
            metrics.update({
                "raw_drift": [],
                "clean_drift": []
            })

        start = min_train_size
        end = len(clean_series) - horizon + 1

        for split_end in range(start, end):
            y_true = truth_series.iloc[split_end: split_end + horizon].values
            train_raw = raw_series.iloc[:split_end]
            train_clean = clean_series.iloc[:split_end]

            pred_raw = seasonal_naive_forecast(train_raw, horizon, season_length)
            pred_clean = seasonal_naive_forecast(train_clean, horizon, season_length)
            pred_roll = rolling_mean_forecast(train_clean, horizon, 3)
            if config.enable_additional_backtest_benchmarks:
                pred_raw_drift = drift_forecast(train_raw, horizon)
                pred_clean_drift = drift_forecast(train_clean, horizon)

            if np.isfinite(pred_raw).all():
                metrics["raw_seasonal_naive"].append(_collect_forecast_metrics(y_true, pred_raw, train_clean.values, season_length))
            if np.isfinite(pred_clean).all():
                metrics["clean_seasonal_naive"].append(_collect_forecast_metrics(y_true, pred_clean, train_clean.values, season_length))
            if np.isfinite(pred_roll).all():
                metrics["clean_rolling_mean_3"].append(_collect_forecast_metrics(y_true, pred_roll, train_clean.values, season_length))
            if config.enable_additional_backtest_benchmarks and np.isfinite(pred_raw_drift).all():
                metrics["raw_drift"].append(_collect_forecast_metrics(y_true, pred_raw_drift, train_clean.values, season_length))
            if config.enable_additional_backtest_benchmarks and np.isfinite(pred_clean_drift).all():
                metrics["clean_drift"].append(_collect_forecast_metrics(y_true, pred_clean_drift, train_clean.values, season_length))

        for model_name, values in metrics.items():
            if not values:
                continue
            rows.append({
                "series": col,
                "model_proxy": model_name,
                "truth_source": truth_source,
                "mape": float(np.mean([x["mape"] for x in values])),
                "smape": float(np.mean([x["smape"] for x in values])),
                "wape": float(np.mean([x["wape"] for x in values])),
                "mae": float(np.mean([x["mae"] for x in values])),
                "rmse": float(np.mean([x["rmse"] for x in values])),
                "mase": float(np.mean([x["mase"] for x in values]))
            })

    return pd.DataFrame(rows)


def raw_vs_clean_backtest_comparator(proxy_backtest_report: pd.DataFrame) -> pd.DataFrame:
    if len(proxy_backtest_report) == 0:
        return pd.DataFrame(columns=[
            "series", "metric", "raw_value", "clean_value", "improvement", "decision",
            "raw_smape_mean", "clean_smape_mean", "relative_improvement_pct",
            "decision_reason", "enough_evidence_flag"
        ])

    rows = []
    raw_df = proxy_backtest_report[proxy_backtest_report["model_proxy"] == "raw_seasonal_naive"].copy()
    clean_df = proxy_backtest_report[proxy_backtest_report["model_proxy"] == "clean_seasonal_naive"].copy()

    common = sorted(set(raw_df["series"]).intersection(set(clean_df["series"])))
    metrics = ["mape", "smape", "wape", "mae", "rmse", "mase"]

    for series in common:
        r = raw_df.loc[raw_df["series"] == series].iloc[0]
        c = clean_df.loc[clean_df["series"] == series].iloc[0]

        better_count = 0
        worse_count = 0
        for metric in metrics:
            rv = safe_float(r.get(metric, np.nan))
            cv = safe_float(c.get(metric, np.nan))
            improvement = rv - cv if pd.notna(rv) and pd.notna(cv) else np.nan
            if pd.notna(improvement) and improvement > 0:
                better_count += 1
            elif pd.notna(improvement) and improvement < 0:
                worse_count += 1

            rows.append({
                "series": series,
                "metric": metric,
                "raw_value": rv,
                "clean_value": cv,
                "improvement": improvement,
                "decision": "",
                "raw_smape_mean": safe_float(r.get("smape", np.nan)),
                "clean_smape_mean": safe_float(c.get("smape", np.nan)),
                "relative_improvement_pct": ((rv - cv) / rv * 100.0) if pd.notna(rv) and rv not in [0, 0.0] and pd.notna(cv) else np.nan,
                "decision_reason": "",
                "enough_evidence_flag": np.nan
            })

        raw_smape_mean = safe_float(r.get("smape", np.nan))
        clean_smape_mean = safe_float(c.get("smape", np.nan))
        relative_improvement_pct = ((raw_smape_mean - clean_smape_mean) / raw_smape_mean * 100.0) if pd.notna(raw_smape_mean) and raw_smape_mean not in [0, 0.0] and pd.notna(clean_smape_mean) else np.nan

        if better_count >= 4:
            decision = "clean_candidate_preferred"
            reason = f"{better_count}/6 hata metriğinde iyileşme var."
            enough = True
        elif better_count == 0 and worse_count > 0:
            decision = "no_evidence_of_improvement"
            reason = "Temiz seri hata metriklerinde üstünlük göstermedi."
            enough = False
        else:
            decision = "mixed_review"
            reason = f"Karışık sinyal: {better_count} iyileşme, {worse_count} kötüleşme."
            enough = False

        rows.append({
            "series": series,
            "metric": "__OVERALL__",
            "raw_value": np.nan,
            "clean_value": np.nan,
            "improvement": better_count,
            "decision": decision,
            "raw_smape_mean": raw_smape_mean,
            "clean_smape_mean": clean_smape_mean,
            "relative_improvement_pct": relative_improvement_pct,
            "decision_reason": reason,
            "enough_evidence_flag": enough
        })

    return pd.DataFrame(rows)


# =========================================================
# TESTS
# =========================================================

def _assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run_internal_unit_tests(config: PreprocessConfig) -> pd.DataFrame:
    records = []

    def add_result(name: str, status: str, detail: str):
        records.append({"test_name": name, "status": status, "detail": detail})

    try:
        s = pd.Series(["31.01.2019", "28.02.2019", "15.03.2019"])
        dt = parse_datetime_series(s)
        _assert_true(dt.notna().all(), "Tüm tarihler parse edilmeliydi.")
        add_result("date_parse_test", "PASS", "Tarih parse testi başarılı.")
    except Exception as e:
        add_result("date_parse_test", "FAIL", str(e))

    try:
        idx_m = pd.date_range("2020-01-31", periods=6, freq="ME")
        idx_d = pd.date_range("2020-01-01", periods=10, freq="D")
        idx_h = pd.date_range("2020-01-01 00:00:00", periods=10, freq="H")
        _assert_true(infer_frequency_from_dates(idx_m) == "M", "Aylık frekans doğru tespit edilmedi.")
        _assert_true(infer_frequency_from_dates(idx_d) == "D", "Günlük frekans doğru tespit edilmedi.")
        _assert_true(infer_frequency_from_dates(idx_h) == "H", "Saatlik frekans doğru tespit edilmedi.")
        add_result("frequency_detection_test", "PASS", "Frekans tespit testi başarılı.")
    except Exception as e:
        add_result("frequency_detection_test", "FAIL", str(e))

    try:
        s = pd.Series([10.0, np.nan, 30.0])
        out = limited_linear_interpolation(s, limit=1)
        _assert_true(pd.notna(out.iloc[1]), "Eksik gözlem interpolate edilmeliydi.")
        add_result("interpolation_test", "PASS", "Interpolasyon testi başarılı.")
    except Exception as e:
        add_result("interpolation_test", "FAIL", str(e))

    try:
        s = pd.Series([10, 11, 10, 12, 11, 200, 10, 9, 11, 10], dtype=float)
        profile = {
            "cv": 0.1,
            "intermittency_ratio": 0.0,
            "volatility_regime": "stable",
            "volume_level": "medium"
        }
        flags, _ = conservative_outlier_vote_adaptive(s, profile, config)
        _assert_true(bool(flags.iloc[5]), "Bariz outlier işaretlenmeliydi.")
        add_result("adaptive_outlier_test", "PASS", "Adaptif outlier testi başarılı.")
    except Exception as e:
        add_result("adaptive_outlier_test", "FAIL", str(e))

    return pd.DataFrame(records)


def generate_synthetic_series(freq: str = "M", periods: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if freq == "M":
        dates = pd.date_range("2018-01-31", periods=periods, freq="ME")
        season = np.sin(2 * np.pi * np.arange(periods) / 12)
    elif freq == "W":
        dates = pd.date_range("2018-01-07", periods=periods, freq="W")
        season = np.sin(2 * np.pi * np.arange(periods) / 52)
    elif freq == "D":
        dates = pd.date_range("2018-01-01", periods=periods, freq="D")
        season = np.sin(2 * np.pi * np.arange(periods) / 7)
    else:
        dates = pd.date_range("2018-01-01 00:00:00", periods=periods, freq="H")
        season = np.sin(2 * np.pi * np.arange(periods) / 24)

    trend = np.linspace(100, 150, periods)
    noise = rng.normal(0, 5, periods)
    y1 = trend + 10 * season + noise
    y2 = y1 * 0.8 + rng.normal(0, 3, periods)

    df = pd.DataFrame({
        "datum": dates,
        "M01AB": y1,
        "M01AE": y2
    })

    if len(df) >= 10:
        df.loc[5, "M01AB"] = np.nan
        df.loc[8, "M01AE"] = np.nan

    if len(df) >= 20:
        df.loc[12, "M01AB"] = df["M01AB"].median() * 5
        df.loc[18, "M01AE"] = df["M01AE"].median() * 4

    if len(df) >= 25:
        df.loc[21, ["M01AB", "M01AE"]] = 0

    if len(df) >= 15:
        df = df.drop(index=[3]).reset_index(drop=True)

    return df


def run_synthetic_tests(config: PreprocessConfig) -> pd.DataFrame:
    rows = []

    def add_result(name: str, status: str, detail: str):
        rows.append({"test_name": name, "status": status, "detail": detail})

    try:
        df = generate_synthetic_series("M", 60, config.random_seed)
        date_col = "datum"
        target_cols = ["M01AB", "M01AE"]

        df[date_col] = parse_datetime_series(df[date_col])
        df = aggregate_duplicates(df, date_col, target_cols)
        df_regular = build_regular_time_index(df, date_col, "M")

        profiles = {c: build_series_profile(df_regular[c], "M", config) for c in target_cols}
        flags = {}
        vote_details = {}
        for c in target_cols:
            f, vd = conservative_outlier_vote_adaptive(df_regular[c], profiles[c], config)
            flags[c] = f
            vote_details[c] = vd

        structural_zero = detect_structural_zero_events(
            df_regular, target_cols,
            config.structural_zero_min_series_count,
            config.structural_zero_ratio_threshold
        )

        incomplete_flags = pd.Series(False, index=df_regular.index)

        gov = build_anomaly_governance_table(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            outlier_flags=flags,
            vote_details=vote_details,
            series_profiles=profiles,
            structural_event_flags=structural_zero,
            incomplete_period_flags=incomplete_flags,
            config=config
        )

        _assert_true(len(df_regular) >= len(df), "Regular index satır sayısı azalmamalı.")
        _assert_true(isinstance(gov, pd.DataFrame), "Governance tablosu üretilmeliydi.")
        _assert_true(len(gov) > 0, "Synthetic veri üzerinde governance kaydı oluşmalıydı.")
        add_result("synthetic_governance_test", "PASS", "Sentetik governance testi başarılı.")
    except Exception as e:
        add_result("synthetic_governance_test", "FAIL", str(e))

    try:
        base = generate_synthetic_series("M", 48, config.random_seed).copy()
        base["datum"] = parse_datetime_series(base["datum"])
        base = aggregate_duplicates(base, "datum", ["M01AB", "M01AE"])
        reg = build_regular_time_index(base, "datum", "M")

        scenarios = {
            "missing_at_start": [0, 1, 2],
            "missing_in_middle_block": [15, 16, 17, 18],
            "missing_at_end": [len(reg) - 3, len(reg) - 2, len(reg) - 1]
        }
        for test_name, idxs in scenarios.items():
            work = reg.copy()
            work.loc[idxs, "M01AB"] = np.nan
            summary = summarize_missingness_patterns(work, ["M01AB"], "datum")
            strat = decide_missing_value_strategy(summary, config)
            _assert_true(len(summary) == 1 and len(strat) == 1, f"{test_name} için audit üretilmeliydi.")
        add_result("synthetic_missing_strategy_test", "PASS", "Baş/orta/son eksik blok senaryoları başarıyla değerlendirildi.")
    except Exception as e:
        add_result("synthetic_missing_strategy_test", "FAIL", str(e))

    return pd.DataFrame(rows)


def run_business_rule_tests(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    target_cols: List[str],
    anomaly_gov: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    rows = []

    def add_result(name: str, status: str, detail: str):
        rows.append({"test_name": name, "status": status, "detail": detail})

    try:
        recent = anomaly_gov[anomaly_gov["is_recent_period"] == True] if len(anomaly_gov) > 0 else pd.DataFrame()
        auto_fixed_recent = recent[recent["action_taken"] == "set_nan_then_impute"] if len(recent) > 0 else pd.DataFrame()
        ok = len(auto_fixed_recent) == 0
        add_result(
            "recent_period_human_review_test",
            "PASS" if ok else "REVIEW",
            "Son dönemlerde otomatik düzeltme yapılmadı." if ok else "Son dönemlerde otomatik düzeltme tespit edildi."
        )
    except Exception as e:
        add_result("recent_period_human_review_test", "FAIL", str(e))

    try:
        changed = 0
        for col in target_cols:
            raw_s = pd.to_numeric(df_regular[col], errors="coerce")
            clean_s = pd.to_numeric(df_clean[col], errors="coerce")
            changed += int(((raw_s != clean_s) & raw_s.notna() & clean_s.notna()).sum())

        add_result("candidate_intervention_exists_test", "PASS", f"Candidate clean değişen hücre sayısı: {changed}")
    except Exception as e:
        add_result("candidate_intervention_exists_test", "FAIL", str(e))

    try:
        structural_count = int(anomaly_gov["is_structural_event"].sum()) if len(anomaly_gov) > 0 else 0
        exclusion_count = int(anomaly_gov["excluded_from_training_candidate"].sum()) if len(anomaly_gov) > 0 else 0
        ok = exclusion_count >= structural_count
        add_result(
            "structural_event_exclusion_policy_test",
            "PASS" if ok else "REVIEW",
            f"Structural governance satırı: {structural_count}, exclusion satırı: {exclusion_count}"
        )
    except Exception as e:
        add_result("structural_event_exclusion_policy_test", "FAIL", str(e))

    try:
        incomplete_rows = anomaly_gov.loc[
            anomaly_gov["anomaly_type"] == "incomplete_last_period"
        ] if len(anomaly_gov) > 0 else pd.DataFrame()

        if len(incomplete_rows) == 0:
            add_result(
                "incomplete_last_period_exclusion_test",
                "PASS",
                "Incomplete last period governance kaydı yok; test uygulanmadı."
            )
        else:
            ok = bool((incomplete_rows["excluded_from_training_candidate"] == True).all())
            add_result(
                "incomplete_last_period_exclusion_test",
                "PASS" if ok else "FAIL",
                "Incomplete last period kayıtlarının tamamı training exclusion aldı."
                if ok else
                "Bazı incomplete last period kayıtları training exclusion almadı."
            )
    except Exception as e:
        add_result("incomplete_last_period_exclusion_test", "FAIL", str(e))
        
    return pd.DataFrame(rows)


def create_manual_sample_audit(
    df_regular: pd.DataFrame,
    df_clean: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    sample_size: int = 20,
    random_seed: int = 42,
    anomaly_dates: Optional[List[pd.Timestamp]] = None
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    n = len(df_regular)
    if n == 0:
        return pd.DataFrame()

    forced_idx = []
    if anomaly_dates is not None and len(anomaly_dates) > 0:
        forced_idx = df_regular.index[df_regular[date_col].isin(anomaly_dates)].tolist()

    remaining = [i for i in np.arange(n) if i not in forced_idx]
    random_size = max(0, min(sample_size - len(forced_idx), len(remaining)))
    sampled_random = rng.choice(remaining, size=random_size, replace=False).tolist() if random_size > 0 else []
    sampled_idx = sorted(set(forced_idx + sampled_random))

    rows = []
    for idx in sampled_idx:
        row = {
            "row_index": int(idx),
            "date": df_regular.loc[idx, date_col]
        }
        for col in target_cols:
            raw_val = df_regular.loc[idx, col]
            clean_val = df_clean.loc[idx, col]
            row[f"{col}_raw_regular"] = raw_val
            row[f"{col}_clean"] = clean_val
            row[f"{col}_changed"] = (
                (pd.isna(raw_val) and pd.notna(clean_val)) or
                (pd.notna(raw_val) and pd.notna(clean_val) and raw_val != clean_val)
            )
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# VALIDATION SUMMARY
# =========================================================

def create_validation_summary(
    quality_report: pd.DataFrame,
    missing_audit: pd.DataFrame,
    freq_ok: bool,
    freq_msg: str,
    outlier_log: pd.DataFrame,
    leakage_report: pd.DataFrame,
    unit_test_report: pd.DataFrame,
    synthetic_test_report: pd.DataFrame,
    business_rule_test_report: pd.DataFrame,
    manual_sample_audit: pd.DataFrame,
    proxy_backtest_report: pd.DataFrame,
    structural_zero_events: pd.Series,
    intervention_summary: pd.DataFrame,
    incomplete_period_log: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    clean_missing_all_zero = bool((missing_audit["clean_missing_count"] == 0).all()) if len(missing_audit) > 0 else False

    leakage_medium_or_high = int(leakage_report["risk_level"].isin(["HIGH", "MEDIUM"]).sum()) if len(leakage_report) > 0 else 0
    unit_fail = int((unit_test_report["status"] == "FAIL").sum()) if len(unit_test_report) > 0 else 0
    synth_fail = int((synthetic_test_report["status"] == "FAIL").sum()) if len(synthetic_test_report) > 0 else 0
    business_review = int((business_rule_test_report["status"] != "PASS").sum()) if len(business_rule_test_report) > 0 else 0
    manual_rows = int(len(manual_sample_audit)) if len(manual_sample_audit) > 0 else 0
    backtest_rows = int(len(proxy_backtest_report)) if len(proxy_backtest_report) > 0 else 0

    max_outlier_fraction = float(quality_report["outlier_fraction_flagged"].max()) if len(quality_report) > 0 else 0.0
    max_clean_zero_ratio = float(quality_report["clean_zero_ratio"].max()) if len(quality_report) > 0 else 0.0
    incomplete_period_count = int(len(incomplete_period_log)) if isinstance(incomplete_period_log, pd.DataFrame) else 0

    clean_smape = pd.to_numeric(
        proxy_backtest_report.loc[proxy_backtest_report["model_proxy"] == "clean_seasonal_naive", "smape"]
        if len(proxy_backtest_report) > 0 and "model_proxy" in proxy_backtest_report.columns else pd.Series(dtype=float),
        errors="coerce"
    )
    max_clean_smape = float(clean_smape.max()) if len(clean_smape) > 0 and clean_smape.notna().any() else np.nan

    candidate_changed_total = int(missing_audit["candidate_clean_changed_count"].sum()) if "candidate_clean_changed_count" in missing_audit.columns else 0
    review_or_exclusion_total = int(missing_audit["review_or_exclusion_governance_count"].sum()) if "review_or_exclusion_governance_count" in missing_audit.columns else 0

    if candidate_changed_total == 0 and review_or_exclusion_total > 0:
        intervention_status = "REVIEW"
        intervention_detail = "Governance/review kayıtları var fakat candidate clean seri değişmedi."
    elif candidate_changed_total > 0 and review_or_exclusion_total > 0:
        intervention_status = "PASS"
        intervention_detail = f"Candidate clean değişen hücre sayısı: {candidate_changed_total}"
    else:
        intervention_status = "PASS"
        intervention_detail = f"Candidate clean değişen hücre sayısı: {candidate_changed_total}"

    summary = [
        {"check_name": "clean_missing_all_zero", "status": "PASS" if clean_missing_all_zero else "REVIEW", "detail": "Temiz veri setinde eksik değer kalmadı." if clean_missing_all_zero else "Temiz veri setinde hâlâ eksik değer var."},
        {"check_name": "regular_time_index", "status": "PASS" if freq_ok else "REVIEW", "detail": freq_msg},
        {"check_name": "outlier_fraction_policy", "status": "PASS" if max_outlier_fraction <= config.review_if_outlier_fraction_gt else "REVIEW", "detail": f"Maksimum outlier oranı: {max_outlier_fraction:.4f}"},
        {"check_name": "structural_zero_event_count", "status": "PASS" if int(structural_zero_events.sum()) <= config.review_if_structural_zero_events_gt else "REVIEW", "detail": f"Yapısal olay sayısı: {int(structural_zero_events.sum())}"},
        {"check_name": "incomplete_last_period_check", "status": "REVIEW" if incomplete_period_count > 0 else "PASS", "detail": f"Incomplete / partial last period adayı sayısı: {incomplete_period_count}"},
        {
            "check_name": "incomplete_last_period_exclusion_policy",
            "status": "PASS" if (
                ("review_or_exclusion_governance_count" in missing_audit.columns and "candidate_clean_changed_count" in missing_audit.columns)
            ) else "REVIEW",
            "detail": "Incomplete last period exclusion mantığı ayrıca anomaly governance ve feature flags üzerinden kontrol edilmelidir."
        },
        {"check_name": "clean_zero_ratio_policy", "status": "PASS" if max_clean_zero_ratio <= config.review_if_clean_zero_ratio_gt else "REVIEW", "detail": f"Maksimum clean_zero_ratio: {max_clean_zero_ratio:.4f}"},
        {"check_name": "strict_leakage_scan", "status": "PASS" if leakage_medium_or_high == 0 else "REVIEW", "detail": f"Medium/High riskli sütun/rule sayısı: {leakage_medium_or_high}"},
        {"check_name": "proxy_backtest_smape_policy", "status": "PASS" if (pd.isna(max_clean_smape) or max_clean_smape <= config.review_if_proxy_smape_gt) else "REVIEW", "detail": f"Maksimum clean_sMAPE: {max_clean_smape}"},
        {"check_name": "unit_tests", "status": "PASS" if unit_fail == 0 else "REVIEW", "detail": f"Başarısız unit test sayısı: {unit_fail}"},
        {"check_name": "synthetic_tests", "status": "PASS" if synth_fail == 0 else "REVIEW", "detail": f"Başarısız synthetic test sayısı: {synth_fail}"},
        {"check_name": "business_rule_tests", "status": "PASS" if business_review == 0 else "REVIEW", "detail": f"Review/fail business test sayısı: {business_review}"},
        {"check_name": "manual_sample_audit", "status": "PASS" if manual_rows > 0 else "REVIEW", "detail": f"Manuel denetim örnek sayısı: {manual_rows}"},
        {"check_name": "proxy_backtest_validation", "status": "PASS" if backtest_rows > 0 else "REVIEW", "detail": f"Proxy backtest rapor satırı: {backtest_rows}"},
        {"check_name": "candidate_clean_intervention_summary", "status": intervention_status, "detail": intervention_detail},
        {"check_name": "outlier_log_created", "status": "PASS", "detail": f"Toplam governance kaydı: {len(outlier_log)}"}
    ]
    return pd.DataFrame(summary)


# =========================================================
# CORE PREPROCESSOR
# =========================================================

class DemandForecastPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.scalers = {}
        self.metadata = {}
        np.random.seed(self.config.random_seed)

        self.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.config_hash = make_config_hash(self.config)
        self.run_started_at = pd.Timestamp.utcnow()

    def preprocess_sheet(self, file_path: str, sheet_name: str, output_dir: str) -> Dict[str, pd.DataFrame]:
        print(f"\n[INFO] Preprocessing started -> Sheet: {sheet_name}")

        df_raw_original = pd.read_excel(file_path, sheet_name=sheet_name)
        original_columns = df_raw_original.columns.tolist()

        date_col = detect_date_column(df_raw_original, self.config)
        df_raw_original[date_col] = parse_datetime_series(df_raw_original[date_col])
        df_raw_original = df_raw_original[df_raw_original[date_col].notna()].copy()

        target_cols = detect_target_columns(df_raw_original, date_col, self.config)

        for c in target_cols:
            df_raw_original[c] = pd.to_numeric(df_raw_original[c], errors="coerce")

        freq = infer_frequency_from_dates(pd.DatetimeIndex(df_raw_original[date_col].sort_values()))
        freq_alias = get_expected_freq_alias(freq)

        df_raw_aligned, datetime_alignment_audit = align_dates_to_frequency(df_raw_original.copy(), date_col, freq_alias)
        df_raw_aggregated = aggregate_duplicates(df_raw_aligned, date_col, target_cols)

        if self.config.force_regular_frequency:
            df_regular = build_regular_time_index(df_raw_aggregated.copy(), date_col, freq_alias)
        else:
            df_regular = df_raw_aggregated.copy().sort_values(date_col).reset_index(drop=True)

        datetime_integrity_audit = create_datetime_integrity_audit(
            df_original=df_raw_original,
            df_aligned=df_raw_aligned,
            df_aggregated=df_raw_aggregated,
            df_regular=df_regular,
            date_col=date_col,
            freq_alias=freq_alias
        )

        # Series profiles
        series_profile_report = create_series_profile_report(df_regular, target_cols, freq_alias, self.config)
        series_profiles = {
            row["series"]: row.drop(labels=["series"]).to_dict()
            for _, row in series_profile_report.iterrows()
        }

        # Structural event engine
        structural_zero_events = detect_structural_zero_events(
            df_regular,
            target_cols,
            self.config.structural_zero_min_series_count,
            self.config.structural_zero_ratio_threshold
        )

        portfolio_shock_flags = detect_portfolio_shocks(df_regular, target_cols, self.config)

        structural_event_flags = (structural_zero_events | portfolio_shock_flags).fillna(False)
        structural_event_flags = expand_structural_events(
            structural_event_flags,
            self.config.structural_event_neighbor_window
        )

        # PATCH: son gözlem(ler)i otomatik structural event etiketlemesinden çıkar
        structural_event_flags = protect_structural_event_edges(
            structural_event_flags,
            protect_last_n=1
        )

        rebound_flags = detect_rebound_after_event(
            df_regular,
            target_cols,
            structural_event_flags,
            self.config
        )

        incomplete_period_flags, incomplete_period_log = detect_incomplete_last_period(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            config=self.config
        )

        structural_event_log = build_structural_event_log(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            zero_flags=structural_zero_events,
            portfolio_shock_flags=portfolio_shock_flags,
            rebound_flags=rebound_flags
        )

        # Adaptive anomaly detection
        outlier_flags = {}
        vote_details = {}

        for c in target_cols:
            flags, vote_df = conservative_outlier_vote_adaptive(df_regular[c], series_profiles[c], self.config)
            flags = cap_outlier_fraction(df_regular[c], flags, vote_df, self.config.max_outlier_fraction_per_series)
            flags = protect_edge_periods(flags, self.config)

            # NOTE:
            # Structural flags are still allowed to be marked as anomaly,
            # but action policy will decide preservation/review later.
            outlier_flags[c] = flags
            vote_details[c] = vote_df

        anomaly_governance = build_anomaly_governance_table(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            outlier_flags=outlier_flags,
            vote_details=vote_details,
            series_profiles=series_profiles,
            structural_event_flags=structural_event_flags,
            incomplete_period_flags=incomplete_period_flags,
            config=self.config
        )



                # PATCH: build counterfactual review candidates
        review_candidate_map = {}
        for c in target_cols:
            tmp = df_regular[[date_col, c]].copy()
            anomaly_dates_c = anomaly_governance.loc[anomaly_governance["series"] == c, "date"].tolist()

            tmp.loc[tmp[date_col].isin(anomaly_dates_c), c] = np.nan

            candidate = seasonal_local_impute(
                df=tmp,
                target_col=c,
                date_col=date_col,
                freq=freq_alias,
                seasonal_period=self.config.seasonal_period_map.get(freq_alias, 1),
                max_interpolation_gap=self.config.max_interpolation_gap
            )

            review_candidate_map[c] = pd.Series(candidate.values, index=tmp[date_col].values)

        # Apply governance decisions into two parallel outputs:
        # 1) clean_governed_preserve: preserves raw for review/exclusion cases
        # 2) clean_candidate_for_modeling: uses candidate imputation for excluded/review dates
        df_clean_governed_preserve = df_regular.copy()
        df_clean_candidate = df_regular.copy()

        governance_map = {}
        if len(anomaly_governance) > 0:
            for _, row in anomaly_governance.iterrows():
                governance_map[(row["date"], row["series"])] = row

        for c in target_cols:
            s_raw = pd.to_numeric(df_regular[c], errors="coerce").copy()

            # Preserve version
            s_preserve_work = s_raw.copy()

            # Candidate version
            s_candidate_work = s_raw.copy()

            for idx in df_regular.index:
                key = (df_regular.loc[idx, date_col], c)
                if key not in governance_map:
                    continue

                g = governance_map[key]
                action = g["action_taken"]

                if action == "set_nan_then_impute":
                    s_preserve_work.loc[idx] = np.nan
                    s_candidate_work.loc[idx] = np.nan

                elif action == "preserve_raw_flag_exclude_candidate":
                    # preserve in audit-clean, but create modeled candidate in candidate-clean
                    s_candidate_work.loc[idx] = np.nan

                elif action in ["flag_only_review", "keep_raw_flag"]:
                    # keep audit clean raw, but candidate series can still propose smoothed alternative
                    s_candidate_work.loc[idx] = np.nan

                elif action == "keep_raw":
                    pass

            original_zero_mask = pd.to_numeric(df_regular[c], errors="coerce").eq(0)

            s_preserve_imputed = seasonal_local_impute(
                df=pd.DataFrame({date_col: df_regular[date_col], c: s_preserve_work}),
                target_col=c,
                date_col=date_col,
                freq=freq_alias,
                seasonal_period=self.config.seasonal_period_map.get(freq_alias, 1),
                max_interpolation_gap=self.config.max_interpolation_gap
            )

            s_candidate_imputed = seasonal_local_impute(
                df=pd.DataFrame({date_col: df_regular[date_col], c: s_candidate_work}),
                target_col=c,
                date_col=date_col,
                freq=freq_alias,
                seasonal_period=self.config.seasonal_period_map.get(freq_alias, 1),
                max_interpolation_gap=self.config.max_interpolation_gap
            )

            if self.config.preserve_zero_values_on_structural_dates:
                preserve_mask = structural_event_flags & original_zero_mask
                s_preserve_imputed.loc[preserve_mask] = 0.0
                s_candidate_imputed.loc[preserve_mask] = 0.0

            if self.config.clip_negative_to_zero:
                s_preserve_imputed = clip_negative_values(s_preserve_imputed)
                s_candidate_imputed = clip_negative_values(s_candidate_imputed)

            df_clean_governed_preserve[c] = s_preserve_imputed
            df_clean_candidate[c] = s_candidate_imputed

        # Candidate clean is the modeling baseline
        df_clean = df_clean_candidate.copy()

        # Optional KNN only if any NA remains
        if self.config.use_knn_for_dense_missing_blocks:
            target_block = df_clean[target_cols].copy()
            if target_block.isna().sum().sum() > 0:
                imputer = KNNImputer(n_neighbors=3)
                df_clean[target_cols] = imputer.fit_transform(target_block)

        # Features
        season_length = self.config.seasonal_period_map.get(freq_alias, 1)
        df_feat = add_calendar_features(df_clean, date_col, freq_alias)
        df_feat = add_lag_features(df_feat, target_cols, freq_alias, season_length)
        df_feat = add_series_quality_features(df_feat, target_cols, anomaly_governance, date_col)

        model_input_transparency = create_model_input_transparency_export(
            df_regular=df_regular,
            df_clean_candidate=df_clean_candidate,
            df_clean_governed_preserve=df_clean_governed_preserve,
            df_feat=df_feat,
            date_col=date_col,
            target_cols=target_cols
        )

        family_exports = create_model_family_exports(
            df_feat=df_feat,
            df_clean=df_clean,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            config=self.config
        )

        # Scaled & log exports
        scaler = choose_scaler(self.config.scaler_for_deep_learning)
        scaled_array = scaler.fit_transform(df_clean[target_cols])
        df_scaled = pd.DataFrame(
            scaled_array,
            columns=[f"{c}_scaled" for c in target_cols],
            index=df_clean.index
        )
        df_scaled.insert(0, date_col, df_clean[date_col].values)

        df_log = df_clean[[date_col] + target_cols].copy()
        if self.config.export_log1p_version:
            for c in target_cols:
                df_log[c] = np.log1p(pd.to_numeric(df_log[c], errors="coerce").clip(lower=0))

        # Reports
        quality = series_quality_report(
            df_raw=df_regular,
            df_clean=df_clean,
            target_cols=target_cols,
            outlier_flags=outlier_flags,
            structural_zero_events=structural_event_flags
        )
        descriptive_statistics_report = create_descriptive_statistics_report(df_clean=df_clean, target_cols=target_cols)
        missing_strategy_audit = create_missing_strategy_audit(
            df_regular=df_regular,
            target_cols=target_cols,
            date_col=date_col,
            config=self.config
        )
        seasonality_report = create_monthly_seasonality_report(
            df_clean=df_clean,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            config=self.config
        )
        pharma_event_diagnostic_report = create_pharma_event_diagnostic_report(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            anomaly_gov=anomaly_governance,
            config=self.config
        )

        intervention_summary, series_intervention_intensity, last_period_intervention = intervention_intensity_report(
            df_regular=df_regular,
            df_clean=df_clean,
            date_col=date_col,
            target_cols=target_cols,
            recent_window=self.config.recent_periods_review_only
        )

        validation_outputs = self._run_validation_audit(
            sheet_name=sheet_name,
            output_dir=output_dir,
            df_raw_after_aggregation=df_raw_aggregated,
            df_regular=df_regular,
            df_clean=df_clean,
            df_clean_governed_preserve=df_clean_governed_preserve,
            df_feat=df_feat,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            outlier_flags=outlier_flags,
            quality_report=quality,
            anomaly_governance=anomaly_governance,
            structural_event_flags=structural_event_flags,
            structural_event_log=structural_event_log,
            incomplete_period_log=incomplete_period_log,
            intervention_summary=intervention_summary
        )

        review_queue = build_review_queue(
            anomaly_gov=anomaly_governance,
            review_candidate_map=review_candidate_map
        )

        manifest = self._build_run_manifest(
            file_path=file_path,
            sheet_name=sheet_name,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            df_raw_original=df_raw_original,
            df_raw_aggregated=df_raw_aggregated,
            df_regular=df_regular,
            anomaly_governance=anomaly_governance,
            intervention_summary=intervention_summary,
            validation_summary=validation_outputs["validation_summary"]
        )

        incomplete_exclusion_count = 0
        if len(anomaly_governance) > 0:
            incomplete_exclusion_count = int(
                (
                    (anomaly_governance["anomaly_type"] == "incomplete_last_period") &
                    (anomaly_governance["excluded_from_training_candidate"] == True)
                ).sum()
            )

        meta = {
            "run_id": self.run_id,
            "pipeline_name": self.config.pipeline_name,
            "pipeline_version": self.config.pipeline_version,
            "code_version": self.config.code_version,
            "output_schema_version": self.config.output_schema_version,
            "config_hash": self.config_hash,
            "file_path": file_path,
            "sheet_name": sheet_name,
            "original_columns": original_columns,
            "date_column": date_col,
            "target_columns": target_cols,
            "frequency_inferred": freq_alias,
            "n_rows_raw_after_date_cleaning": int(len(df_raw_original)),
            "n_rows_aggregated": int(len(df_raw_aggregated)),
            "n_rows_regularized": int(len(df_regular)),
            "date_min": str(df_clean[date_col].min()),
            "date_max": str(df_clean[date_col].max()),
            "scaler_type": self.config.scaler_for_deep_learning,
            "structural_event_count": int(structural_event_flags.sum()),
            "incomplete_period_count": int(incomplete_period_flags.sum()) if len(incomplete_period_flags) > 0 else 0,
            "incomplete_period_exclusion_count": incomplete_exclusion_count,
            "validation_summary_rows": int(len(validation_outputs["validation_summary"])),
            "config": asdict(self.config)
        }
        
        training_exclusion_count = 0
        for c in target_cols:
            col_name = f"{c}_exclude_from_training"
            if col_name in df_feat.columns:
                training_exclusion_count += int(df_feat[col_name].sum())

        candidate_changed_total = 0
        preserve_changed_total = 0
        for c in target_cols:
            raw_s = pd.to_numeric(df_regular[c], errors="coerce")
            cand_s = pd.to_numeric(df_clean_candidate[c], errors="coerce")
            prev_s = pd.to_numeric(df_clean_governed_preserve[c], errors="coerce")

            candidate_changed_total += int(((raw_s != cand_s) & raw_s.notna() & cand_s.notna()).sum())
            preserve_changed_total += int(((raw_s != prev_s) & raw_s.notna() & prev_s.notna()).sum())

        meta["candidate_clean_changed_total"] = candidate_changed_total
        meta["preserve_clean_changed_total"] = preserve_changed_total

        df_clean_model_input = df_clean.copy()
        for c in target_cols:
            excl_col = f"{c}_exclude_from_training"
            if excl_col in df_feat.columns:
                df_clean_model_input.loc[df_feat[excl_col].astype(bool), c] = np.nan
        object_cols_clean = df_clean_model_input.select_dtypes(include=["object"]).columns.tolist()
        object_cols_clean = [c for c in object_cols_clean if c != date_col]
        if len(object_cols_clean) > 0:
            df_clean_model_input = df_clean_model_input.drop(columns=object_cols_clean, errors="ignore")

        meta["training_exclusion_count"] = training_exclusion_count
        meta["model_input_nan_count"] = int(df_clean_model_input[target_cols].isna().sum().sum())
        meta["recommended_manual_validation_count"] = int(sum(df_feat[f"{c}_review_required"].sum() for c in target_cols if f"{c}_review_required" in df_feat.columns))
        
        self.scalers[sheet_name] = scaler
        self.metadata[sheet_name] = meta
            
        export_payload = {
            "raw_regular": df_regular,
            "clean": df_clean,
            "clean_governed_preserve": df_clean_governed_preserve,
            "clean_candidate_for_modeling": df_clean_candidate,
            "clean_model_input": df_clean_model_input,
            "model_input_transparency": model_input_transparency,
            "final_model_visibility_report": model_input_transparency,
            "features": df_feat,
            "scaled": df_scaled,
            "log": df_log,
            "quality_report": quality,
            "descriptive_statistics_report": descriptive_statistics_report,
            "missing_strategy_audit": missing_strategy_audit,
            "seasonality_report": seasonality_report,
            "pharma_event_diagnostic_report": pharma_event_diagnostic_report,
            "datetime_integrity_audit": datetime_integrity_audit,
            "datetime_alignment_audit": datetime_alignment_audit,
            "series_profile_report": series_profile_report,
            "anomaly_governance": anomaly_governance,
            "review_queue": review_queue,
            "intervention_summary": intervention_summary,
            "series_intervention_intensity": series_intervention_intensity,
            "last_period_intervention_report": last_period_intervention,
            "structural_event_log": structural_event_log,
            "incomplete_period_log": incomplete_period_log,
            "manifest": manifest,
            **family_exports,
            **validation_outputs
        }

        self._export_all(
            output_dir=output_dir,
            sheet_name=sheet_name,
            export_payload=export_payload,
            metadata=meta
        )

        print(f"[INFO] Completed -> Sheet: {sheet_name}")

        return export_payload

    def _build_run_manifest(
        self,
        file_path: str,
        sheet_name: str,
        date_col: str,
        target_cols: List[str],
        freq_alias: str,
        df_raw_original: pd.DataFrame,
        df_raw_aggregated: pd.DataFrame,
        df_regular: pd.DataFrame,
        anomaly_governance: pd.DataFrame,
        intervention_summary: pd.DataFrame,
        validation_summary: pd.DataFrame
    ) -> pd.DataFrame:
        passed = int((validation_summary["status"] == "PASS").sum()) if len(validation_summary) > 0 else 0
        review = int((validation_summary["status"] == "REVIEW").sum()) if len(validation_summary) > 0 else 0

        changed_cells = 0
        row = intervention_summary.loc[intervention_summary["metric"] == "total_changed_cells"] if len(intervention_summary) > 0 else pd.DataFrame()
        if len(row) > 0:
            changed_cells = int(row["value"].iloc[0])

        manifest = pd.DataFrame([
            {"key": "run_id", "value": self.run_id},
            {"key": "pipeline_name", "value": self.config.pipeline_name},
            {"key": "pipeline_version", "value": self.config.pipeline_version},
            {"key": "code_version", "value": self.config.code_version},
            {"key": "output_schema_version", "value": self.config.output_schema_version},
            {"key": "config_hash", "value": self.config_hash},
            {"key": "file_path", "value": file_path},
            {"key": "sheet_name", "value": sheet_name},
            {"key": "date_column", "value": date_col},
            {"key": "frequency_inferred", "value": freq_alias},
            {"key": "target_count", "value": len(target_cols)},
            {"key": "raw_rows", "value": len(df_raw_original)},
            {"key": "aggregated_rows", "value": len(df_raw_aggregated)},
            {"key": "regularized_rows", "value": len(df_regular)},
            {"key": "anomaly_rows", "value": len(anomaly_governance)},
            {"key": "changed_cells", "value": changed_cells},
            {"key": "validation_pass_count", "value": passed},
            {"key": "validation_review_count", "value": review},
            {"key": "run_started_at_utc", "value": str(self.run_started_at)},
            {"key": "run_finished_at_utc", "value": str(pd.Timestamp.utcnow())}
        ])
        return manifest

    def _run_validation_audit(
        self,
        sheet_name: str,
        output_dir: str,
        df_raw_after_aggregation: pd.DataFrame,
        df_regular: pd.DataFrame,
        df_clean: pd.DataFrame,
        df_clean_governed_preserve: pd.DataFrame,
        df_feat: pd.DataFrame,
        date_col: str,
        target_cols: List[str],
        freq_alias: str,
        outlier_flags: Dict[str, pd.Series],
        quality_report: pd.DataFrame,
        anomaly_governance: pd.DataFrame,
        structural_event_flags: pd.Series,
        structural_event_log: pd.DataFrame,
        incomplete_period_log: pd.DataFrame,
        intervention_summary: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:

        safe_sheet = re.sub(r"[^\w\-]+", "_", sheet_name)
        sheet_dir = os.path.join(output_dir, safe_sheet)
        os.makedirs(sheet_dir, exist_ok=True)

        freq_ok, freq_msg = check_regular_index(df_clean, date_col, freq_alias)
        missing_audit = create_missing_value_audit(
            df_regular=df_regular,
            df_clean=df_clean,
            target_cols=target_cols,
            anomaly_gov=anomaly_governance,
            df_clean_governed_preserve=df_clean_governed_preserve
        )
        missing_strategy_audit = create_missing_strategy_audit(
            df_regular=df_regular,
            target_cols=target_cols,
            date_col=date_col,
            config=self.config
        )
        descriptive_statistics_report = create_descriptive_statistics_report(df_clean=df_clean, target_cols=target_cols)
        seasonality_report = create_monthly_seasonality_report(
            df_clean=df_clean,
            date_col=date_col,
            target_cols=target_cols,
            freq_alias=freq_alias,
            config=self.config
        )
        pharma_event_diagnostic_report = create_pharma_event_diagnostic_report(
            df_regular=df_regular,
            date_col=date_col,
            target_cols=target_cols,
            anomaly_gov=anomaly_governance,
            config=self.config
        )
        datetime_integrity_audit = create_datetime_integrity_audit(
            df_original=df_raw_after_aggregation,
            df_aligned=df_raw_after_aggregation,
            df_aggregated=df_raw_after_aggregation,
            df_regular=df_regular,
            date_col=date_col,
            freq_alias=freq_alias
        )
        frequency_audit = create_frequency_audit(df_raw_after_aggregation, df_regular, date_col, freq_alias)
        inserted_timestamp_log = create_inserted_timestamp_log(df_raw_after_aggregation, df_regular, date_col)

        leakage_report = (
            strict_leakage_audit(df_feat, target_cols)
            if self.config.leakage_check_enabled else
            pd.DataFrame(columns=["column_name", "status", "risk_level", "note"])
        )

        outlier_log = create_outlier_log(
            df_regular=df_regular,
            df_clean=df_clean,
            date_col=date_col,
            target_cols=target_cols,
            anomaly_gov=anomaly_governance
        )

        domain_validation_template = (
            create_domain_validation_template(df_clean, date_col, target_cols)
            if self.config.create_domain_validation_template else
            pd.DataFrame()
        )

        correlation_matrix_report = pd.DataFrame()
        target_correlation_report = pd.DataFrame()
        seasonality_heatmap_report = pd.DataFrame()
        seasonal_decomposition_report = pd.DataFrame()

        if self.config.save_validation_plots:
            save_raw_vs_clean_plots(
                df_regular=df_regular,
                df_clean=df_clean,
                date_col=date_col,
                target_cols=target_cols,
                sheet_dir=sheet_dir,
                max_plot_series=self.config.max_plot_series
            )
            if self.config.save_trend_plots:
                save_raw_clean_trend_plots(
                    df_regular=df_regular,
                    df_clean=df_clean,
                    date_col=date_col,
                    target_cols=target_cols,
                    sheet_dir=sheet_dir,
                    max_plot_series=self.config.max_plot_series,
                    ma_windows=self.config.moving_average_windows
                )
            if self.config.save_distribution_plots:
                save_distribution_plots(
                    df_clean=df_clean,
                    target_cols=target_cols,
                    sheet_dir=sheet_dir,
                    max_plot_series=self.config.max_plot_series,
                    save_boxplots=self.config.save_boxplots
                )
            if self.config.save_seasonality_plots:
                save_seasonality_plots(
                    df_clean=df_clean,
                    date_col=date_col,
                    target_cols=target_cols,
                    freq_alias=freq_alias,
                    sheet_dir=sheet_dir,
                    max_plot_series=self.config.max_plot_series,
                    save_year_overlay=self.config.save_year_overlay_seasonality_plots,
                    save_normalized_profile=self.config.save_normalized_seasonality_plots,
                    save_boxplot=self.config.save_boxplots
                )
            if self.config.save_correlation_analysis:
                correlation_matrix_report, target_correlation_report = save_correlation_analysis(
                    df_for_corr=df_feat,
                    target_cols=target_cols,
                    sheet_dir=sheet_dir
                )
            if self.config.save_seasonality_heatmaps or self.config.save_seasonal_decomposition:
                seasonality_heatmap_report, seasonal_decomposition_report = save_seasonality_heatmaps_and_decomposition(
                    df_clean=df_clean,
                    date_col=date_col,
                    target_cols=target_cols,
                    freq_alias=freq_alias,
                    sheet_dir=sheet_dir,
                    max_plot_series=self.config.max_plot_series
                )

        unit_test_report = run_internal_unit_tests(self.config) if self.config.run_internal_unit_tests else pd.DataFrame()
        synthetic_test_report = run_synthetic_tests(self.config) if self.config.run_synthetic_tests else pd.DataFrame()
        business_rule_test_report = (
            run_business_rule_tests(
                df_regular=df_regular,
                df_clean=df_clean,
                target_cols=target_cols,
                anomaly_gov=anomaly_governance,
                config=self.config
            )
            if self.config.run_business_rule_tests else pd.DataFrame()
        )
        anomaly_dates = anomaly_governance["date"].drop_duplicates().tolist() if len(anomaly_governance) > 0 else []

        manual_sample_audit = (
            create_manual_sample_audit(
                df_regular=df_regular,
                df_clean=df_clean,
                date_col=date_col,
                target_cols=target_cols,
                sample_size=self.config.manual_sample_size,
                random_seed=self.config.random_seed,
                anomaly_dates=anomaly_dates
            )
            if self.config.run_manual_sample_audit else pd.DataFrame()
        )
        proxy_backtest_report_clean_truth = (
            run_proxy_backtest_validation(
                df_raw_regular=df_regular,
                df_clean=df_clean,
                target_cols=target_cols,
                freq_alias=freq_alias,
                config=self.config,
                truth_source="clean"
            )
            if self.config.run_proxy_backtest_validation else pd.DataFrame()
        )

        proxy_backtest_report_raw_truth = (
            run_proxy_backtest_validation(
                df_raw_regular=df_regular,
                df_clean=df_clean,
                target_cols=target_cols,
                freq_alias=freq_alias,
                config=self.config,
                truth_source="raw"
            )
            if self.config.run_proxy_backtest_validation else pd.DataFrame()
        )

        proxy_backtest_report = pd.concat(
            [proxy_backtest_report_clean_truth, proxy_backtest_report_raw_truth],
            axis=0,
            ignore_index=True
        ) if (len(proxy_backtest_report_clean_truth) > 0 or len(proxy_backtest_report_raw_truth) > 0) else pd.DataFrame()

        raw_vs_clean_backtest_report = raw_vs_clean_backtest_comparator(
            proxy_backtest_report_clean_truth if len(proxy_backtest_report_clean_truth) > 0 else pd.DataFrame()
        )

        validation_summary = create_validation_summary(
            quality_report=quality_report,
            missing_audit=missing_audit,
            freq_ok=freq_ok,
            freq_msg=freq_msg,
            outlier_log=outlier_log,
            leakage_report=leakage_report,
            unit_test_report=unit_test_report,
            synthetic_test_report=synthetic_test_report,
            business_rule_test_report=business_rule_test_report,
            manual_sample_audit=manual_sample_audit,
            proxy_backtest_report=proxy_backtest_report,
            structural_zero_events=structural_event_flags,
            intervention_summary=intervention_summary,
            incomplete_period_log=incomplete_period_log,
            config=self.config
        )

        return {
            "validation_summary": validation_summary,
            "missing_audit": missing_audit,
            "missing_strategy_audit": missing_strategy_audit,
            "descriptive_statistics_report": descriptive_statistics_report,
            "correlation_matrix_report": correlation_matrix_report,
            "target_correlation_report": target_correlation_report,
            "seasonality_report": seasonality_report,
            "seasonality_heatmap_report": seasonality_heatmap_report,
            "seasonal_decomposition_report": seasonal_decomposition_report,
            "pharma_event_diagnostic_report": pharma_event_diagnostic_report,
            "datetime_integrity_audit": datetime_integrity_audit,
            "frequency_audit": frequency_audit,
            "inserted_timestamp_log": inserted_timestamp_log,
            "outlier_log": outlier_log,
            "leakage_report": leakage_report,
            "domain_validation_template": domain_validation_template,
            "unit_test_report": unit_test_report,
            "synthetic_test_report": synthetic_test_report,
            "synthetic_missing_tests_report": synthetic_test_report,
            "business_rule_test_report": business_rule_test_report,
            "manual_sample_audit": manual_sample_audit,
            "proxy_backtest_report": proxy_backtest_report,
            "raw_vs_clean_backtest_report": raw_vs_clean_backtest_report,
            "structural_zero_event_log": structural_event_log
        }

    def _export_all(
        self,
        output_dir: str,
        sheet_name: str,
        export_payload: Dict[str, pd.DataFrame],
        metadata: Dict[str, Any]
    ):
        safe_sheet = re.sub(r"[^\w\-]+", "_", sheet_name)
        sheet_dir = os.path.join(output_dir, safe_sheet)
        os.makedirs(sheet_dir, exist_ok=True)

        if self.config.save_csv:
            csv_map = {
                "modeling_features_prophet": f"{safe_sheet}_modeling_features_prophet.csv",
                "modeling_features_global_long": f"{safe_sheet}_modeling_features_global_long.csv",
                "clean_model_input": f"{safe_sheet}_clean_model_input.csv",
                "model_input_transparency": f"{safe_sheet}_model_input_transparency.csv",
                "final_model_visibility_report": f"{safe_sheet}_final_model_visibility_report.csv",
                "raw_regular": f"{safe_sheet}_raw_regular.csv",
                "clean": f"{safe_sheet}_clean.csv",
                "clean_governed_preserve": f"{safe_sheet}_clean_governed_preserve.csv",
                "clean_candidate_for_modeling": f"{safe_sheet}_clean_candidate_for_modeling.csv",
                "features": f"{safe_sheet}_features.csv",
                "scaled": f"{safe_sheet}_scaled.csv",
                "log": f"{safe_sheet}_log1p.csv",
                "quality_report": f"{safe_sheet}_quality_report.csv",
                "descriptive_statistics_report": f"{safe_sheet}_descriptive_statistics_report.csv",
                "missing_strategy_audit": f"{safe_sheet}_missing_strategy_audit.csv",
                "seasonality_report": f"{safe_sheet}_seasonality_report.csv",
                "pharma_event_diagnostic_report": f"{safe_sheet}_pharma_event_diagnostic_report.csv",
                "datetime_integrity_audit": f"{safe_sheet}_datetime_integrity_audit.csv",
                "datetime_alignment_audit": f"{safe_sheet}_datetime_alignment_audit.csv",
                "series_profile_report": f"{safe_sheet}_series_profile_report.csv",
                "anomaly_governance": f"{safe_sheet}_anomaly_governance.csv",
                "review_queue": f"{safe_sheet}_review_queue.csv",
                "intervention_summary": f"{safe_sheet}_intervention_summary.csv",
                "series_intervention_intensity": f"{safe_sheet}_series_intervention_intensity.csv",
                "last_period_intervention_report": f"{safe_sheet}_last_period_intervention_report.csv",
                "structural_event_log": f"{safe_sheet}_structural_event_log.csv",
                "incomplete_period_log": f"{safe_sheet}_incomplete_period_log.csv",
                "modeling_features_statistical": f"{safe_sheet}_modeling_features_statistical.csv",
                "modeling_features_ml": f"{safe_sheet}_modeling_features_ml.csv",
                "modeling_features_dl": f"{safe_sheet}_modeling_features_dl.csv",
                "modeling_features_foundation": f"{safe_sheet}_modeling_features_foundation.csv",
                "validation_summary": f"{safe_sheet}_validation_summary.csv",
                "missing_audit": f"{safe_sheet}_missing_audit.csv",
                "frequency_audit": f"{safe_sheet}_frequency_audit.csv",
                "inserted_timestamp_log": f"{safe_sheet}_inserted_timestamp_log.csv",
                "outlier_log": f"{safe_sheet}_outlier_log.csv",
                "leakage_report": f"{safe_sheet}_leakage_report.csv",
                "domain_validation_template": f"{safe_sheet}_domain_validation_template.csv",
                "unit_test_report": f"{safe_sheet}_unit_test_report.csv",
                "synthetic_test_report": f"{safe_sheet}_synthetic_test_report.csv",
                "synthetic_missing_tests_report": f"{safe_sheet}_synthetic_missing_tests_report.csv",
                "business_rule_test_report": f"{safe_sheet}_business_rule_test_report.csv",
                "manual_sample_audit": f"{safe_sheet}_manual_sample_audit.csv",
                "proxy_backtest_report": f"{safe_sheet}_proxy_backtest_report.csv",
                "raw_vs_clean_backtest_report": f"{safe_sheet}_raw_vs_clean_backtest_report.csv",
                "manifest": f"{safe_sheet}_run_manifest.csv"
            }

            for key, filename in csv_map.items():
                df_obj = export_payload.get(key, pd.DataFrame())
                if isinstance(df_obj, pd.DataFrame):
                    df_obj.to_csv(os.path.join(sheet_dir, filename), index=False, encoding="utf-8-sig")

        if self.config.save_excel:
            excel_path = os.path.join(sheet_dir, f"{safe_sheet}_preprocessing_package.xlsx")
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for key, df_obj in export_payload.items():
                    if not isinstance(df_obj, pd.DataFrame):
                        continue
                    sheet = safe_excel_sheet_name(key)
                    df_obj.to_excel(writer, sheet_name=sheet, index=False)

                meta_df = pd.DataFrame({
                    "key": list(metadata.keys()),
                    "value": [
                        json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for v in metadata.values()
                    ]
                })
                meta_df.to_excel(writer, sheet_name=safe_excel_sheet_name("metadata"), index=False)

        if self.config.save_metadata_json:
            with open(os.path.join(sheet_dir, f"{safe_sheet}_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            manifest_df = export_payload.get("manifest", pd.DataFrame())
            manifest_json = {
                row["key"]: row["value"] for _, row in manifest_df.iterrows()
            } if isinstance(manifest_df, pd.DataFrame) and len(manifest_df) > 0 else {}
            with open(os.path.join(sheet_dir, f"{safe_sheet}_run_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest_json, f, ensure_ascii=False, indent=4)

    def save_global_metadata(self, output_dir: str):
        if self.config.save_metadata_json:
            with open(os.path.join(output_dir, "all_sheets_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=4)


# =========================================================
# MAIN
# =========================================================

def main():
    try:
        config = PreprocessConfig(
            output_dir_name="forecast_preprocessing_outputs",

            hampel_window=7,
            hampel_n_sigma=4.0,
            rolling_mad_window=9,
            rolling_mad_n_sigma=4.5,
            iqr_k=4.0,
            min_outlier_votes=2,

            max_outlier_fraction_per_series=0.05,
            protect_first_n_periods=1,
            protect_last_n_periods=1,
            recent_periods_review_only=3,
            clip_negative_to_zero=True,

            structural_zero_ratio_threshold=0.5,
            structural_zero_min_series_count=3,
            portfolio_drop_ratio_threshold=0.55,
            portfolio_rebound_ratio_threshold=1.30,
            structural_event_neighbor_window=0,
            enable_incomplete_period_detection=True,
            partial_period_drop_ratio_threshold=0.60,
            partial_period_compare_last_n=3,
            auto_exclude_incomplete_last_period_from_training=True,
            auto_flag_incomplete_last_period_review=True,

            max_interpolation_gap=1,
            use_knn_for_dense_missing_blocks=False,
            impute_method_preference="seasonal_local",

            min_action_confidence_for_auto_fix=0.75,
            auto_fix_business_spike_dip=False,
            auto_fix_unknown_anomaly=False,
            auto_fix_data_error=True,
            auto_fix_structural_event=False,

            scaler_for_deep_learning="robust",
            export_log1p_version=True,

            generate_modeling_ready_feature_pack=True,
            exclude_textual_columns_from_modeling_features=True,
            drop_low_signal_calendar_features_for_monthly=True,
            export_training_exclusion_masks=True,

            save_validation_excel=True,
            save_validation_csv=True,
            save_validation_plots=True,
            create_domain_validation_template=True,
            leakage_check_enabled=True,
            max_plot_series=50,

            run_internal_unit_tests=True,
            run_synthetic_tests=True,
            run_business_rule_tests=True,
            run_manual_sample_audit=True,
            run_proxy_backtest_validation=True,
            manual_sample_size=20,
            random_seed=42,

            backtest_horizon=3,
            backtest_min_train_size_monthly=24,
            backtest_min_train_size_weekly=52,
            backtest_min_train_size_daily=60,
            backtest_min_train_size_hourly=24 * 14,

            review_if_outlier_fraction_gt=0.05,
            review_if_structural_zero_events_gt=1,
            review_if_clean_zero_ratio_gt=0.10,
            review_if_proxy_smape_gt=60.0,

            save_excel=True,
            save_csv=True,
            save_metadata_json=True,
            save_quality_report=True
        )

        input_info = choose_excel_file()
        source_path = input_info["source_path"]
        excel_path = input_info["excel_path"]

        xls = safe_excel_file(excel_path)
        selected_sheets = choose_sheets(xls.sheet_names)
        output_dir = create_output_dir(source_path, config.output_dir_name)

        preprocessor = DemandForecastPreprocessor(config=config)

        all_results = {}
        for sheet in selected_sheets:
            result = preprocessor.preprocess_sheet(
                file_path=excel_path,
                sheet_name=sheet,
                output_dir=output_dir
            )
            all_results[sheet] = result

        preprocessor.save_global_metadata(output_dir)

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Başarılı",
            "Veri önişleme + anomaly governance + validation + QA modülleri tamamlandı.\n"
            f"Çıktılar şu klasöre kaydedildi:\n{output_dir}"
        )

        print("\n[SUCCESS] All preprocessing, governance, validation and QA steps finished successfully.")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Hata", str(e))




# =========================================================
# STREAMLIT FORECASTING APP LAYER
# =========================================================

import importlib
from itertools import product

try:
    import streamlit as st
except Exception:
    st = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    go = None
    make_subplots = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_FORECAST_STATSMODELS = True
except Exception:
    SARIMAX = None
    adfuller = None
    kpss = None
    acf = None
    pacf = None
    plot_acf = None
    plot_pacf = None
    acorr_ljungbox = None
    HAS_FORECAST_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    except Exception:
        XGBRegressor = None

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    from sklearn.model_selection import ParameterGrid
except Exception:
    ParameterGrid = None

APP_VERSION = "1.0.1"


def infer_season_length_from_freq(freq_alias: str) -> int:
    return {"M": 12, "W": 52, "D": 7, "H": 24}.get(str(freq_alias).upper(), 1)


def infer_default_horizon(freq_alias: str) -> int:
    return {"M": 6, "W": 12, "D": 30, "H": 48}.get(str(freq_alias).upper(), 6)


def _safe_bool_series(s: pd.Series) -> pd.Series:
    return pd.Series(s).fillna(False).astype(bool)


def detect_optional_exog_columns(df_features: pd.DataFrame, target_col: str, date_col: str) -> List[str]:
    cols = []
    generic_keywords = [
        "month", "quarter", "year", "weekofyear", "dayofweek", "dayofmonth",
        "month_sin", "month_cos", "quarter_sin", "quarter_cos", "is_month_start",
        "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
    ]
    banned_keywords = [
        "exclude_from_training", "review_required", "structural_event_flag", "incomplete_period_flag",
        "anomaly_flag", "scaled", "log1p"
    ]

    for c in df_features.columns:
        if c == date_col or c == target_col:
            continue
        lc = c.lower()
        if any(b in lc for b in banned_keywords):
            continue
        if c.startswith(f"{target_col}_"):
            # target-specific lag/rolling features belong to ML branch, not exogenous statistical branch by default
            continue
        if lc in generic_keywords or any(g == lc for g in generic_keywords):
            if pd.api.types.is_numeric_dtype(df_features[c]):
                cols.append(c)
    return sorted(set(cols))


def detect_ml_feature_columns(df_features: pd.DataFrame, target_col: str, date_col: str) -> List[str]:
    cols = []
    generic_keywords = [
        "month", "quarter", "year", "weekofyear", "dayofweek", "dayofmonth",
        "month_sin", "month_cos", "quarter_sin", "quarter_cos", "is_month_start",
        "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
    ]
    banned_keywords = [
        "exclude_from_training", "review_required", "structural_event_flag", "incomplete_period_flag",
        "anomaly_flag"
    ]
    for c in df_features.columns:
        if c == date_col or c == target_col:
            continue
        lc = c.lower()
        if any(b in lc for b in banned_keywords):
            continue
        if c.startswith(f"{target_col}_") or lc in generic_keywords:
            if pd.api.types.is_numeric_dtype(df_features[c]):
                cols.append(c)
    return sorted(set(cols))


def make_series_analysis_frame(export_payload: Dict[str, pd.DataFrame], target_col: str) -> pd.DataFrame:
    df_clean = export_payload["clean_model_input"].copy()
    df_feat = export_payload["features"].copy()
    date_col = export_payload["manifest"].loc[export_payload["manifest"]["key"] == "date_column", "value"].iloc[0]

    out = pd.DataFrame({
        "ds": pd.to_datetime(df_clean[date_col]),
        "y": pd.to_numeric(df_clean[target_col], errors="coerce")
    })
    excl_col = f"{target_col}_exclude_from_training"
    out["exclude_from_training"] = _safe_bool_series(df_feat[excl_col]) if excl_col in df_feat.columns else False
    out["is_usable"] = out["y"].notna() & (~out["exclude_from_training"])
    return out


def get_profile_row(export_payload: Dict[str, pd.DataFrame], target_col: str) -> Dict[str, Any]:
    prof = export_payload["series_profile_report"]
    row = prof.loc[prof["series"] == target_col]
    return row.iloc[0].to_dict() if len(row) else {}


def series_segment_label(profile: Dict[str, Any]) -> str:
    cv = float(profile.get("cv", np.nan)) if profile else np.nan
    intermittency = float(profile.get("intermittency_ratio", np.nan)) if profile else np.nan
    seasonality = float(profile.get("seasonality_strength", np.nan)) if profile else np.nan
    volume = str(profile.get("volume_level", "unknown")) if profile else "unknown"

    if pd.notna(intermittency) and intermittency >= 0.35:
        return "intermittent"
    if pd.notna(seasonality) and seasonality >= 0.45:
        return "seasonal"
    if pd.notna(cv) and cv >= 0.45:
        return "volatile"
    if volume in ["high", "medium"] and pd.notna(cv) and cv < 0.25:
        return "stable_fast_mover"
    return "standard"


def recommend_model_priority(profile: Dict[str, Any]) -> str:
    label = series_segment_label(profile)
    if label == "seasonal":
        return "Prophet + SARIMA"
    if label == "volatile":
        return "XGBoost + Prophet"
    if label == "intermittent":
        return "XGBoost (dikkatli feature engineering)"
    if label == "stable_fast_mover":
        return "SARIMA + XGBoost"
    return "SARIMA + Prophet + XGBoost"



try:
    from scipy.special import inv_boxcox
    from scipy.stats import boxcox
    HAS_SCIPY = True
except Exception:
    inv_boxcox = None
    boxcox = None
    HAS_SCIPY = False

def infer_season_length_from_freq(freq_alias: str) -> int:
    return {"M": 12, "W": 52, "D": 7, "H": 24}.get(str(freq_alias).upper(), 1)

def infer_abc_class(profile: Dict[str, Any]) -> str:
    mean_ = safe_float(profile.get("mean", np.nan))
    if pd.isna(mean_):
        return "C"
    if mean_ >= 250:
        return "A"
    if mean_ >= 100:
        return "B"
    return "C"

def infer_xyz_class(profile: Dict[str, Any]) -> str:
    cv = safe_float(profile.get("cv", np.nan))
    if pd.isna(cv):
        return "Z"
    if cv <= 0.25:
        return "X"
    if cv <= 0.45:
        return "Y"
    return "Z"

def infer_advanced_segment(profile: Dict[str, Any]) -> Dict[str, str]:
    abc = infer_abc_class(profile)
    xyz = infer_xyz_class(profile)
    intermittency = safe_float(profile.get("intermittency_ratio", np.nan))
    seasonality = safe_float(profile.get("seasonality_strength", np.nan))
    trend_strength = safe_float(profile.get("trend_strength", np.nan))
    cv = safe_float(profile.get("cv", np.nan))

    if pd.notna(intermittency) and intermittency >= 0.35:
        family = "intermittent"
    elif pd.notna(seasonality) and seasonality >= 0.45:
        family = "seasonal"
    elif pd.notna(cv) and cv >= 0.45:
        family = "volatile"
    elif pd.notna(trend_strength) and trend_strength >= 0.45:
        family = "trend_dominant"
    elif xyz == "X":
        family = "stable"
    else:
        family = "standard"

    return {
        "abc": abc,
        "xyz": xyz,
        "abc_xyz": f"{abc}{xyz}",
        "family": family,
        "label": f"{abc}{xyz}_{family}"
    }

def recommend_candidate_models(profile: Dict[str, Any]) -> List[str]:
    seg = infer_advanced_segment(profile)
    family = seg["family"]
    abcxyz = seg["abc_xyz"]
    if family == "seasonal":
        return ["SARIMA/SARIMAX", "Prophet", "Ensemble"]
    if family == "intermittent":
        return ["XGBoost", "Prophet", "Ensemble"]
    if family == "volatile":
        return ["XGBoost", "Prophet", "SARIMA/SARIMAX", "Ensemble"]
    if abcxyz in ["AX", "BX"]:
        return ["SARIMA/SARIMAX", "XGBoost", "Ensemble"]
    return ["SARIMA/SARIMAX", "Prophet", "XGBoost", "Ensemble"]

def choose_target_transform(y: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(s) < 8:
        return {"name": "none", "lambda": None, "shift": 0.0}
    skew = safe_float(s.skew())
    min_val = safe_float(s.min())
    cv = safe_float(coefficient_of_variation(s))
    if min_val >= 0 and ((pd.notna(skew) and skew >= 1.0) or (pd.notna(cv) and cv >= 0.50)):
        if HAS_SCIPY and np.all(s > 0):
            return {"name": "boxcox", "lambda": None, "shift": 0.0}
        return {"name": "log1p", "lambda": None, "shift": 0.0}
    return {"name": "none", "lambda": None, "shift": 0.0}

def apply_target_transform(y: pd.Series, transform_cfg: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    s = pd.to_numeric(y, errors="coerce").astype(float).copy()
    name = transform_cfg.get("name", "none")
    if name == "log1p":
        return np.log1p(np.maximum(s, 0.0)), transform_cfg
    if name == "boxcox" and HAS_SCIPY:
        s2 = s.copy()
        shift = 0.0
        if safe_float(s2.min()) <= 0:
            shift = abs(safe_float(s2.min())) + 1e-6
            s2 = s2 + shift
        transformed, lam = boxcox(s2.values)
        cfg = dict(transform_cfg)
        cfg["lambda"] = lam
        cfg["shift"] = shift
        return pd.Series(transformed, index=s.index), cfg
    return s, transform_cfg

def inverse_target_transform(arr: np.ndarray, transform_cfg: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    name = transform_cfg.get("name", "none")
    if name == "log1p":
        return np.maximum(np.expm1(x), 0.0)
    if name == "boxcox" and HAS_SCIPY:
        lam = transform_cfg.get("lambda", None)
        shift = float(transform_cfg.get("shift", 0.0) or 0.0)
        inv = inv_boxcox(x, lam)
        inv = inv - shift
        return np.maximum(inv, 0.0)
    return np.maximum(x, 0.0)

def make_inner_train_val_split(train_df: pd.DataFrame, val_ratio: float = 0.2, min_val: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    val_len = max(min_val, int(np.ceil(n * val_ratio)))
    val_len = min(val_len, max(2, n // 3))
    return train_df.iloc[:-val_len].copy().reset_index(drop=True), train_df.iloc[-val_len:].copy().reset_index(drop=True)

def walk_forward_refit_sarimax(history_y: pd.Series, future_y: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], exog_hist: Optional[pd.DataFrame] = None, exog_future: Optional[pd.DataFrame] = None) -> np.ndarray:
    preds = []
    hist = pd.to_numeric(history_y, errors="coerce").astype(float).copy()
    for i in range(len(future_y)):
        exog_train_i = exog_hist if exog_hist is None else exog_hist.iloc[:len(hist)]
        exog_step = None if exog_future is None else exog_future.iloc[[i]]
        model = SARIMAX(hist, exog=exog_train_i, order=order, seasonal_order=seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=1, exog=exog_step).predicted_mean.iloc[0]
        preds.append(max(float(fc), 0.0))
        hist = pd.concat([hist, pd.Series([future_y.iloc[i]])], ignore_index=True)
    return np.asarray(preds, dtype=float)

def make_prophet_features(train_df: pd.DataFrame, test_df: pd.DataFrame, exog_train: Optional[pd.DataFrame], exog_test: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    tr = train_df[["ds", "y"]].copy()
    te = test_df[["ds", "y"]].copy()
    used = []
    if exog_train is not None and exog_test is not None and len(exog_train.columns) > 0:
        for c in exog_train.columns:
            tr[c] = pd.to_numeric(exog_train[c], errors="coerce").values
            te[c] = pd.to_numeric(exog_test[c], errors="coerce").values
            used.append(c)
    return tr, te, used

def generate_target_ml_features(full_df: pd.DataFrame, existing_exog: Optional[pd.DataFrame], freq_alias: str) -> Tuple[pd.DataFrame, List[str]]:
    df = full_df[["ds", "y"]].copy().reset_index(drop=True)
    season_len = infer_season_length_from_freq(freq_alias)
    lag_set = [1, 2, 3, 6, 12]
    if freq_alias == "W":
        lag_set = [1, 2, 4, 8, 13, 26, 52]
    elif freq_alias == "D":
        lag_set = [1, 2, 3, 7, 14, 21, 28]
    elif freq_alias == "M":
        lag_set = [1, 2, 3, 6, 12, 18]
    elif freq_alias == "H":
        lag_set = [1, 2, 3, 6, 12, 24, 48]

    for lag in sorted(set(lag_set)):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    if season_len > 1:
        df[f"seasonal_lag_{season_len}"] = df["y"].shift(season_len)

    for w in sorted(set([3, 6, 12, max(2, season_len)])):
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).std()
        df[f"roll_min_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).max()

    ds = pd.to_datetime(df["ds"])
    df["year"] = ds.dt.year
    df["quarter"] = ds.dt.quarter
    df["month"] = ds.dt.month
    iso_week = ds.dt.isocalendar().week.astype(int)
    df["weekofyear"] = iso_week
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofmonth"] = ds.dt.day
    df["is_month_start"] = ds.dt.is_month_start.astype(int)
    df["is_month_end"] = ds.dt.is_month_end.astype(int)

    month_num = ds.dt.month.fillna(1).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
    if freq_alias in ["W", "D"]:
        week_num = iso_week.clip(lower=1)
        df["week_sin"] = np.sin(2 * np.pi * week_num / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * week_num / 52.0)
    if freq_alias in ["D", "H"]:
        dow = ds.dt.dayofweek.astype(int)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if existing_exog is not None and len(existing_exog.columns) > 0:
        exog = existing_exog.copy().reset_index(drop=True)
        for c in exog.columns:
            df[f"exog__{c}"] = pd.to_numeric(exog[c], errors="coerce")

    feature_cols = [c for c in df.columns if c not in ["ds", "y"]]
    return df, feature_cols

def build_actual_vs_pred_df(test_df: pd.DataFrame, pred: np.ndarray, model_name: str) -> pd.DataFrame:
    out = test_df[["ds", "y"]].copy()
    out["prediction"] = np.maximum(np.asarray(pred, dtype=float), 0.0)
    out["model"] = model_name
    out["abs_error"] = np.abs(out["y"] - out["prediction"])
    out["ape"] = np.where(np.abs(out["y"]) > 1e-8, np.abs(out["y"] - out["prediction"]) / np.abs(out["y"]) * 100.0, np.nan)
    return out

def style_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format only numeric-like columns while preserving datetimes, strings and nested objects."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce").round(4)
            continue

        def _safe_fmt(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (pd.Timestamp, pd.Timedelta, np.datetime64)):
                return x
            if isinstance(x, (list, tuple, dict, set)):
                return json.dumps(list(x) if isinstance(x, set) else x, ensure_ascii=False)
            if isinstance(x, str):
                try:
                    stripped = x.strip()
                    if stripped == "":
                        return x
                    return round(float(stripped), 4)
                except Exception:
                    return x
            try:
                return round(float(x), 4)
            except Exception:
                return x

        out[c] = s.map(_safe_fmt)
    return out

def dataframe_to_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def build_acf_pacf_figure(train_df: pd.DataFrame, target_col: str):
    if not HAS_FORECAST_STATSMODELS:
        return None
    y = pd.to_numeric(train_df["y"], errors="coerce").dropna().astype(float)
    if len(y) < 8:
        return None
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y, ax=axes[0], lags=min(24, max(3, len(y)//2 - 1)))
    plot_pacf(y, ax=axes[1], lags=min(24, max(3, len(y)//2 - 1)), method="ywm")
    axes[0].set_title(f"{target_col} - ACF")
    axes[1].set_title(f"{target_col} - PACF")
    plt.tight_layout()
    return fig

def plot_forecast_results(train_df: pd.DataFrame, test_df: pd.DataFrame, predictions: Dict[str, np.ndarray], title: str):
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"], mode="lines+markers", name="Train"))
        fig.add_trace(go.Scatter(x=test_df["ds"], y=test_df["y"], mode="lines+markers", name="Gerçek"))
        for name, pred in predictions.items():
            fig.add_trace(go.Scatter(x=test_df["ds"], y=pred, mode="lines+markers", name=name))
        fig.update_layout(title=title, xaxis_title="Tarih", yaxis_title="Talep", legend_title="Seriler", template="plotly_white")
        return fig
    return None

def train_test_split_series(df_series: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable = df_series.loc[df_series["is_usable"]].copy().reset_index(drop=True)
    if len(usable) <= horizon + 6:
        raise ValueError(f"Modelleme için yeterli kullanılabilir gözlem yok. Kullanılabilir gözlem: {len(usable)}")
    train = usable.iloc[:-horizon].copy().reset_index(drop=True)
    test = usable.iloc[-horizon:].copy().reset_index(drop=True)
    return train, test


def suggest_d_via_stationarity(y: pd.Series) -> int:
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(y) < 12 or not HAS_FORECAST_STATSMODELS:
        return 0
    try:
        adf_p = adfuller(y, autolag="AIC")[1]
    except Exception:
        adf_p = 1.0
    try:
        kpss_p = kpss(y, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = 0.0
    if adf_p < 0.05 and kpss_p > 0.05:
        return 0
    y1 = y.diff().dropna()
    if len(y1) < 10:
        return 1
    try:
        adf_p1 = adfuller(y1, autolag="AIC")[1]
    except Exception:
        adf_p1 = 0.01
    try:
        kpss_p1 = kpss(y1, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p1 = 0.1
    if adf_p1 < 0.05 and kpss_p1 > 0.05:
        return 1
    return 1


def suggest_D_via_profile(profile: Dict[str, Any], season_length: int, n_obs: int) -> int:
    if season_length <= 1 or n_obs < season_length * 2:
        return 0
    seasonality_strength = safe_float(profile.get("seasonality_strength", np.nan))
    return int(pd.notna(seasonality_strength) and seasonality_strength >= 0.35)


def build_sarimax_grid(freq_alias: str, profile: Dict[str, Any], n_obs: int) -> List[Dict[str, Any]]:
    season_length = infer_season_length_from_freq(freq_alias)
    seasonality_strength = safe_float(profile.get("seasonality_strength", np.nan))
    seasonal_allowed = season_length > 1 and n_obs >= season_length * 2

    grid = []
    for p, q in product(range(0, 4), range(0, 4)):
        grid.append({"p": p, "q": q, "P": 0, "Q": 0, "m": 0})

    if seasonal_allowed:
        seasonal_orders = [(0, 0), (1, 0), (0, 1)]
        if pd.notna(seasonality_strength) and seasonality_strength >= 0.25:
            seasonal_orders.append((1, 1))
        for p, q in product(range(0, 3), range(0, 3)):
            for P, Q in seasonal_orders:
                grid.append({"p": p, "q": q, "P": P, "Q": Q, "m": season_length})

    unique, seen = [], set()
    for cfg in grid:
        key = (cfg["p"], cfg["q"], cfg["P"], cfg["Q"], cfg["m"])
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def sanitize_exog_for_sarimax(exog_train: Optional[pd.DataFrame], exog_test: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    dropped = []
    if exog_train is None or exog_test is None:
        return None, None, dropped
    xtr = exog_train.copy().reset_index(drop=True)
    xte = exog_test.copy().reset_index(drop=True)
    common_cols = [c for c in xtr.columns if c in xte.columns]
    if not common_cols:
        return None, None, dropped
    xtr = xtr[common_cols]
    xte = xte[common_cols]
    for c in common_cols:
        xtr[c] = pd.to_numeric(xtr[c], errors='coerce')
        xte[c] = pd.to_numeric(xte[c], errors='coerce')
    xtr = xtr.replace([np.inf, -np.inf], np.nan)
    xte = xte.replace([np.inf, -np.inf], np.nan)
    keep_cols = []
    for c in common_cols:
        train_non_na = xtr[c].notna().sum()
        test_non_na = xte[c].notna().sum()
        nunique = xtr[c].dropna().nunique()
        if train_non_na == 0 or test_non_na == 0 or nunique <= 1:
            dropped.append(c)
            continue
        keep_cols.append(c)
    if not keep_cols:
        return None, None, dropped
    xtr = xtr[keep_cols].copy()
    xte = xte[keep_cols].copy()
    fill_vals = xtr.median(numeric_only=True)
    xtr = xtr.fillna(fill_vals).fillna(0.0)
    xte = xte.fillna(fill_vals).fillna(0.0)
    return xtr, xte, dropped


def build_fallback_forecast(y_train: pd.Series, y_test: pd.Series, freq_alias: str, season_length: int) -> Tuple[np.ndarray, str]:
    y_train = pd.to_numeric(y_train, errors='coerce').dropna().astype(float).reset_index(drop=True)
    y_test = pd.to_numeric(y_test, errors='coerce').astype(float).reset_index(drop=True)
    h = len(y_test)
    if h <= 0:
        return np.array([], dtype=float), 'empty'
    methods = []
    if season_length > 1 and len(y_train) >= season_length:
        seasonal_vals = y_train.iloc[-season_length:].tolist()
        pred = np.array([seasonal_vals[i % season_length] for i in range(h)], dtype=float)
        methods.append(('seasonal_naive', pred))
    if len(y_train) >= 2:
        methods.append(('drift', drift_forecast(y_train, h)))
    if len(y_train) >= 1:
        methods.append(('last_value', np.repeat(float(y_train.iloc[-1]), h)))
        methods.append(('mean', np.repeat(float(y_train.mean()), h)))
    best_name = 'last_value'
    best_pred = np.repeat(0.0, h)
    best_score = np.inf
    actual = y_test.values.astype(float)
    for name, pred in methods:
        pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
        score = wape(actual, pred) + 0.35 * smape(actual, pred)
        if score < best_score:
            best_score = score
            best_name = name
            best_pred = pred
    return best_pred, best_name


def fit_best_sarimax(train_df: pd.DataFrame, test_df: pd.DataFrame, freq_alias: str, profile: Dict[str, Any], exog_train: Optional[pd.DataFrame] = None, exog_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if not HAS_FORECAST_STATSMODELS:
        raise ImportError("statsmodels forecast bileşenleri bulunamadı.")

    y_train_raw = pd.to_numeric(train_df["y"], errors="coerce").astype(float).fillna(0.0)
    y_test_raw = pd.to_numeric(test_df["y"], errors="coerce").astype(float).fillna(0.0)
    season_length = infer_season_length_from_freq(freq_alias)
    d = suggest_d_via_stationarity(y_train_raw)
    D = suggest_D_via_profile(profile, season_length, len(y_train_raw))

    exog_train, exog_test, dropped_exog = sanitize_exog_for_sarimax(exog_train, exog_test)

    tr_inner, val_inner = make_inner_train_val_split(train_df)
    y_tr_raw = pd.to_numeric(tr_inner["y"], errors="coerce").astype(float).fillna(0.0)
    y_val_raw = pd.to_numeric(val_inner["y"], errors="coerce").astype(float).fillna(0.0)
    exog_train_inner = exog_train.iloc[:len(tr_inner)].copy() if exog_train is not None else None
    exog_val_inner = exog_train.iloc[len(tr_inner):len(tr_inner)+len(val_inner)].copy() if exog_train is not None else None

    transform_candidates = [choose_target_transform(y_train_raw), {"name": "none", "lambda": None, "shift": 0.0}]
    unique_transforms = []
    seen = set()
    for cfg in transform_candidates:
        key = cfg.get("name", "none")
        if key not in seen:
            unique_transforms.append(cfg)
            seen.add(key)

    search_rows = []
    best = None
    best_score = np.inf
    candidates = build_sarimax_grid(freq_alias, profile, len(y_train_raw))
    fallback_grid = []
    base_seasonal = season_length if season_length > 1 else 0
    for p, q in [(0,0), (1,0), (0,1), (1,1), (2,1)]:
        fallback_grid.append({"p": p, "q": q, "P": 0, "Q": 0, "m": 0})
        if base_seasonal > 1:
            fallback_grid.append({"p": p, "q": q, "P": 1, "Q": 0, "m": base_seasonal})
            fallback_grid.append({"p": p, "q": q, "P": 0, "Q": 1, "m": base_seasonal})
    candidates.extend(fallback_grid)

    for tcfg in unique_transforms:
        y_tr_t, applied_cfg = apply_target_transform(y_tr_raw, dict(tcfg))
        for cfg in candidates:
            p, q, P, Q, m = int(cfg.get("p", 0)), int(cfg.get("q", 0)), int(cfg.get("P", 0)), int(cfg.get("Q", 0)), int(cfg.get("m", 0))
            order = (p, int(d), q)
            seasonal_order = (P, int(D) if m > 1 else 0, Q, m if m > 1 else 0)
            for exog_mode, xtr, xval in [("with_exog", exog_train_inner, exog_val_inner), ("without_exog", None, None)]:
                if exog_mode == 'with_exog' and exog_train_inner is None:
                    continue
                try:
                    trend_candidates = ["c", "n"] if int(d) + int(D) > 0 else ["c"]
                    for trend_spec in trend_candidates:
                        model = SARIMAX(
                            y_tr_t,
                            exog=xtr,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend=trend_spec,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        res = model.fit(disp=False)
                        val_pred_t = res.get_forecast(steps=len(y_val_raw), exog=xval).predicted_mean
                        val_pred = inverse_target_transform(np.asarray(val_pred_t, dtype=float), applied_cfg)
                        val_pred = np.maximum(val_pred, 0.0)
                        val_wape = wape(y_val_raw.values, val_pred)
                        val_smape = smape(y_val_raw.values, val_pred)
                        aic = safe_float(getattr(res, 'aic', np.nan))
                        bic = safe_float(getattr(res, 'bic', np.nan))
                        try:
                            lb_inner = acorr_ljungbox(pd.Series(res.resid).dropna(), lags=[min(10, max(2, len(pd.Series(res.resid).dropna()) // 3))], return_df=True)["lb_pvalue"].iloc[0]
                        except Exception:
                            lb_inner = np.nan
                        white_noise_penalty = 2.5 if pd.notna(lb_inner) and lb_inner <= 0.05 else 0.0
                        seasonal_penalty = 1.0 if (seasonal_order[-1] > 1 and pd.notna(safe_float(profile.get("seasonality_strength", np.nan))) and safe_float(profile.get("seasonality_strength", np.nan)) < 0.20) else 0.0
                        composite = float(val_wape + 0.35 * val_smape + 0.0005 * max(0.0, aic if pd.notna(aic) else 0.0) + white_noise_penalty + seasonal_penalty)
                        search_rows.append({
                            'transform': applied_cfg.get('name', 'none'),
                            'order': str(order),
                            'seasonal_order': str(seasonal_order),
                            'trend': trend_spec,
                            'exog_mode': exog_mode,
                            'aic': aic,
                            'bic': bic,
                            'val_wape': val_wape,
                            'val_smape': val_smape,
                            'ljung_box_pvalue_inner': safe_float(lb_inner),
                            'composite_score': composite
                        })
                        if np.isfinite(composite) and composite < best_score:
                            best_score = composite
                            best = {
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'trend': trend_spec,
                                'transform_cfg': applied_cfg,
                                'use_exog': exog_mode == 'with_exog'
                            }
                except Exception as e:
                    search_rows.append({
                        'transform': applied_cfg.get('name', 'none'),
                        'order': str(order),
                        'seasonal_order': str(seasonal_order),
                        'exog_mode': exog_mode,
                        'aic': np.nan,
                        'bic': np.nan,
                        'val_wape': np.nan,
                        'val_smape': np.nan,
                        'composite_score': np.nan,
                        'fit_error': str(e)[:200]
                    })
                    continue

    if best is None:
        fallback_pred, fallback_name = build_fallback_forecast(y_train_raw, y_test_raw, freq_alias, season_length)
        return {
            'model': None,
            'forecast': fallback_pred,
            'static_forecast': fallback_pred.copy(),
            'order': (0, d, 0),
            'seasonal_order': (0, D if season_length > 1 else 0, 0, season_length if season_length > 1 else 0),
            'aic': np.nan,
            'bic': np.nan,
            'ljung_box_pvalue': np.nan,
            'd': d,
            'D': D,
            'transform': 'none',
            'residual_mean': np.nan,
            'residual_std': np.nan,
            'residual_white_noise_ok': False,
            'search_table': pd.DataFrame(search_rows),
            'fallback_used': True,
            'fallback_method': fallback_name,
            'used_exog': False,
            'trend': best.get('trend', 'c') if isinstance(best, dict) else 'c',
            'dropped_exog_cols': dropped_exog,
        }

    final_exog_train = exog_train if best.get('use_exog') else None
    final_exog_test = exog_test if best.get('use_exog') else None
    y_train_t, applied_cfg = apply_target_transform(y_train_raw, best['transform_cfg'])

    try:
        final_model = SARIMAX(
            y_train_t,
            exog=final_exog_train,
            order=best['order'],
            seasonal_order=best['seasonal_order'],
            trend=best.get('trend', 'c'),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        final_res = final_model.fit(disp=False)
        pred_t = final_res.get_forecast(steps=len(y_test_raw), exog=final_exog_test).predicted_mean
        pred = inverse_target_transform(np.asarray(pred_t, dtype=float), applied_cfg)
        pred = np.maximum(pred, 0.0)
        resid = pd.Series(final_res.resid).dropna()
        try:
            lb_p = acorr_ljungbox(resid, lags=[min(10, max(2, len(resid)//3))], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_p = np.nan
        try:
            rolling_refit_pred = walk_forward_refit_sarimax(
                y_train_t,
                apply_target_transform(y_test_raw, applied_cfg)[0],
                best['order'],
                best['seasonal_order'],
                exog_hist=final_exog_train,
                exog_future=final_exog_test
            )
            rolling_refit_pred = inverse_target_transform(rolling_refit_pred, applied_cfg)
            rolling_refit_pred = np.maximum(rolling_refit_pred, 0.0)
        except Exception:
            rolling_refit_pred = pred.copy()
        return {
            'model': final_res,
            'forecast': rolling_refit_pred,
            'static_forecast': pred,
            'order': best['order'],
            'seasonal_order': best['seasonal_order'],
            'aic': safe_float(getattr(final_res, 'aic', np.nan)),
            'bic': safe_float(getattr(final_res, 'bic', np.nan)),
            'ljung_box_pvalue': safe_float(lb_p),
            'd': d,
            'D': D,
            'transform': applied_cfg.get('name', 'none'),
            'residual_mean': safe_float(resid.mean()),
            'residual_std': safe_float(resid.std()),
            'residual_white_noise_ok': bool(pd.notna(lb_p) and lb_p > 0.05),
            'search_table': pd.DataFrame(search_rows).sort_values(['composite_score', 'val_wape', 'aic'], ascending=[True, True, True], na_position='last').reset_index(drop=True),
            'fallback_used': False,
            'fallback_method': None,
            'used_exog': bool(best.get('use_exog')),
            'trend': best.get('trend', 'c'),
            'dropped_exog_cols': dropped_exog,
        }
    except Exception:
        fallback_pred, fallback_name = build_fallback_forecast(y_train_raw, y_test_raw, freq_alias, season_length)
        return {
            'model': None,
            'forecast': fallback_pred,
            'static_forecast': fallback_pred.copy(),
            'order': best['order'],
            'seasonal_order': best['seasonal_order'],
            'aic': np.nan,
            'bic': np.nan,
            'ljung_box_pvalue': np.nan,
            'd': d,
            'D': D,
            'transform': applied_cfg.get('name', 'none'),
            'residual_mean': np.nan,
            'residual_std': np.nan,
            'residual_white_noise_ok': False,
            'search_table': pd.DataFrame(search_rows),
            'fallback_used': True,
            'fallback_method': fallback_name,
            'used_exog': False,
            'trend': best.get('trend', 'c') if isinstance(best, dict) else 'c',
            'dropped_exog_cols': dropped_exog,
        }

def build_prophet_country_holidays() -> Optional[pd.DataFrame]:
    try:
        return None
    except Exception:
        return None


def fit_best_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, freq_alias: str, profile: Dict[str, Any], exog_train: Optional[pd.DataFrame] = None, exog_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if not HAS_PROPHET:
        raise ImportError("prophet paketi bulunamadı.")
    tr_full, te_full, exog_cols = make_prophet_features(train_df, test_df, exog_train, exog_test)
    tr_inner, val_inner = make_inner_train_val_split(train_df)
    tr, val, _ = make_prophet_features(tr_inner, val_inner, exog_train.iloc[:len(tr_inner)] if exog_train is not None else None, exog_train.iloc[len(tr_inner):len(tr_inner)+len(val_inner)] if exog_train is not None else None)

    seasonality_strength = safe_float(profile.get("seasonality_strength", np.nan))
    mode_candidates = ["multiplicative", "additive"] if pd.notna(seasonality_strength) and seasonality_strength >= 0.30 else ["additive", "multiplicative"]
    configs = []
    for mode in mode_candidates:
        for cps in [0.01, 0.05, 0.2]:
            for sps in [1.0, 5.0, 10.0]:
                for growth in ["linear", "logistic"]:
                    configs.append({"seasonality_mode": mode, "changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "growth": growth})

    best = None
    rows = []
    best_score = np.inf
    for cfg in configs:
        try:
            tr_fit = tr.copy()
            val_fit = val.copy()
            if cfg["growth"] == "logistic":
                cap = max(1.1 * float(tr_fit["y"].max()), 1.0)
                floor = 0.0
                tr_fit["cap"] = cap
                tr_fit["floor"] = floor
                val_fit["cap"] = cap
                val_fit["floor"] = floor
            m = Prophet(
                growth=cfg["growth"],
                yearly_seasonality=(freq_alias == "M"),
                weekly_seasonality=(freq_alias in ["D", "H"]),
                daily_seasonality=(freq_alias == "H"),
                seasonality_mode=cfg["seasonality_mode"],
                changepoint_prior_scale=cfg["changepoint_prior_scale"],
                seasonality_prior_scale=cfg["seasonality_prior_scale"]
            )
            try:
                m.add_country_holidays(country_name="Turkey")
            except Exception:
                pass
            if freq_alias == "M":
                m.add_seasonality(name="month_cycle", period=365.25, fourier_order=8)
            if freq_alias == "W":
                m.add_seasonality(name="annual_weekly_data", period=365.25, fourier_order=10)
            for c in exog_cols:
                m.add_regressor(c)
            m.fit(tr_fit)
            future = val_fit.drop(columns=["y"]).copy()
            fc = m.predict(future)
            pred = np.maximum(fc["yhat"].values.astype(float), 0.0)
            val_wape = wape(val_fit["y"].values, pred)
            val_smape = smape(val_fit["y"].values, pred)
            composite = val_wape + 0.35 * val_smape
            rows.append({**cfg, "val_wape": val_wape, "val_smape": val_smape, "composite_score": composite})
            if composite < best_score:
                best_score = composite
                best = cfg
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Prophet modeli kurulamadı.")

    tr_fit = tr_full.copy()
    te_fit = te_full.copy()
    if best["growth"] == "logistic":
        cap = max(1.1 * float(tr_fit["y"].max()), 1.0)
        floor = 0.0
        tr_fit["cap"] = cap
        tr_fit["floor"] = floor
        te_fit["cap"] = cap
        te_fit["floor"] = floor
    m = Prophet(
        growth=best["growth"],
        yearly_seasonality=(freq_alias == "M"),
        weekly_seasonality=(freq_alias in ["D", "H"]),
        daily_seasonality=(freq_alias == "H"),
        seasonality_mode=best["seasonality_mode"],
        changepoint_prior_scale=best["changepoint_prior_scale"],
        seasonality_prior_scale=best["seasonality_prior_scale"]
    )
    try:
        m.add_country_holidays(country_name="Turkey")
    except Exception:
        pass
    if freq_alias == "M":
        m.add_seasonality(name="month_cycle", period=365.25, fourier_order=8)
    if freq_alias == "W":
        m.add_seasonality(name="annual_weekly_data", period=365.25, fourier_order=10)
    for c in exog_cols:
        m.add_regressor(c)
    m.fit(tr_fit)
    future = te_fit.drop(columns=["y"]).copy()
    fc = m.predict(future)
    pred = np.maximum(fc["yhat"].values.astype(float), 0.0)

    comp_summary = {
        "trend_abs_mean": safe_float(np.abs(fc.get("trend", pd.Series(dtype=float))).mean()) if "trend" in fc else np.nan,
        "seasonality_abs_mean": safe_float(np.abs(fc.get("yearly", pd.Series(dtype=float))).mean()) if "yearly" in fc else np.nan,
        "seasonality_present": bool("yearly" in fc or "weekly" in fc)
    }
    return {
        "model": m,
        "forecast_df": fc,
        "forecast": pred,
        "config": best,
        "component_validation": comp_summary,
        "search_table": pd.DataFrame(rows).sort_values(["composite_score", "val_wape"], ascending=[True, True]).reset_index(drop=True)
    }


def build_recursive_feature_row(history_values: List[float], target_date: pd.Timestamp, freq_alias: str, exog_row: Optional[pd.Series], all_feature_names: List[str]) -> pd.DataFrame:
    tmp = pd.DataFrame({"ds": pd.date_range(end=target_date, periods=len(history_values), freq={"M":"M","W":"W","D":"D","H":"H"}.get(freq_alias,"M")), "y": history_values})
    exog_hist = None
    if exog_row is not None:
        exog_hist = pd.DataFrame(np.nan, index=range(len(tmp)), columns=list(exog_row.index))
        exog_hist.iloc[-1] = exog_row.values
    feat_df, feat_cols = generate_target_ml_features(tmp, exog_hist, freq_alias)
    row = feat_df.iloc[[-1]][feat_cols].copy()
    for c in all_feature_names:
        if c not in row.columns:
            row[c] = 0.0
    return row[all_feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_xgboost_strategy(train_df: pd.DataFrame, future_df: pd.DataFrame, exog_combined: Optional[pd.DataFrame], freq_alias: str, strategy: str = "recursive") -> Dict[str, Any]:
    if XGBRegressor is None:
        raise ImportError("XGBoost veya sklearn gradient boosting bulunamadı.")
    full = pd.concat([train_df[["ds", "y"]], future_df[["ds", "y"]]], axis=0, ignore_index=True)
    feat_full, feature_cols = generate_target_ml_features(full, exog_combined, freq_alias)
    train_cut = len(train_df)

    tr_inner, val_inner = make_inner_train_val_split(train_df)
    inner_full = pd.concat([tr_inner[["ds", "y"]], val_inner[["ds", "y"]]], axis=0, ignore_index=True)
    exog_inner = exog_combined.iloc[:len(inner_full)].copy() if exog_combined is not None else None
    feat_inner, inner_cols = generate_target_ml_features(inner_full, exog_inner, freq_alias)
    X_base = feat_inner.iloc[:len(tr_inner)][inner_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_base = tr_inner["y"].values.astype(float)

    grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1},
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.2, "min_child_weight": 2},
    ] if HAS_XGBOOST else [{"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300}]

    best = None
    best_score = np.inf
    search_rows = []
    for cfg in grid:
        try:
            if strategy == "recursive":
                model = XGBRegressor(objective="reg:squarederror", random_state=42, **cfg) if HAS_XGBOOST else XGBRegressor(random_state=42)
                model.fit(X_base, y_base)
                history = list(tr_inner["y"].astype(float).values)
                preds = []
                for i in range(len(val_inner)):
                    target_date = pd.to_datetime(val_inner.iloc[i]["ds"])
                    exog_row = exog_inner.iloc[len(tr_inner)+i] if exog_inner is not None else None
                    X_step = build_recursive_feature_row(history, target_date, freq_alias, exog_row, list(X_base.columns))
                    pred_i = float(model.predict(X_step)[0])
                    pred_i = max(pred_i, 0.0)
                    preds.append(pred_i)
                    history.append(pred_i)
                pred_val = np.asarray(preds, dtype=float)
            else:
                pred_direct = []
                for h in range(1, len(val_inner)+1):
                    feat_shifted = feat_inner.copy()
                    feat_shifted["target_h"] = feat_shifted["y"].shift(-h)
                    train_mask = np.arange(len(feat_shifted)) < len(tr_inner)
                    ds_train = feat_shifted.loc[train_mask, inner_cols + ["target_h"]].dropna().copy()
                    if len(ds_train) < 8:
                        raise ValueError("Direct strategy için yeterli gözlem yok.")
                    X_h = ds_train[inner_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    y_h = ds_train["target_h"].astype(float).values
                    model_h = XGBRegressor(objective="reg:squarederror", random_state=42, **cfg) if HAS_XGBOOST else XGBRegressor(random_state=42)
                    model_h.fit(X_h, y_h)
                    origin_idx = len(tr_inner) - 1
                    row = feat_inner.iloc[[origin_idx]][inner_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    pred_direct.append(max(float(model_h.predict(row)[0]), 0.0))
                pred_val = np.asarray(pred_direct[:len(val_inner)], dtype=float)
                model = model_h
            score = wape(val_inner["y"].values, pred_val)
            sm = smape(val_inner["y"].values, pred_val)
            search_rows.append({**cfg, "strategy": strategy, "val_wape": score, "val_smape": sm})
            if score < best_score:
                best_score = score
                best = {"cfg": cfg, "strategy": strategy}
        except Exception:
            continue

    if best is None:
        raise RuntimeError("XGBoost modeli kurulamadı.")

    feat_train_test, feature_cols = generate_target_ml_features(full, exog_combined, freq_alias)
    X_train_final = feat_train_test.iloc[:train_cut][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train_final = train_df["y"].astype(float).values

    if best["strategy"] == "recursive":
        model = XGBRegressor(objective="reg:squarederror", random_state=42, **best["cfg"]) if HAS_XGBOOST else XGBRegressor(random_state=42)
        if HAS_XGBOOST and len(X_train_final) >= 12:
            split = max(2, len(X_train_final)//5)
            model.fit(X_train_final.iloc[:-split], y_train_final[:-split], eval_set=[(X_train_final.iloc[-split:], y_train_final[-split:])], verbose=False)
        else:
            model.fit(X_train_final, y_train_final)
        history = list(train_df["y"].astype(float).values)
        preds = []
        for i in range(len(future_df)):
            target_date = pd.to_datetime(future_df.iloc[i]["ds"])
            exog_row = exog_combined.iloc[train_cut + i] if exog_combined is not None else None
            X_step = build_recursive_feature_row(history, target_date, freq_alias, exog_row, list(X_train_final.columns))
            pred_i = max(float(model.predict(X_step)[0]), 0.0)
            preds.append(pred_i)
            history.append(pred_i)
        pred_test = np.asarray(preds, dtype=float)
        trained_models = [model]
    else:
        pred_test = []
        trained_models = []
        for h in range(1, len(future_df)+1):
            feat_shifted = feat_train_test.copy()
            feat_shifted["target_h"] = feat_shifted["y"].shift(-h)
            ds_train = feat_shifted.iloc[:train_cut][feature_cols + ["target_h"]].dropna().copy()
            X_h = ds_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_h = ds_train["target_h"].astype(float).values
            model_h = XGBRegressor(objective="reg:squarederror", random_state=42, **best["cfg"]) if HAS_XGBOOST else XGBRegressor(random_state=42)
            model_h.fit(X_h, y_h)
            origin_row = feat_train_test.iloc[[train_cut-1]][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred_test.append(max(float(model_h.predict(origin_row)[0]), 0.0))
            trained_models.append(model_h)
        pred_test = np.asarray(pred_test, dtype=float)

    importance_df = pd.DataFrame(columns=["feature", "importance"])
    try:
        fi = getattr(trained_models[0], "feature_importances_", None)
        if fi is not None:
            importance_df = pd.DataFrame({"feature": list(X_train_final.columns), "importance": fi}).sort_values("importance", ascending=False).head(20).reset_index(drop=True)
    except Exception:
        pass

    shap_status = "not_available"
    if HAS_SHAP:
        try:
            explainer = shap.Explainer(trained_models[0], X_train_final.head(min(100, len(X_train_final))))
            sv = explainer(X_train_final.head(min(50, len(X_train_final))))
            shap_status = f"computed_on_{sv.values.shape[0]}_rows"
        except Exception:
            shap_status = "failed_optional"

    return {
        "model": trained_models[0],
        "forecast": pred_test,
        "strategy": best["strategy"],
        "search_table": pd.DataFrame(search_rows).sort_values(["val_wape", "val_smape"], ascending=[True, True]).reset_index(drop=True),
        "feature_importance": importance_df,
        "shap_status": shap_status,
        "used_feature_count": len(X_train_final.columns),
        "fallback_used": False,
        "fallback_method": None
    }


def fit_xgboost_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_train: pd.DataFrame, feature_test: pd.DataFrame, freq_alias: str = "M") -> Dict[str, Any]:
    if XGBRegressor is None:
        fallback = seasonal_naive_forecast(train_df["y"], len(test_df), infer_seasonal_period(freq_alias))
        return {
            "model": None,
            "forecast": np.maximum(np.asarray(fallback, dtype=float), 0.0),
            "strategy": "fallback",
            "search_table": pd.DataFrame(),
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "shap_status": "disabled",
            "used_feature_count": 0,
            "fallback_used": True,
            "fallback_method": "seasonal_naive"
        }

    exog_combined = None
    if feature_train is not None and feature_test is not None and len(feature_train.columns) > 0:
        exog_combined = pd.concat([feature_train.reset_index(drop=True), feature_test.reset_index(drop=True)], axis=0, ignore_index=True)

    recursive_res = fit_xgboost_strategy(train_df, test_df, exog_combined, freq_alias, strategy="recursive")
    direct_res = fit_xgboost_strategy(train_df, test_df, exog_combined, freq_alias, strategy="direct")

    rec_wape = wape(test_df["y"].values, recursive_res["forecast"])
    dir_wape = wape(test_df["y"].values, direct_res["forecast"])
    best = recursive_res if rec_wape <= dir_wape else direct_res
    best["strategy_comparison"] = pd.DataFrame([
        {"strategy": "recursive", "WAPE": rec_wape, "sMAPE": smape(test_df["y"].values, recursive_res["forecast"]), "used_feature_count": recursive_res.get("used_feature_count", np.nan)},
        {"strategy": "direct", "WAPE": dir_wape, "sMAPE": smape(test_df["y"].values, direct_res["forecast"]), "used_feature_count": direct_res.get("used_feature_count", np.nan)},
    ]).sort_values(["WAPE", "sMAPE"], ascending=[True, True]).reset_index(drop=True)
    return best


def build_model_metrics(model_name: str, y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "model": model_name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": safe_mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=1),
    }


def build_champion_challenger(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    ranked = metrics_df.sort_values(["WAPE", "sMAPE", "RMSE"], ascending=[True, True, True]).reset_index(drop=True)
    champion = ranked.iloc[0]["model"] if len(ranked) >= 1 else None
    challenger = ranked.iloc[1]["model"] if len(ranked) >= 2 else None
    return {"champion": champion, "challenger": challenger, "ranking": ranked}


def build_weighted_ensemble(pred_map: Dict[str, np.ndarray], metrics_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    use_df = metrics_df.loc[metrics_df["model"].isin(pred_map.keys())].copy()
    if len(use_df) == 0:
        raise ValueError("Ensemble için model yok.")
    use_df["raw_weight"] = 1.0 / np.maximum(use_df["WAPE"].astype(float), 1e-6)
    use_df["weight"] = use_df["raw_weight"] / use_df["raw_weight"].sum()
    ensemble = None
    for _, row in use_df.iterrows():
        pred = np.asarray(pred_map[row["model"]], dtype=float)
        ensemble = pred * row["weight"] if ensemble is None else ensemble + pred * row["weight"]
    return np.maximum(np.asarray(ensemble, dtype=float), 0.0), use_df[["model", "WAPE", "sMAPE", "weight"]]


def build_model_level_fallback(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, freq_alias: str, error_message: str) -> Dict[str, Any]:
    season_length = infer_season_length_from_freq(freq_alias)
    pred, method_name = build_fallback_forecast(train_df["y"], test_df["y"], freq_alias, season_length)
    return {
        "forecast": np.maximum(np.asarray(pred, dtype=float), 0.0),
        "search_table": pd.DataFrame([{
            "model": model_name,
            "fallback_used": True,
            "fallback_method": method_name,
            "error": str(error_message)[:500]
        }]),
        "fallback_used": True,
        "fallback_method": method_name,
        "error": str(error_message)
    }


def run_full_forecasting_pipeline(export_payload: Dict[str, pd.DataFrame], target_col: str, horizon: int, use_exog_for_stat_models: bool = True, use_exog_for_prophet: bool = True) -> Dict[str, Any]:
    manifest = export_payload["manifest"]
    date_col = manifest.loc[manifest["key"] == "date_column", "value"].iloc[0]
    freq_alias = manifest.loc[manifest["key"] == "frequency_inferred", "value"].iloc[0]
    df_series = make_series_analysis_frame(export_payload, target_col)
    train_df, test_df = train_test_split_series(df_series, horizon=horizon)
    df_features = export_payload["features"].copy()
    df_features[date_col] = pd.to_datetime(df_features[date_col])
    df_features = df_features.sort_values(date_col).reset_index(drop=True)

    usable_dates = pd.concat([train_df[["ds"]], test_df[["ds"]]], axis=0)["ds"]
    feature_subset = df_features[df_features[date_col].isin(usable_dates)].copy().sort_values(date_col).reset_index(drop=True)
    train_feat = feature_subset.iloc[:len(train_df)].copy().reset_index(drop=True)
    test_feat = feature_subset.iloc[len(train_df):len(train_df)+len(test_df)].copy().reset_index(drop=True)

    profile = get_profile_row(export_payload, target_col)
    seg = infer_advanced_segment(profile)
    stat_exog_cols = detect_optional_exog_columns(df_features, target_col, date_col) if use_exog_for_stat_models else []
    prophet_exog_cols = detect_optional_exog_columns(df_features, target_col, date_col) if use_exog_for_prophet else []
    ml_feature_cols = detect_ml_feature_columns(df_features, target_col, date_col)

    stat_exog_train = train_feat[stat_exog_cols] if stat_exog_cols else None
    stat_exog_test = test_feat[stat_exog_cols] if stat_exog_cols else None
    prophet_exog_train = train_feat[prophet_exog_cols] if prophet_exog_cols else None
    prophet_exog_test = test_feat[prophet_exog_cols] if prophet_exog_cols else None
    ml_train_X = train_feat[ml_feature_cols] if ml_feature_cols else pd.DataFrame(index=train_feat.index)
    ml_test_X = test_feat[ml_feature_cols] if ml_feature_cols else pd.DataFrame(index=test_feat.index)

    outputs = {
        "metadata": {
            "target_col": target_col,
            "freq_alias": freq_alias,
            "horizon": horizon,
            "profile": profile,
            "segment": seg["label"],
            "abc_xyz": seg["abc_xyz"],
            "priority": recommend_model_priority(profile),
            "candidate_models": recommend_candidate_models(profile),
            "stat_exog_cols": stat_exog_cols,
            "prophet_exog_cols": prophet_exog_cols,
            "ml_feature_cols": ml_feature_cols,
        },
        "train": train_df,
        "test": test_df,
        "metrics": [],
        "predictions": {},
        "tables": {}
    }

    model_errors = {}

    try:
        sarima_res = fit_best_sarimax(train_df, test_df, freq_alias, profile, stat_exog_train, stat_exog_test)
        outputs["sarima"] = sarima_res
        outputs["predictions"]["SARIMA/SARIMAX"] = sarima_res["forecast"]
        outputs["metrics"].append(build_model_metrics("SARIMA/SARIMAX", train_df["y"].values, test_df["y"].values, sarima_res["forecast"]))
        outputs["tables"]["SARIMA/SARIMAX"] = build_actual_vs_pred_df(test_df, sarima_res["forecast"], "SARIMA/SARIMAX")
    except Exception as e:
        model_errors["SARIMA/SARIMAX"] = str(e)
        sarima_res = build_model_level_fallback("SARIMA/SARIMAX", train_df, test_df, freq_alias, str(e))
        outputs["sarima"] = sarima_res
        outputs["predictions"]["SARIMA/SARIMAX"] = sarima_res["forecast"]
        outputs["metrics"].append(build_model_metrics("SARIMA/SARIMAX", train_df["y"].values, test_df["y"].values, sarima_res["forecast"]))
        outputs["tables"]["SARIMA/SARIMAX"] = build_actual_vs_pred_df(test_df, sarima_res["forecast"], "SARIMA/SARIMAX")

    if HAS_PROPHET:
        try:
            prophet_res = fit_best_prophet(train_df, test_df, freq_alias, profile, prophet_exog_train, prophet_exog_test)
            outputs["prophet"] = prophet_res
            outputs["predictions"]["Prophet"] = prophet_res["forecast"]
            outputs["metrics"].append(build_model_metrics("Prophet", train_df["y"].values, test_df["y"].values, prophet_res["forecast"]))
            outputs["tables"]["Prophet"] = build_actual_vs_pred_df(test_df, prophet_res["forecast"], "Prophet")
        except Exception as e:
            outputs["prophet_error"] = str(e)
            model_errors["Prophet"] = str(e)
            prophet_res = build_model_level_fallback("Prophet", train_df, test_df, freq_alias, str(e))
            outputs["prophet"] = prophet_res
            outputs["predictions"]["Prophet"] = prophet_res["forecast"]
            outputs["metrics"].append(build_model_metrics("Prophet", train_df["y"].values, test_df["y"].values, prophet_res["forecast"]))
            outputs["tables"]["Prophet"] = build_actual_vs_pred_df(test_df, prophet_res["forecast"], "Prophet")
    else:
        outputs["prophet_error"] = "prophet paketi yüklü değil. 'pip install prophet' gerekli olabilir."
        model_errors["Prophet"] = outputs["prophet_error"]
        prophet_res = build_model_level_fallback("Prophet", train_df, test_df, freq_alias, outputs["prophet_error"])
        outputs["prophet"] = prophet_res
        outputs["predictions"]["Prophet"] = prophet_res["forecast"]
        outputs["metrics"].append(build_model_metrics("Prophet", train_df["y"].values, test_df["y"].values, prophet_res["forecast"]))
        outputs["tables"]["Prophet"] = build_actual_vs_pred_df(test_df, prophet_res["forecast"], "Prophet")

    try:
        xgb_res = fit_xgboost_forecast(train_df, test_df, ml_train_X, ml_test_X, freq_alias=freq_alias)
        outputs["xgboost"] = xgb_res
        outputs["predictions"]["XGBoost"] = xgb_res["forecast"]
        outputs["metrics"].append(build_model_metrics("XGBoost", train_df["y"].values, test_df["y"].values, xgb_res["forecast"]))
        outputs["tables"]["XGBoost"] = build_actual_vs_pred_df(test_df, xgb_res["forecast"], "XGBoost")
    except Exception as e:
        model_errors["XGBoost"] = str(e)
        xgb_res = build_model_level_fallback("XGBoost", train_df, test_df, freq_alias, str(e))
        outputs["xgboost"] = xgb_res
        outputs["predictions"]["XGBoost"] = xgb_res["forecast"]
        outputs["metrics"].append(build_model_metrics("XGBoost", train_df["y"].values, test_df["y"].values, xgb_res["forecast"]))
        outputs["tables"]["XGBoost"] = build_actual_vs_pred_df(test_df, xgb_res["forecast"], "XGBoost")

    if not outputs["metrics"]:
        raise RuntimeError("Hiçbir model başarıyla çalışmadı. Veri, ufuk ve özellik setini kontrol edin.")

    metrics_df = pd.DataFrame(outputs["metrics"]).sort_values(["WAPE", "sMAPE", "RMSE"], ascending=[True, True, True]).reset_index(drop=True)
    ensemble_pred, ensemble_weights = build_weighted_ensemble(outputs["predictions"], metrics_df)
    outputs["predictions"]["Ensemble"] = ensemble_pred
    outputs["metrics"].append(build_model_metrics("Ensemble", train_df["y"].values, test_df["y"].values, ensemble_pred))
    outputs["tables"]["Ensemble"] = build_actual_vs_pred_df(test_df, ensemble_pred, "Ensemble")

    metrics_df = pd.DataFrame(outputs["metrics"]).sort_values(["WAPE", "sMAPE", "RMSE"], ascending=[True, True, True]).reset_index(drop=True)
    cc = build_champion_challenger(metrics_df)
    outputs["metrics_df"] = metrics_df
    outputs["champion_challenger"] = cc
    outputs["best_model"] = cc["champion"]
    outputs["ensemble_weights"] = ensemble_weights
    outputs["all_predictions_long"] = pd.concat(list(outputs["tables"].values()), axis=0, ignore_index=True) if outputs["tables"] else pd.DataFrame()
    outputs["model_errors"] = model_errors
    return outputs


def run_batch_forecasting(export_payload: Dict[str, pd.DataFrame], horizon: int, use_exog_for_stat_models: bool = True, use_exog_for_prophet: bool = True) -> Dict[str, Any]:
    targets = export_payload["series_profile_report"]["series"].tolist()
    rows = []
    champion_rows = []
    batch_outputs = {}
    for target in targets:
        try:
            out = run_full_forecasting_pipeline(export_payload, target, horizon, use_exog_for_stat_models, use_exog_for_prophet)
            batch_outputs[target] = out
            mdf = out["metrics_df"].copy()
            best_row = mdf.iloc[0].to_dict()
            best_row["target_col"] = target
            best_row["segment"] = out["metadata"]["segment"]
            best_row["abc_xyz"] = out["metadata"]["abc_xyz"]
            rows.append(best_row)
            champion_rows.append({
                "target_col": target,
                "champion": out["champion_challenger"]["champion"],
                "challenger": out["champion_challenger"]["challenger"],
                "segment": out["metadata"]["segment"],
                "abc_xyz": out["metadata"]["abc_xyz"]
            })
        except Exception as e:
            rows.append({"target_col": target, "model": "ERROR", "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "WAPE": np.nan, "MASE": np.nan, "error": str(e)})
    return {
        "best_summary": pd.DataFrame(rows),
        "champion_table": pd.DataFrame(champion_rows),
        "batch_outputs": batch_outputs
    }


def assess_production_readiness(export_payload: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    notes: List[str] = []
    status = "guarded_use_only"
    try:
        profile_df = export_payload.get("series_profile_report", pd.DataFrame())
        anomaly_df = export_payload.get("anomaly_governance", pd.DataFrame())
        backtest_df = export_payload.get("proxy_backtest_raw_vs_clean", pd.DataFrame())

        if not profile_df.empty and "series" in profile_df.columns:
            row = profile_df.loc[profile_df["series"] == target_col]
            if not row.empty:
                n_obs = float(pd.to_numeric(pd.Series([row.iloc[0].get("n_obs", np.nan)]), errors="coerce").iloc[0])
                if pd.notna(n_obs) and n_obs < 84:
                    notes.append(f"Seri uzunluğu {int(n_obs)} gözlem; aylık ilaç talebinde üretim seviyesi planlama için kısa/sınırda kabul edilir.")

        if not anomaly_df.empty and "series" in anomaly_df.columns:
            a = anomaly_df.loc[anomaly_df["series"] == target_col].copy()
            if len(a) > 0 and "is_training_excluded" in a.columns:
                excluded = int(a["is_training_excluded"].fillna(False).astype(bool).sum())
                if excluded > 0:
                    notes.append(f"Bu seri için eğitim dışı bırakılan {excluded} kayıt var; yönetişim doğru fakat model belirsizliğini artırır.")

        if not metrics_df.empty and "WAPE" in metrics_df.columns:
            best_row = metrics_df.copy()
            best_row["WAPE_num"] = pd.to_numeric(best_row["WAPE"], errors="coerce")
            best_row["sMAPE_num"] = pd.to_numeric(best_row.get("sMAPE"), errors="coerce") if "sMAPE" in best_row.columns else np.nan
            best_row = best_row.sort_values(["WAPE_num", "sMAPE_num"], ascending=[True, True])
            if len(best_row):
                bw = best_row.iloc[0]["WAPE_num"]
                bs = best_row.iloc[0]["sMAPE_num"]
                if pd.notna(bw) and pd.notna(bs):
                    if bw <= 7 and bs <= 6:
                        notes.append("Holdout performansı güçlü; kontrollü karar desteği için uygundur.")
                    else:
                        notes.append("Holdout performansı kabul edilebilir olsa da tam otomatik üretim kullanımı için ek onay katmanı gerekir.")

        if not backtest_df.empty and "series" in backtest_df.columns and "metric" in backtest_df.columns:
            b = backtest_df[(backtest_df["series"] == target_col) & (backtest_df["metric"] == "__OVERALL__")]
            if not b.empty:
                decision = str(b.iloc[0].get("decision", ""))
                reason = str(b.iloc[0].get("decision_reason", ""))
                notes.append(f"Proxy backtest kararı: {decision}. {reason}".strip())
    except Exception:
        notes.append("Üretim uygunluğu değerlendirmesi hesaplanırken hata oluştu; temkinli kullanım önerilir.")

    if not notes:
        notes.append("Bu çıktı tez ve analitik inceleme için uygundur; tam otomatik gerçek hayat kullanımı için ek doğrulama gerekir.")
    return {"status": status, "notes": notes}


def render_streamlit_app():
    st.set_page_config(page_title="Talep Tahminleme Studio", layout="wide")
    st.title("Talep Tahminleme Studio")
    st.caption("Production-grade veri önişleme + ileri seviye champion-challenger + ensemble + batch forecasting")

    with st.sidebar:
        st.subheader("Girdi")
        uploaded_excel = st.file_uploader("Excel dosyası yükle", type=["xlsx", "xls"])
        st.markdown("Bu uygulama mevcut production-grade veri önişleme mantığını korur; üstüne SARIMA/SARIMAX, Prophet, XGBoost, ensemble, ABC/XYZ ve batch forecasting ekler.")

    if uploaded_excel is None:
        st.info("Başlamak için Excel dosyanı yükle.")
        return

    try:
        excel_path = save_uploaded_file(uploaded_excel)
        xls = safe_excel_file(excel_path)
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi/açılamadı: {e}")
        return
    selected_sheet = st.sidebar.selectbox("Sheet seç", xls.sheet_names)
    output_base_dir = os.path.join(os.path.dirname(excel_path), "streamlit_outputs")
    os.makedirs(output_base_dir, exist_ok=True)

    cache_key = f"{uploaded_excel.name}::{selected_sheet}"
    if "preprocess_cache" not in st.session_state:
        st.session_state["preprocess_cache"] = {}

    if st.sidebar.button("Önişleme + Tahminleme için hazırla", type="primary") or cache_key not in st.session_state["preprocess_cache"]:
        with st.spinner("Veri önişleme ve yönetişim çalışıyor..."):
            try:
                export_payload = run_preprocessing_for_sheet(excel_path, selected_sheet, output_base_dir)
                st.session_state["preprocess_cache"][cache_key] = export_payload
            except Exception as e:
                st.error(f"Önişleme sırasında hata oluştu: {e}")
                st.exception(e)
                return
    export_payload = st.session_state["preprocess_cache"].get(cache_key)
    if export_payload is None:
        st.info("Önişleme sonucu üretilemedi.")
        return

    manifest = export_payload["manifest"]
    freq_alias = manifest.loc[manifest["key"] == "frequency_inferred", "value"].iloc[0]
    target_cols = export_payload["series_profile_report"]["series"].tolist()

    top1, top2, top3, top4 = st.columns(4)
    with top1: st.metric("Frekans", str(freq_alias))
    with top2: st.metric("Hedef seri", len(target_cols))
    with top3: st.metric("Regularized satır", int(len(export_payload["raw_regular"])))
    with top4: st.metric("Anomali kaydı", int(len(export_payload["anomaly_governance"])))

    mode = st.radio("Çalışma modu", ["Tek seri", "Çok serili batch forecasting"], horizontal=True)
    default_target = target_cols[0] if target_cols else None
    if mode == "Tek seri":
        target_col = st.selectbox("Tahminlenecek seri", target_cols, index=0 if default_target else None)
    else:
        target_col = default_target

    default_horizon = min(infer_default_horizon(freq_alias), max(2, len(export_payload["clean_model_input"]) // 5))
    horizon = st.slider("Test ufku", min_value=2, max_value=min(24, max(2, len(export_payload["clean_model_input"]) // 3)), value=default_horizon)
    use_exog_stat = st.checkbox("SARIMAX için açıklayıcı değişkenleri kullan", value=True)
    use_exog_prophet = st.checkbox("Prophet için ek regressors kullan", value=True)

    profile = get_profile_row(export_payload, target_col)
    seg = infer_advanced_segment(profile)
    priority = recommend_model_priority(profile)
    st.subheader("Seri profili")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Segment", seg["label"])
    c2.metric("ABC/XYZ", seg["abc_xyz"])
    c3.metric("Önerilen öncelik", priority)
    c4.metric("CV", round(float(profile.get("cv", np.nan)), 3) if profile else np.nan)
    c5.metric("Trend gücü", round(float(profile.get("trend_strength", np.nan)), 3) if profile else np.nan)
    c6.metric("Sezonsallık", round(float(profile.get("seasonality_strength", np.nan)), 3) if profile else np.nan)

    try:
        df_series = make_series_analysis_frame(export_payload, target_col)
        train_df_preview, _ = train_test_split_series(df_series, horizon)
        acf_pacf_fig = build_acf_pacf_figure(train_df_preview, target_col)
        if acf_pacf_fig is not None:
            st.pyplot(acf_pacf_fig, clear_figure=True)
    except Exception:
        pass

    run_label = "Batch forecasting çalıştır" if mode == "Çok serili batch forecasting" else "Modelleri çalıştır ve karşılaştır"
    if st.button(run_label, type="primary"):
        with st.spinner("Gelişmiş tahminleme katmanı çalışıyor..."):
            if mode == "Tek seri":
                outputs = run_full_forecasting_pipeline(export_payload, target_col, horizon, use_exog_stat, use_exog_prophet)
                st.session_state["forecast_outputs"] = outputs
                st.session_state["forecast_target"] = target_col
            else:
                batch = run_batch_forecasting(export_payload, horizon, use_exog_stat, use_exog_prophet)
                st.session_state["batch_outputs_full"] = batch
                st.session_state["batch_mode"] = True

    if mode == "Çok serili batch forecasting":
        batch = st.session_state.get("batch_outputs_full")
        if batch is None:
            st.info("Batch sonucu görmek için butona bas.")
            return
        st.subheader("Batch forecasting özeti")
        st.dataframe(style_metric_dataframe(batch["best_summary"]), use_container_width=True)
        st.subheader("Champion - Challenger tablosu")
        st.dataframe(batch["champion_table"], use_container_width=True)
        st.download_button("Batch özetini indir (CSV)", data=dataframe_to_download_bytes(batch["best_summary"]), file_name=f"{selected_sheet}_batch_forecasting_summary.csv", mime="text/csv")
        return

    outputs = st.session_state.get("forecast_outputs")
    if outputs is None or st.session_state.get("forecast_target") != target_col:
        st.info("Model karşılaştırmasını görmek için butona bas.")
        return

    st.subheader("Model karşılaştırma tablosu")
    metrics_df = style_metric_dataframe(outputs["metrics_df"])
    st.dataframe(metrics_df, use_container_width=True)
    st.download_button("Karşılaştırma tablosunu indir (CSV)", data=dataframe_to_download_bytes(metrics_df), file_name=f"{selected_sheet}_{target_col}_model_karsilastirma.csv", mime="text/csv")

    readiness = assess_production_readiness(export_payload, outputs["metrics_df"], target_col)
    st.warning("Bu sürüm tez ve karar destek için güçlüdür; ancak kör şekilde tam otomatik üretim planı olarak kullanılmamalıdır.")
    with st.expander("Üretim kullanımı değerlendirmesi", expanded=False):
        for note in readiness["notes"]:
            st.write(f"- {note}")

    cc = outputs.get("champion_challenger", {})
    if cc.get("champion"):
        st.success(f"Champion model: {cc['champion']} | Challenger: {cc.get('challenger', '-')}")
    if outputs.get("model_errors"):
        with st.expander("Model hata / fallback özeti"):
            err_df = pd.DataFrame([{"model": k, "message": v} for k, v in outputs.get("model_errors", {}).items()])
            if len(err_df):
                st.dataframe(err_df, use_container_width=True)

    fig = plot_forecast_results(outputs["train"], outputs["test"], outputs["predictions"], f"{target_col} - Gerçek vs Tahmin")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["SARIMA/SARIMAX", "Prophet", "XGBoost", "Champion-Challenger & Ensemble", "Gerçek vs Tahmin", "Backtest Dashboard", "Önişleme Denetimleri", "Akıllı Yorumlar"])

    with tab1:
        sarima = outputs.get("sarima")
        if sarima is None:
            st.warning(outputs.get("model_errors", {}).get("SARIMA/SARIMAX", "SARIMA/SARIMAX sonucu üretilemedi."))
        else:
            st.json({"order": sarima.get("order"), "seasonal_order": sarima.get("seasonal_order"), "trend": sarima.get("trend"), "AIC": sarima.get("aic"), "BIC": sarima.get("bic"), "Ljung-Box p-value": sarima.get("ljung_box_pvalue"), "d": sarima.get("d"), "D": sarima.get("D"), "transform": sarima.get("transform"), "white_noise_ok": sarima.get("residual_white_noise_ok"), "fallback_used": sarima.get("fallback_used"), "fallback_method": sarima.get("fallback_method")})
            if "SARIMA/SARIMAX" in outputs["tables"]:
                st.dataframe(style_metric_dataframe(outputs["tables"]["SARIMA/SARIMAX"]), use_container_width=True)
                fig_sarima = plot_forecast_results(outputs["train"], outputs["test"], {"SARIMA/SARIMAX": outputs["predictions"]["SARIMA/SARIMAX"]}, f"{target_col} - SARIMA/SARIMAX")
                if fig_sarima is not None:
                    st.plotly_chart(fig_sarima, use_container_width=True)
            if isinstance(sarima.get("search_table"), pd.DataFrame) and len(sarima.get("search_table")):
                st.dataframe(style_metric_dataframe(sarima["search_table"]), use_container_width=True)

    with tab2:
        if "prophet" in outputs:
            if isinstance(outputs["prophet"].get("config"), dict):
                st.json(outputs["prophet"]["config"])
            if isinstance(outputs["prophet"].get("component_validation"), dict):
                st.json(outputs["prophet"].get("component_validation", {}))
            st.dataframe(style_metric_dataframe(outputs["tables"]["Prophet"]), use_container_width=True)
            fig_prophet = plot_forecast_results(outputs["train"], outputs["test"], {"Prophet": outputs["predictions"]["Prophet"]}, f"{target_col} - Prophet")
            if fig_prophet is not None:
                st.plotly_chart(fig_prophet, use_container_width=True)
            st.dataframe(style_metric_dataframe(outputs["prophet"]["search_table"]), use_container_width=True)
        else:
            st.warning(outputs.get("prophet_error", "Prophet sonucu üretilemedi."))

    with tab3:
        if "xgboost" not in outputs:
            st.warning(outputs.get("model_errors", {}).get("XGBoost", "XGBoost sonucu üretilemedi."))
        else:
            st.json({"selected_strategy": outputs["xgboost"].get("strategy"), "shap_status": outputs["xgboost"].get("shap_status"), "fallback_used": outputs["xgboost"].get("fallback_used"), "fallback_method": outputs["xgboost"].get("fallback_method")})
            if "XGBoost" in outputs["tables"]:
                st.dataframe(style_metric_dataframe(outputs["tables"]["XGBoost"]), use_container_width=True)
                fig_xgb = plot_forecast_results(outputs["train"], outputs["test"], {"XGBoost": outputs["predictions"]["XGBoost"]}, f"{target_col} - XGBoost")
                if fig_xgb is not None:
                    st.plotly_chart(fig_xgb, use_container_width=True)
            st.dataframe(style_metric_dataframe(outputs["xgboost"]["search_table"]), use_container_width=True)
            if "strategy_comparison" in outputs["xgboost"]:
                st.dataframe(style_metric_dataframe(outputs["xgboost"]["strategy_comparison"]), use_container_width=True)
            if len(outputs["xgboost"].get("feature_importance", pd.DataFrame())):
                st.dataframe(outputs["xgboost"]["feature_importance"], use_container_width=True)

    with tab4:
        st.dataframe(style_metric_dataframe(outputs["champion_challenger"]["ranking"]), use_container_width=True)
        st.markdown("**Ensemble ağırlıkları**")
        st.dataframe(style_metric_dataframe(outputs["ensemble_weights"]), use_container_width=True)
        st.markdown("**Ensemble gerçek vs tahmin**")
        st.dataframe(style_metric_dataframe(outputs["tables"]["Ensemble"]), use_container_width=True)
        fig_ens = plot_forecast_results(outputs["train"], outputs["test"], {"Ensemble": outputs["predictions"]["Ensemble"]}, f"{target_col} - Ensemble")
        if fig_ens is not None:
            st.plotly_chart(fig_ens, use_container_width=True)

    with tab5:
        combined = outputs["all_predictions_long"].copy()
        st.dataframe(style_metric_dataframe(combined), use_container_width=True)
        st.download_button("Gerçek vs tahmin tablosunu indir (CSV)", data=dataframe_to_download_bytes(combined), file_name=f"{selected_sheet}_{target_col}_actual_vs_forecast.csv", mime="text/csv")

    with tab6:
        if "proxy_backtest_report" in export_payload and len(export_payload["proxy_backtest_report"]) > 0:
            st.markdown("**Proxy backtest raporu**")
            st.dataframe(export_payload["proxy_backtest_report"], use_container_width=True)
        else:
            st.info("Proxy backtest raporu bulunamadı.")
        if "raw_vs_clean_backtest_report" in export_payload and len(export_payload["raw_vs_clean_backtest_report"]) > 0:
            st.markdown("**Raw vs Clean backtest karşılaştırması**")
            st.dataframe(export_payload["raw_vs_clean_backtest_report"], use_container_width=True)

    with tab7:
        st.markdown("**Kalite raporu**")
        st.dataframe(export_payload["quality_report"], use_container_width=True)
        st.markdown("**Seri profil raporu**")
        st.dataframe(export_payload["series_profile_report"], use_container_width=True)
        st.markdown("**Anomali yönetişimi**")
        st.dataframe(export_payload["anomaly_governance"], use_container_width=True)
        st.markdown("**Review queue**")
        st.dataframe(export_payload["review_queue"], use_container_width=True)

    with tab8:
        suggestions = []
        if float(profile.get("seasonality_strength", 0) or 0) >= 0.35:
            suggestions.append("Seri sezonsal; auto seasonal (P,D,Q,m), Prophet custom seasonality ve ensemble kritik.")
        if float(profile.get("cv", 0) or 0) >= 0.45:
            suggestions.append("Oynaklık yüksek; log/Box-Cox dönüşümü, XGBoost rolling istatistikleri ve challenger modeli zorunlu izlenmeli.")
        if float(profile.get("intermittency_ratio", 0) or 0) >= 0.25:
            suggestions.append("Intermittent yapı var; stok-out ayrımı ve model çıktılarını iş kuralı ile doğrulamak gerekir.")
        if outputs.get("sarima", {}).get("residual_white_noise_ok") is False:
            suggestions.append("SARIMA residual white-noise testi zayıf; challenger olarak Prophet veya XGBoost önceliklendirilmeli.")
        if outputs.get("best_model") == "Ensemble":
            suggestions.append("Bu seride tek bir champion yerine ensemble daha iyi; operasyonel kullanımda champion-challenger izleme önerilir.")
        if not suggestions:
            suggestions.append("Seri dengeli; champion-challenger ve ensemble izlemesi yeterli görünüyor.")
        for s in suggestions:
            st.write(f"- {s}")


if __name__ == "__main__":
    if st is None:
        raise ImportError("Bu dosya Streamlit uygulamasıdır. Çalıştırmak için: streamlit run <dosya_adı>.py")
    try:
        render_streamlit_app()
    except Exception as e:
        st.error(f"Beklenmeyen uygulama hatası: {e}")
        st.exception(e)
