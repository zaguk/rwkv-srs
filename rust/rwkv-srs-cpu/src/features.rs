use pyo3::prelude::*;

pub const CARD_FEATURE_COLUMNS: [&str; 24] = [
    "scaled_elapsed_days",
    "scaled_elapsed_days_cumulative",
    "scaled_elapsed_seconds",
    "elapsed_seconds_sin",
    "elapsed_seconds_cos",
    "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin",
    "elapsed_seconds_cumulative_cos",
    "scaled_duration",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "note_id_is_nan",
    "deck_id_is_nan",
    "preset_id_is_nan",
    "day_offset_diff",
    "day_of_week",
    "diff_new_cards",
    "diff_reviews",
    "cum_new_cards_today",
    "cum_reviews_today",
    "scaled_state",
    "is_query",
];

pub const ID_PLACEHOLDER: i64 = 314_159_265_358_979_323;
pub const DAY_OFFSET_ENCODE_PERIODS: [f64; 7] = [3.0, 7.0, 30.0, 100.0, 365.0, 3650.0, 36500.0];

const SECONDS_PER_DAY: f64 = 86_400.0;

const ELAPSED_DAYS_MEAN: f64 = 1.51;
const ELAPSED_DAYS_STD: f64 = 1.62;
const ELAPSED_DAYS_CUMULATIVE_MEAN: f64 = 2.14;
const ELAPSED_DAYS_CUMULATIVE_STD: f64 = 2.25;
const ELAPSED_SECONDS_MEAN: f64 = 9.96;
const ELAPSED_SECONDS_STD: f64 = 5.21;
const ELAPSED_SECONDS_CUMULATIVE_MEAN: f64 = 10.86;
const ELAPSED_SECONDS_CUMULATIVE_STD: f64 = 5.8;
const DURATION_MEAN: f64 = 8.9;
const DURATION_STD: f64 = 1.07;
const DIFF_NEW_CARDS_MEAN: f64 = 2.945;
const DIFF_NEW_CARDS_STD: f64 = 2.011;
const DIFF_REVIEWS_MEAN: f64 = 4.64;
const DIFF_REVIEWS_STD: f64 = 2.59;
const CUM_NEW_CARDS_TODAY_MEAN: f64 = 2.55;
const CUM_NEW_CARDS_TODAY_STD: f64 = 1.41;
const CUM_REVIEWS_TODAY_MEAN: f64 = 4.59;
const CUM_REVIEWS_TODAY_STD: f64 = 1.30;

pub fn scale_elapsed_days(x: f64) -> f64 {
    (elapsed_log_or_zero(x) - ELAPSED_DAYS_MEAN) / ELAPSED_DAYS_STD
}

pub fn scale_elapsed_days_cumulative(x: f64) -> f64 {
    (elapsed_log_or_zero(x) - ELAPSED_DAYS_CUMULATIVE_MEAN) / ELAPSED_DAYS_CUMULATIVE_STD
}

pub fn scale_elapsed_seconds(x: f64) -> f64 {
    (elapsed_log_or_zero(x) - ELAPSED_SECONDS_MEAN) / ELAPSED_SECONDS_STD
}

pub fn scale_elapsed_seconds_cumulative(x: f64) -> f64 {
    (elapsed_log_or_zero(x) - ELAPSED_SECONDS_CUMULATIVE_MEAN) / ELAPSED_SECONDS_CUMULATIVE_STD
}

pub fn scale_duration(x: f64) -> f64 {
    ((10.0 + x).ln() - DURATION_MEAN) / DURATION_STD
}

pub fn scale_diff_new_cards(x: f64) -> f64 {
    ((3.0 + x).ln() - DIFF_NEW_CARDS_MEAN) / DIFF_NEW_CARDS_STD
}

pub fn scale_diff_reviews(x: f64) -> f64 {
    ((3.0 + x).ln() - DIFF_REVIEWS_MEAN) / DIFF_REVIEWS_STD
}

pub fn scale_cum_new_cards_today(x: f64) -> f64 {
    ((3.0 + x).ln() - CUM_NEW_CARDS_TODAY_MEAN) / CUM_NEW_CARDS_TODAY_STD
}

pub fn scale_cum_reviews_today(x: f64) -> f64 {
    ((3.0 + x).ln() - CUM_REVIEWS_TODAY_MEAN) / CUM_REVIEWS_TODAY_STD
}

pub fn scale_state(x: f64) -> f64 {
    x - 2.0
}

pub fn scale_day_offset_diff(x: f64) -> f64 {
    (std::f64::consts::E + x).ln().ln()
}

pub fn elapsed_seconds_sin(x: f64) -> f64 {
    (py_mod(x, SECONDS_PER_DAY) * 2.0 * std::f64::consts::PI / SECONDS_PER_DAY).sin()
}

pub fn elapsed_seconds_cos(x: f64) -> f64 {
    (py_mod(x, SECONDS_PER_DAY) * 2.0 * std::f64::consts::PI / SECONDS_PER_DAY).cos()
}

pub fn day_offset_encoding(day_offset: f64, day_offset_first: f64) -> Vec<f32> {
    let mut encoded = Vec::with_capacity(DAY_OFFSET_ENCODE_PERIODS.len() * 4);
    for period in DAY_OFFSET_ENCODE_PERIODS {
        let f = 2.0 * std::f64::consts::PI / period;
        let day_offset_mod = py_mod(day_offset, period);
        encoded.push((f * day_offset_mod).sin() as f32);
        encoded.push((f * day_offset_mod).cos() as f32);

        let day_offset_first_mod = py_mod(day_offset_first, period);
        encoded.push((f * day_offset_first_mod).sin() as f32);
        encoded.push((f * day_offset_first_mod).cos() as f32);
    }
    encoded
}

fn elapsed_log_or_zero(x: f64) -> f64 {
    if x == -1.0 {
        0.0
    } else {
        (1.0 + 1e-5 + x).ln()
    }
}

fn py_mod(value: f64, modulus: f64) -> f64 {
    ((value % modulus) + modulus) % modulus
}

#[pyfunction(name = "scale_elapsed_days")]
pub fn scale_elapsed_days_py(x: f64) -> f64 {
    scale_elapsed_days(x)
}

#[pyfunction(name = "scale_elapsed_days_cumulative")]
pub fn scale_elapsed_days_cumulative_py(x: f64) -> f64 {
    scale_elapsed_days_cumulative(x)
}

#[pyfunction(name = "scale_elapsed_seconds")]
pub fn scale_elapsed_seconds_py(x: f64) -> f64 {
    scale_elapsed_seconds(x)
}

#[pyfunction(name = "scale_elapsed_seconds_cumulative")]
pub fn scale_elapsed_seconds_cumulative_py(x: f64) -> f64 {
    scale_elapsed_seconds_cumulative(x)
}

#[pyfunction(name = "scale_duration")]
pub fn scale_duration_py(x: f64) -> f64 {
    scale_duration(x)
}

#[pyfunction(name = "scale_diff_new_cards")]
pub fn scale_diff_new_cards_py(x: f64) -> f64 {
    scale_diff_new_cards(x)
}

#[pyfunction(name = "scale_diff_reviews")]
pub fn scale_diff_reviews_py(x: f64) -> f64 {
    scale_diff_reviews(x)
}

#[pyfunction(name = "scale_cum_new_cards_today")]
pub fn scale_cum_new_cards_today_py(x: f64) -> f64 {
    scale_cum_new_cards_today(x)
}

#[pyfunction(name = "scale_cum_reviews_today")]
pub fn scale_cum_reviews_today_py(x: f64) -> f64 {
    scale_cum_reviews_today(x)
}

#[pyfunction(name = "scale_state")]
pub fn scale_state_py(x: f64) -> f64 {
    scale_state(x)
}

#[pyfunction(name = "scale_day_offset_diff")]
pub fn scale_day_offset_diff_py(x: f64) -> f64 {
    scale_day_offset_diff(x)
}

#[pyfunction(name = "day_offset_encoding")]
pub fn day_offset_encoding_py(day_offset: f64, day_offset_first: f64) -> Vec<f32> {
    day_offset_encoding(day_offset, day_offset_first)
}

#[pyfunction(name = "card_feature_columns")]
pub fn card_feature_columns_py() -> Vec<&'static str> {
    CARD_FEATURE_COLUMNS.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalers_match_python_formulas() {
        assert!((scale_elapsed_days(-1.0) - (-0.9320987654320988)).abs() < 1e-12);
        assert!((scale_duration(2.5) - (-5.957262949244621)).abs() < 1e-12);
        assert_eq!(scale_state(4.0), 2.0);
        assert_eq!(CARD_FEATURE_COLUMNS.len(), 24);
    }

    #[test]
    fn day_encoding_has_expected_layout() {
        let encoded = day_offset_encoding(1.0, 2.0);
        assert_eq!(encoded.len(), 28);
        assert!((encoded[0] - 0.8660254).abs() < 1e-6);
        assert!((encoded[1] - (-0.5)).abs() < 1e-6);
        assert!((encoded[2] - (-0.8660254)).abs() < 1e-6);
        assert!((encoded[3] - (-0.5)).abs() < 1e-6);
    }
}
