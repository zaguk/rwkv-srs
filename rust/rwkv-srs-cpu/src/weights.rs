use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use safetensors::{Dtype as SafeDType, SafeTensors};
use thiserror::Error;

pub(crate) const D_MODEL: usize = 128;
pub(crate) const N_HEADS: usize = 4;
pub(crate) const HEAD_SIZE: usize = D_MODEL / N_HEADS;
pub(crate) const CARD_FEATURES_DIM: usize = 92;
pub(crate) const FEATURES_FC_DIM: usize = 4 * D_MODEL;
pub(crate) const HEAD_DIM: usize = 4 * D_MODEL;
pub(crate) const NUM_CURVES: usize = 128;
pub(crate) const NUM_POINTS: usize = 128;
const EXPECTED_PARAMETER_COUNT: usize = 2_762_884;

pub(crate) const MODULE_CONFIGS: [ModuleConfig; 5] = [
    ModuleConfig {
        n_layers: 3,
        channel_mixer_factor_num: 3,
        channel_mixer_factor_den: 2,
    },
    ModuleConfig {
        n_layers: 4,
        channel_mixer_factor_num: 2,
        channel_mixer_factor_den: 1,
    },
    ModuleConfig {
        n_layers: 2,
        channel_mixer_factor_num: 3,
        channel_mixer_factor_den: 2,
    },
    ModuleConfig {
        n_layers: 3,
        channel_mixer_factor_num: 2,
        channel_mixer_factor_den: 1,
    },
    ModuleConfig {
        n_layers: 4,
        channel_mixer_factor_num: 2,
        channel_mixer_factor_den: 1,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<usize>,
    pub actual: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeMismatch {
    pub key: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightValidationReport {
    pub source: PathBuf,
    pub expected_key_count: usize,
    pub loaded_key_count: usize,
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
    pub dtype_mismatches: Vec<DTypeMismatch>,
}

impl WeightValidationReport {
    pub fn ok(&self) -> bool {
        self.missing_keys.is_empty()
            && self.unexpected_keys.is_empty()
            && self.shape_mismatches.is_empty()
            && self.dtype_mismatches.is_empty()
    }
}

#[derive(Debug)]
pub struct LoadedWeights {
    pub tensors: BTreeMap<String, Tensor>,
    pub validation: WeightValidationReport,
}

impl LoadedWeights {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Debug, Error)]
pub enum WeightLoadError {
    #[error("failed to read safetensors checkpoint {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse safetensors checkpoint {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    #[error("{0}")]
    Validation(Box<WeightValidationError>),
    #[error("failed to load safetensors checkpoint {path} through Candle: {source}")]
    CandleLoad {
        path: PathBuf,
        #[source]
        source: candle_core::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightValidationError {
    pub report: WeightValidationReport,
}

impl std::fmt::Display for WeightValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let report = &self.report;
        writeln!(
            f,
            "weight validation failed for {}: expected {} keys, loaded {} keys",
            report.source.display(),
            report.expected_key_count,
            report.loaded_key_count
        )?;
        if !report.missing_keys.is_empty() {
            writeln!(f, "missing keys: {}", preview(&report.missing_keys, 12))?;
        }
        if !report.unexpected_keys.is_empty() {
            writeln!(
                f,
                "unexpected keys: {}",
                preview(&report.unexpected_keys, 12)
            )?;
        }
        if !report.shape_mismatches.is_empty() {
            writeln!(
                f,
                "shape mismatches: {}",
                preview_shape_mismatches(&report.shape_mismatches, 6)
            )?;
        }
        if !report.dtype_mismatches.is_empty() {
            writeln!(
                f,
                "dtype mismatches: {}",
                preview_dtype_mismatches(&report.dtype_mismatches, 6)
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for WeightValidationError {}

pub fn expected_weight_specs() -> BTreeMap<String, TensorSpec> {
    let mut specs = BTreeMap::new();

    add_linear(
        &mut specs,
        "features2card.0",
        FEATURES_FC_DIM,
        CARD_FEATURES_DIM,
        true,
    );
    add_layer_norm(&mut specs, "features2card.2", FEATURES_FC_DIM);
    add_linear(
        &mut specs,
        "features2card.3",
        D_MODEL,
        FEATURES_FC_DIM,
        true,
    );

    for (module_index, config) in MODULE_CONFIGS.iter().enumerate() {
        for layer_index in 0..config.n_layers {
            add_rwkv_block(&mut specs, module_index, layer_index, *config);
        }
    }

    add_layer_norm(&mut specs, "prehead_norm", D_MODEL);
    add_linear(&mut specs, "head_ahead_logits.0", HEAD_DIM, D_MODEL, true);
    add_linear(&mut specs, "head_w.0", D_MODEL, D_MODEL, true);
    add_layer_norm(&mut specs, "head_w.2", D_MODEL);
    add_linear(&mut specs, "head_w.4", HEAD_DIM, D_MODEL, true);
    add_linear(&mut specs, "head_p.0", HEAD_DIM, D_MODEL, true);
    add_linear(&mut specs, "ahead_linear", NUM_POINTS, HEAD_DIM, true);
    add_linear(&mut specs, "w_linear", NUM_CURVES, HEAD_DIM, true);
    add_linear(&mut specs, "p_linear", 4, HEAD_DIM, true);

    debug_assert_eq!(specs.len(), 504);
    debug_assert_eq!(parameter_count_from_specs(&specs), EXPECTED_PARAMETER_COUNT);
    specs
}

pub fn expected_parameter_count() -> usize {
    EXPECTED_PARAMETER_COUNT
}

pub fn parameter_count_from_specs(specs: &BTreeMap<String, TensorSpec>) -> usize {
    specs
        .values()
        .map(|spec| spec.shape.iter().product::<usize>())
        .sum()
}

pub fn validate_safetensors_file<P: AsRef<Path>>(
    path: P,
) -> Result<WeightValidationReport, WeightLoadError> {
    let path = path.as_ref();
    let data = fs::read(path).map_err(|source| WeightLoadError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let safetensors = SafeTensors::deserialize(&data).map_err(|source| WeightLoadError::Parse {
        path: path.to_path_buf(),
        source,
    })?;

    let loaded = safetensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                LoadedTensorMeta {
                    shape: view.shape().to_vec(),
                    dtype: safe_dtype_name(view.dtype()),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    Ok(validate_loaded_meta(
        path,
        &expected_weight_specs(),
        &loaded,
    ))
}

pub fn load_rwkv_srs_weights<P: AsRef<Path>>(path: P) -> Result<LoadedWeights, WeightLoadError> {
    let path = path.as_ref();
    let validation = validate_safetensors_file(path)?;
    if !validation.ok() {
        return Err(WeightLoadError::Validation(Box::new(
            WeightValidationError { report: validation },
        )));
    }

    let loaded = candle_core::safetensors::load(path, &Device::Cpu).map_err(|source| {
        WeightLoadError::CandleLoad {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let tensors = loaded.into_iter().collect::<BTreeMap<_, _>>();
    let loaded_validation = validate_loaded_tensors(path, &expected_weight_specs(), &tensors);
    if !loaded_validation.ok() {
        return Err(WeightLoadError::Validation(Box::new(
            WeightValidationError {
                report: loaded_validation,
            },
        )));
    }

    Ok(LoadedWeights {
        tensors,
        validation,
    })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModuleConfig {
    pub(crate) n_layers: usize,
    pub(crate) channel_mixer_factor_num: usize,
    pub(crate) channel_mixer_factor_den: usize,
}

impl ModuleConfig {
    pub(crate) fn channel_dim(self) -> usize {
        D_MODEL * self.channel_mixer_factor_num / self.channel_mixer_factor_den
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadedTensorMeta {
    shape: Vec<usize>,
    dtype: String,
}

fn add_rwkv_block(
    specs: &mut BTreeMap<String, TensorSpec>,
    module_index: usize,
    layer_index: usize,
    config: ModuleConfig,
) {
    let prefix = format!("rwkv_modules.{module_index}.blocks.{layer_index}");
    let time = format!("{prefix}.time_mixer");
    add_layer_norm(specs, &format!("{time}.layer_norm"), D_MODEL);
    add_param(specs, format!("{time}.rkvdag_lerp"), vec![8, 1, 1, D_MODEL]);
    add_param(
        specs,
        format!("{time}.bonus"),
        vec![1, 1, N_HEADS, HEAD_SIZE],
    );
    add_linear(specs, &format!("{time}.W_r"), D_MODEL, D_MODEL, false);
    add_linear(specs, &format!("{time}.W_k"), D_MODEL, D_MODEL, false);
    add_linear(specs, &format!("{time}.W_v"), D_MODEL, D_MODEL, false);
    add_linear(specs, &format!("{time}.W_o"), D_MODEL, D_MODEL, false);
    add_linear(
        specs,
        &format!("{time}.k_scale_linear"),
        N_HEADS,
        D_MODEL,
        true,
    );
    add_linear(
        specs,
        &format!("{time}.v_scale_linear"),
        N_HEADS,
        D_MODEL,
        true,
    );
    add_lora_simple(specs, &format!("{time}.v_lora_simple"), 8);
    add_lora_simple(specs, &format!("{time}.a_lora_simple"), 16);
    add_lora_simple(specs, &format!("{time}.d_lora_mlp"), 16);
    add_linear(specs, &format!("{time}.lora_A_g"), 16, D_MODEL, false);
    add_linear(specs, &format!("{time}.lora_B_g"), D_MODEL, 16, false);
    add_layer_norm(specs, &format!("{time}.out_group_norm"), D_MODEL);

    let channel = format!("{prefix}.channel_mixer");
    add_layer_norm(specs, &format!("{channel}.layer_norm"), D_MODEL);
    add_param(specs, format!("{channel}.lerp_k"), vec![1, 1, D_MODEL]);
    add_linear(
        specs,
        &format!("{channel}.W_k"),
        config.channel_dim(),
        D_MODEL,
        false,
    );
    add_linear(
        specs,
        &format!("{channel}.W_v"),
        D_MODEL,
        config.channel_dim(),
        false,
    );
}

fn add_lora_simple(specs: &mut BTreeMap<String, TensorSpec>, prefix: &str, d_lora: usize) {
    add_linear(specs, &format!("{prefix}.A"), d_lora, D_MODEL, false);
    add_linear(
        specs,
        &format!("{prefix}.B_and_lamb"),
        D_MODEL,
        d_lora,
        true,
    );
}

fn add_layer_norm(specs: &mut BTreeMap<String, TensorSpec>, prefix: &str, dim: usize) {
    add_param(specs, format!("{prefix}.weight"), vec![dim]);
    add_param(specs, format!("{prefix}.bias"), vec![dim]);
}

fn add_linear(
    specs: &mut BTreeMap<String, TensorSpec>,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
    bias: bool,
) {
    add_param(specs, format!("{prefix}.weight"), vec![out_dim, in_dim]);
    if bias {
        add_param(specs, format!("{prefix}.bias"), vec![out_dim]);
    }
}

fn add_param(specs: &mut BTreeMap<String, TensorSpec>, name: String, shape: Vec<usize>) {
    let previous = specs.insert(
        name.clone(),
        TensorSpec {
            name,
            shape,
            dtype: DType::F32,
        },
    );
    debug_assert!(previous.is_none());
}

fn validate_loaded_tensors(
    source: &Path,
    expected: &BTreeMap<String, TensorSpec>,
    loaded: &BTreeMap<String, Tensor>,
) -> WeightValidationReport {
    let loaded_meta = loaded
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                LoadedTensorMeta {
                    shape: tensor.dims().to_vec(),
                    dtype: candle_dtype_name(tensor.dtype()).to_string(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    validate_loaded_meta(source, expected, &loaded_meta)
}

fn validate_loaded_meta(
    source: &Path,
    expected: &BTreeMap<String, TensorSpec>,
    loaded: &BTreeMap<String, LoadedTensorMeta>,
) -> WeightValidationReport {
    let expected_keys = expected.keys().cloned().collect::<BTreeSet<_>>();
    let loaded_keys = loaded.keys().cloned().collect::<BTreeSet<_>>();

    let mut shape_mismatches = Vec::new();
    let mut dtype_mismatches = Vec::new();
    for key in expected_keys.intersection(&loaded_keys) {
        let expected_spec = &expected[key];
        let actual = &loaded[key];
        if actual.shape != expected_spec.shape {
            shape_mismatches.push(ShapeMismatch {
                key: key.clone(),
                expected: expected_spec.shape.clone(),
                actual: actual.shape.clone(),
            });
        }
        let expected_dtype = candle_dtype_name(expected_spec.dtype);
        if actual.dtype != expected_dtype {
            dtype_mismatches.push(DTypeMismatch {
                key: key.clone(),
                expected: expected_dtype.to_string(),
                actual: actual.dtype.clone(),
            });
        }
    }

    WeightValidationReport {
        source: source.to_path_buf(),
        expected_key_count: expected.len(),
        loaded_key_count: loaded.len(),
        missing_keys: expected_keys.difference(&loaded_keys).cloned().collect(),
        unexpected_keys: loaded_keys.difference(&expected_keys).cloned().collect(),
        shape_mismatches,
        dtype_mismatches,
    }
}

fn safe_dtype_name(dtype: SafeDType) -> String {
    match dtype {
        SafeDType::U8 => "U8".to_string(),
        SafeDType::U32 => "U32".to_string(),
        SafeDType::I16 => "I16".to_string(),
        SafeDType::I32 => "I32".to_string(),
        SafeDType::I64 => "I64".to_string(),
        SafeDType::BF16 => "BF16".to_string(),
        SafeDType::F16 => "F16".to_string(),
        SafeDType::F32 => "F32".to_string(),
        SafeDType::F64 => "F64".to_string(),
        other => format!("{other:?}"),
    }
}

fn candle_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::F8E4M3 => "F8E4M3",
        DType::F6E2M3 => "F6E2M3",
        DType::F6E3M2 => "F6E3M2",
        DType::F4 => "F4",
        DType::F8E8M0 => "F8E8M0",
    }
}

fn preview(values: &[String], limit: usize) -> String {
    let mut shown = values.iter().take(limit).cloned().collect::<Vec<_>>();
    if values.len() > limit {
        shown.push(format!("... and {} more", values.len() - limit));
    }
    format!("{shown:?}")
}

fn preview_shape_mismatches(values: &[ShapeMismatch], limit: usize) -> String {
    let shown = values
        .iter()
        .take(limit)
        .map(|mismatch| {
            format!(
                "{} expected {:?} actual {:?}",
                mismatch.key, mismatch.expected, mismatch.actual
            )
        })
        .collect::<Vec<_>>();
    append_preview_suffix(shown, values.len(), limit)
}

fn preview_dtype_mismatches(values: &[DTypeMismatch], limit: usize) -> String {
    let shown = values
        .iter()
        .take(limit)
        .map(|mismatch| {
            format!(
                "{} expected {:?} actual {:?}",
                mismatch.key, mismatch.expected, mismatch.actual
            )
        })
        .collect::<Vec<_>>();
    append_preview_suffix(shown, values.len(), limit)
}

fn append_preview_suffix(mut shown: Vec<String>, len: usize, limit: usize) -> String {
    if len > limit {
        shown.push(format!("... and {} more", len - limit));
    }
    format!("{shown:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn expected_schema_matches_python_checkpoint_contract() {
        let specs = expected_weight_specs();
        assert_eq!(specs.len(), 504);
        assert_eq!(
            parameter_count_from_specs(&specs),
            expected_parameter_count()
        );

        assert_eq!(
            specs["features2card.0.weight"].shape,
            vec![FEATURES_FC_DIM, CARD_FEATURES_DIM]
        );
        assert_eq!(
            specs["rwkv_modules.0.blocks.0.time_mixer.rkvdag_lerp"].shape,
            vec![8, 1, 1, D_MODEL]
        );
        assert_eq!(
            specs["rwkv_modules.4.blocks.3.channel_mixer.W_k.weight"].shape,
            vec![256, D_MODEL]
        );
        assert_eq!(specs["p_linear.bias"].shape, vec![4]);
        assert!(specs
            .keys()
            .all(|key| !key.ends_with("lora_B_g.bias") && !key.ends_with("W_k.bias")));
    }

    #[test]
    fn loader_accepts_a_complete_schema_conformant_safetensors_file() {
        let path = TempSafetensors::new("complete");
        write_zero_checkpoint(&path.path);

        let weights = load_rwkv_srs_weights(&path.path).unwrap();

        assert!(weights.validation.ok());
        assert_eq!(weights.tensor_count(), 504);
        assert_eq!(
            weights.tensors["features2card.0.weight"].dims(),
            &[FEATURES_FC_DIM, CARD_FEATURES_DIM]
        );
        assert_eq!(
            weights.tensors["features2card.0.weight"].dtype(),
            DType::F32
        );
    }

    #[test]
    fn committed_pretrained_checkpoints_load() {
        let repo = repo_root();
        let paths = [
            repo.join("tests/fixtures/models/RWKV_trained_on_101_4999.safetensors"),
            repo.join("tests/fixtures/models/RWKV_trained_on_5000_10000.safetensors"),
        ];

        for path in &paths {
            assert!(
                path.exists(),
                "committed safetensors checkpoint is absent: {}",
                path.display()
            );
        }

        for path in paths {
            let weights = load_rwkv_srs_weights(&path).unwrap();
            assert!(weights.validation.ok(), "{}", path.display());
            assert_eq!(weights.tensor_count(), 504, "{}", path.display());
            assert_eq!(
                weights.tensors["rwkv_modules.4.blocks.3.time_mixer.bonus"].dims(),
                &[1, 1, N_HEADS, HEAD_SIZE],
                "{}",
                path.display()
            );
        }
    }

    #[test]
    fn validation_reports_missing_unexpected_shape_and_dtype_errors() {
        let path = TempSafetensors::new("bad");
        let mut tensors = HashMap::new();
        tensors.insert(
            "features2card.0.weight".to_string(),
            Tensor::zeros((1usize, CARD_FEATURES_DIM), DType::F64, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "unexpected.weight".to_string(),
            Tensor::zeros((2usize, 2usize), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path.path).unwrap();

        let report = validate_safetensors_file(&path.path).unwrap();

        assert!(!report.ok());
        assert_eq!(report.expected_key_count, 504);
        assert_eq!(report.loaded_key_count, 2);
        assert!(report
            .missing_keys
            .contains(&"features2card.0.bias".to_string()));
        assert_eq!(report.unexpected_keys, vec!["unexpected.weight"]);
        assert_eq!(
            report.shape_mismatches,
            vec![ShapeMismatch {
                key: "features2card.0.weight".to_string(),
                expected: vec![FEATURES_FC_DIM, CARD_FEATURES_DIM],
                actual: vec![1, CARD_FEATURES_DIM],
            }]
        );
        assert_eq!(
            report.dtype_mismatches,
            vec![DTypeMismatch {
                key: "features2card.0.weight".to_string(),
                expected: "F32".to_string(),
                actual: "F64".to_string(),
            }]
        );
    }

    #[test]
    fn loader_fails_fast_with_validation_error_before_returning_tensors() {
        let path = TempSafetensors::new("invalid");
        let tensors = HashMap::from([(
            "features2card.0.weight".to_string(),
            Tensor::zeros((1usize, CARD_FEATURES_DIM), DType::F32, &Device::Cpu).unwrap(),
        )]);
        candle_core::safetensors::save(&tensors, &path.path).unwrap();

        let err = load_rwkv_srs_weights(&path.path).unwrap_err();

        match err {
            WeightLoadError::Validation(err) => {
                assert!(!err.report.ok());
                assert_eq!(err.report.loaded_key_count, 1);
                assert!(!err.report.missing_keys.is_empty());
                assert!(!err.to_string().is_empty());
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    fn write_zero_checkpoint(path: &Path) {
        let tensors = expected_weight_specs()
            .into_iter()
            .map(|(name, spec)| {
                let tensor = Tensor::zeros(spec.shape.as_slice(), DType::F32, &Device::Cpu)
                    .unwrap_or_else(|err| panic!("failed to create tensor {name}: {err}"));
                (name, tensor)
            })
            .collect::<HashMap<_, _>>();
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives under rust/rwkv-srs-cpu")
            .to_path_buf()
    }

    struct TempSafetensors {
        path: PathBuf,
    }

    impl TempSafetensors {
        fn new(name: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock is before unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "rwkv_srs_rs_{name}_{}_{}.safetensors",
                std::process::id(),
                nanos
            ));
            Self { path }
        }
    }

    impl Drop for TempSafetensors {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
