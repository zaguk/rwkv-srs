use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use candle_core::Tensor;
use thiserror::Error;

use crate::weights::{
    load_rwkv_srs_weights, LoadedWeights, ModuleConfig, WeightLoadError, WeightValidationError,
    HEAD_SIZE, MODULE_CONFIGS, N_HEADS,
};

#[derive(Debug)]
pub struct SrsRwkvRnnWeights {
    pub features2card: Features2CardWeights,
    pub rwkv_modules: Vec<Rwkv7RnnWeights>,
    pub prehead_norm: LayerNormWeights,
    pub head_ahead_logits: LinearWeights,
    pub head_w: WHeadWeights,
    pub head_p: LinearWeights,
    pub ahead_linear: LinearWeights,
    pub w_linear: LinearWeights,
    pub p_linear: LinearWeights,
}

impl SrsRwkvRnnWeights {
    pub fn from_loaded_weights(weights: LoadedWeights) -> Result<Self, ModelWeightsError> {
        let LoadedWeights {
            tensors,
            validation,
        } = weights;
        if !validation.ok() {
            return Err(ModelWeightsError::Validation(Box::new(
                WeightValidationError { report: validation },
            )));
        }
        Self::from_tensor_map(tensors)
    }

    pub(crate) fn from_tensor_map(
        tensors: BTreeMap<String, Tensor>,
    ) -> Result<Self, ModelWeightsError> {
        let mut tensors = TensorConsumer::new(tensors);

        let features2card = Features2CardWeights::consume(&mut tensors)?;
        let mut rwkv_modules = Vec::with_capacity(MODULE_CONFIGS.len());
        for (module_index, config) in MODULE_CONFIGS.iter().copied().enumerate() {
            rwkv_modules.push(Rwkv7RnnWeights::consume(
                &mut tensors,
                module_index,
                config,
            )?);
        }

        let model = Self {
            features2card,
            rwkv_modules,
            prehead_norm: tensors.layer_norm("prehead_norm")?,
            head_ahead_logits: tensors.linear("head_ahead_logits.0", true)?,
            head_w: WHeadWeights::consume(&mut tensors)?,
            head_p: tensors.linear("head_p.0", true)?,
            ahead_linear: tensors.linear("ahead_linear", true)?,
            w_linear: tensors.linear("w_linear", true)?,
            p_linear: tensors.linear("p_linear", true)?,
        };

        tensors.finish()?;
        Ok(model)
    }

    pub fn tensor_count(&self) -> usize {
        self.features2card.tensor_count()
            + self
                .rwkv_modules
                .iter()
                .map(Rwkv7RnnWeights::tensor_count)
                .sum::<usize>()
            + self.prehead_norm.tensor_count()
            + self.head_ahead_logits.tensor_count()
            + self.head_w.tensor_count()
            + self.head_p.tensor_count()
            + self.ahead_linear.tensor_count()
            + self.w_linear.tensor_count()
            + self.p_linear.tensor_count()
    }

    pub fn block_count(&self) -> usize {
        self.rwkv_modules
            .iter()
            .map(|module| module.blocks.len())
            .sum()
    }
}

impl TryFrom<LoadedWeights> for SrsRwkvRnnWeights {
    type Error = ModelWeightsError;

    fn try_from(weights: LoadedWeights) -> Result<Self, Self::Error> {
        Self::from_loaded_weights(weights)
    }
}

#[derive(Debug)]
pub struct Features2CardWeights {
    pub input_linear: LinearWeights,
    pub norm: LayerNormWeights,
    pub output_linear: LinearWeights,
}

impl Features2CardWeights {
    fn consume(tensors: &mut TensorConsumer) -> Result<Self, ModelWeightsError> {
        Ok(Self {
            input_linear: tensors.linear("features2card.0", true)?,
            norm: tensors.layer_norm("features2card.2")?,
            output_linear: tensors.linear("features2card.3", true)?,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.input_linear.tensor_count()
            + self.norm.tensor_count()
            + self.output_linear.tensor_count()
    }
}

#[derive(Debug)]
pub struct Rwkv7RnnWeights {
    pub module_index: usize,
    pub blocks: Vec<Rwkv7RnnLayerWeights>,
}

impl Rwkv7RnnWeights {
    fn consume(
        tensors: &mut TensorConsumer,
        module_index: usize,
        config: ModuleConfig,
    ) -> Result<Self, ModelWeightsError> {
        let mut blocks = Vec::with_capacity(config.n_layers);
        for layer_index in 0..config.n_layers {
            blocks.push(Rwkv7RnnLayerWeights::consume(
                tensors,
                module_index,
                layer_index,
                config,
            )?);
        }

        Ok(Self {
            module_index,
            blocks,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.blocks
            .iter()
            .map(Rwkv7RnnLayerWeights::tensor_count)
            .sum()
    }
}

#[derive(Debug)]
pub struct Rwkv7RnnLayerWeights {
    pub layer_id: usize,
    pub time_mixer: Rwkv7RnnTimeMixerWeights,
    pub channel_mixer: Rwkv7RnnChannelMixerWeights,
}

impl Rwkv7RnnLayerWeights {
    fn consume(
        tensors: &mut TensorConsumer,
        module_index: usize,
        layer_index: usize,
        config: ModuleConfig,
    ) -> Result<Self, ModelWeightsError> {
        Ok(Self {
            layer_id: layer_index,
            time_mixer: Rwkv7RnnTimeMixerWeights::consume(tensors, module_index, layer_index)?,
            channel_mixer: Rwkv7RnnChannelMixerWeights::consume(
                tensors,
                module_index,
                layer_index,
                config,
            )?,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.time_mixer.tensor_count() + self.channel_mixer.tensor_count()
    }
}

#[derive(Debug)]
pub struct Rwkv7RnnTimeMixerWeights {
    pub layer_id: usize,
    pub n_heads: usize,
    pub head_size: usize,
    pub layer_norm: LayerNormWeights,
    pub rkvdag_lerp: Tensor,
    pub bonus: Tensor,
    pub w_r: LinearWeights,
    pub w_k: LinearWeights,
    pub w_v: LinearWeights,
    pub w_o: LinearWeights,
    pub k_scale_linear: LinearWeights,
    pub v_scale_linear: LinearWeights,
    pub v_lora_simple: LoraSimpleWeights,
    pub a_lora_simple: LoraSimpleWeights,
    pub d_lora_mlp: LoraSimpleWeights,
    pub gate_lora: GateLoraWeights,
    pub out_group_norm: GroupNormWeights,
    /// Lazily populated CPU copies used by TimeMixer native fast paths.
    /// Model weights are immutable after load; tests should finish mutating
    /// fixture weights before the first native helper call.
    pub(crate) native_f32_cache: OnceLock<TimeMixerNativeF32Weights>,
}

impl Rwkv7RnnTimeMixerWeights {
    fn consume(
        tensors: &mut TensorConsumer,
        module_index: usize,
        layer_index: usize,
    ) -> Result<Self, ModelWeightsError> {
        let prefix = format!("rwkv_modules.{module_index}.blocks.{layer_index}.time_mixer");
        Ok(Self {
            layer_id: layer_index,
            n_heads: N_HEADS,
            head_size: HEAD_SIZE,
            layer_norm: tensors.layer_norm(&format!("{prefix}.layer_norm"))?,
            rkvdag_lerp: tensors.take(format!("{prefix}.rkvdag_lerp"))?,
            bonus: tensors.take(format!("{prefix}.bonus"))?,
            w_r: tensors.linear(&format!("{prefix}.W_r"), false)?,
            w_k: tensors.linear(&format!("{prefix}.W_k"), false)?,
            w_v: tensors.linear(&format!("{prefix}.W_v"), false)?,
            w_o: tensors.linear(&format!("{prefix}.W_o"), false)?,
            k_scale_linear: tensors.linear(&format!("{prefix}.k_scale_linear"), true)?,
            v_scale_linear: tensors.linear(&format!("{prefix}.v_scale_linear"), true)?,
            v_lora_simple: tensors.lora_simple(&format!("{prefix}.v_lora_simple"))?,
            a_lora_simple: tensors.lora_simple(&format!("{prefix}.a_lora_simple"))?,
            d_lora_mlp: tensors.lora_simple(&format!("{prefix}.d_lora_mlp"))?,
            gate_lora: GateLoraWeights {
                a: tensors.linear(&format!("{prefix}.lora_A_g"), false)?,
                b: tensors.linear(&format!("{prefix}.lora_B_g"), false)?,
            },
            out_group_norm: tensors.group_norm(&format!("{prefix}.out_group_norm"))?,
            native_f32_cache: OnceLock::new(),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.layer_norm.tensor_count()
            + 2
            + self.w_r.tensor_count()
            + self.w_k.tensor_count()
            + self.w_v.tensor_count()
            + self.w_o.tensor_count()
            + self.k_scale_linear.tensor_count()
            + self.v_scale_linear.tensor_count()
            + self.v_lora_simple.tensor_count()
            + self.a_lora_simple.tensor_count()
            + self.d_lora_mlp.tensor_count()
            + self.gate_lora.tensor_count()
            + self.out_group_norm.tensor_count()
    }
}

#[derive(Debug)]
pub(crate) struct LinearF32Weights {
    pub(crate) out_dim: usize,
    pub(crate) in_dim: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) blocked_input8_dot12: Option<LinearBlockedInput8Dot12Weights>,
}

#[derive(Debug)]
pub(crate) struct LinearBlockedInput8Dot12Weights {
    pub(crate) values: Vec<f32>,
    pub(crate) full_out_dim: usize,
    pub(crate) in_dim: usize,
}

#[derive(Debug)]
pub(crate) struct TimeMixerNativeF32Weights {
    pub(crate) rkvdag_lerp: Vec<f32>,
    pub(crate) bonus: Vec<f32>,
    pub(crate) out_group_norm_weight: Vec<f32>,
    pub(crate) out_group_norm_bias: Vec<f32>,
    pub(crate) w_r: LinearF32Weights,
    pub(crate) w_k: LinearF32Weights,
    pub(crate) w_v: LinearF32Weights,
    pub(crate) w_o: LinearF32Weights,
    pub(crate) k_scale_linear: LinearF32Weights,
    pub(crate) v_scale_linear: LinearF32Weights,
    pub(crate) v_lora_a: LinearF32Weights,
    pub(crate) v_lora_b: LinearF32Weights,
    pub(crate) a_lora_a: LinearF32Weights,
    pub(crate) a_lora_b: LinearF32Weights,
    pub(crate) d_lora_a: LinearF32Weights,
    pub(crate) d_lora_b: LinearF32Weights,
    pub(crate) gate_lora_a: LinearF32Weights,
    pub(crate) gate_lora_b: LinearF32Weights,
}

#[derive(Debug)]
pub struct Rwkv7RnnChannelMixerWeights {
    pub layer_id: usize,
    pub channel_dim: usize,
    pub layer_norm: LayerNormWeights,
    pub lerp_k: Tensor,
    pub w_k: LinearWeights,
    pub w_v: LinearWeights,
    pub(crate) native_f32_cache: OnceLock<ChannelMixerNativeF32Weights>,
}

#[derive(Debug)]
pub(crate) struct ChannelMixerNativeF32Weights {
    pub(crate) w_k: LinearF32Weights,
    pub(crate) w_v: LinearF32Weights,
}

impl Rwkv7RnnChannelMixerWeights {
    fn consume(
        tensors: &mut TensorConsumer,
        module_index: usize,
        layer_index: usize,
        config: ModuleConfig,
    ) -> Result<Self, ModelWeightsError> {
        let prefix = format!("rwkv_modules.{module_index}.blocks.{layer_index}.channel_mixer");
        Ok(Self {
            layer_id: layer_index,
            channel_dim: config.channel_dim(),
            layer_norm: tensors.layer_norm(&format!("{prefix}.layer_norm"))?,
            lerp_k: tensors.take(format!("{prefix}.lerp_k"))?,
            w_k: tensors.linear(&format!("{prefix}.W_k"), false)?,
            w_v: tensors.linear(&format!("{prefix}.W_v"), false)?,
            native_f32_cache: OnceLock::new(),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.layer_norm.tensor_count() + 1 + self.w_k.tensor_count() + self.w_v.tensor_count()
    }
}

#[derive(Debug)]
pub struct WHeadWeights {
    pub input_linear: LinearWeights,
    pub norm: LayerNormWeights,
    pub output_linear: LinearWeights,
}

impl WHeadWeights {
    fn consume(tensors: &mut TensorConsumer) -> Result<Self, ModelWeightsError> {
        Ok(Self {
            input_linear: tensors.linear("head_w.0", true)?,
            norm: tensors.layer_norm("head_w.2")?,
            output_linear: tensors.linear("head_w.4", true)?,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.input_linear.tensor_count()
            + self.norm.tensor_count()
            + self.output_linear.tensor_count()
    }
}

#[derive(Debug)]
pub struct LoraSimpleWeights {
    pub a: LinearWeights,
    pub b_and_lamb: LinearWeights,
}

impl LoraSimpleWeights {
    pub fn tensor_count(&self) -> usize {
        self.a.tensor_count() + self.b_and_lamb.tensor_count()
    }
}

#[derive(Debug)]
pub struct GateLoraWeights {
    pub a: LinearWeights,
    pub b: LinearWeights,
}

impl GateLoraWeights {
    pub fn tensor_count(&self) -> usize {
        self.a.tensor_count() + self.b.tensor_count()
    }
}

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearWeights {
    pub fn tensor_count(&self) -> usize {
        1 + usize::from(self.bias.is_some())
    }
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LayerNormWeights {
    pub fn tensor_count(&self) -> usize {
        2
    }
}

#[derive(Debug)]
pub struct GroupNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl GroupNormWeights {
    pub fn tensor_count(&self) -> usize {
        2
    }
}

#[derive(Debug, Error)]
pub enum ModelWeightsError {
    #[error(transparent)]
    Load(#[from] WeightLoadError),
    #[error("{0}")]
    Validation(Box<WeightValidationError>),
    #[error("missing tensor while constructing model weights: {key}")]
    MissingTensor { key: String },
    #[error("unused tensors after constructing model weights: {}", preview_keys(.keys, 12))]
    UnusedTensors { keys: Vec<String> },
}

pub fn load_srs_rwkv_rnn_weights<P: AsRef<Path>>(
    path: P,
) -> Result<SrsRwkvRnnWeights, ModelWeightsError> {
    SrsRwkvRnnWeights::from_loaded_weights(load_rwkv_srs_weights(path)?)
}

struct TensorConsumer {
    tensors: BTreeMap<String, Tensor>,
}

impl TensorConsumer {
    fn new(tensors: BTreeMap<String, Tensor>) -> Self {
        Self { tensors }
    }

    fn take(&mut self, key: impl Into<String>) -> Result<Tensor, ModelWeightsError> {
        let key = key.into();
        self.tensors
            .remove(&key)
            .ok_or(ModelWeightsError::MissingTensor { key })
    }

    fn finish(self) -> Result<(), ModelWeightsError> {
        if self.tensors.is_empty() {
            Ok(())
        } else {
            Err(ModelWeightsError::UnusedTensors {
                keys: self.tensors.into_keys().collect(),
            })
        }
    }

    fn linear(&mut self, prefix: &str, bias: bool) -> Result<LinearWeights, ModelWeightsError> {
        Ok(LinearWeights {
            weight: self.take(format!("{prefix}.weight"))?,
            bias: if bias {
                Some(self.take(format!("{prefix}.bias"))?)
            } else {
                None
            },
        })
    }

    fn layer_norm(&mut self, prefix: &str) -> Result<LayerNormWeights, ModelWeightsError> {
        Ok(LayerNormWeights {
            weight: self.take(format!("{prefix}.weight"))?,
            bias: self.take(format!("{prefix}.bias"))?,
        })
    }

    fn group_norm(&mut self, prefix: &str) -> Result<GroupNormWeights, ModelWeightsError> {
        Ok(GroupNormWeights {
            weight: self.take(format!("{prefix}.weight"))?,
            bias: self.take(format!("{prefix}.bias"))?,
        })
    }

    fn lora_simple(&mut self, prefix: &str) -> Result<LoraSimpleWeights, ModelWeightsError> {
        Ok(LoraSimpleWeights {
            a: self.linear(&format!("{prefix}.A"), false)?,
            b_and_lamb: self.linear(&format!("{prefix}.B_and_lamb"), true)?,
        })
    }
}

fn preview_keys(keys: &[String], limit: usize) -> String {
    let mut shown = keys.iter().take(limit).cloned().collect::<Vec<_>>();
    if keys.len() > limit {
        shown.push(format!("... and {} more", keys.len() - limit));
    }
    format!("{shown:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{expected_weight_specs, HEAD_SIZE, MODULE_CONFIGS, N_HEADS};
    use candle_core::{DType, Device};

    #[test]
    fn typed_model_consumes_complete_expected_schema_exactly_once() {
        let model = SrsRwkvRnnWeights::from_tensor_map(zero_tensor_map()).unwrap();

        assert_eq!(model.tensor_count(), expected_weight_specs().len());
        assert_eq!(model.rwkv_modules.len(), MODULE_CONFIGS.len());
        assert_eq!(model.block_count(), 16);
        assert_eq!(
            model
                .rwkv_modules
                .iter()
                .map(|module| module.blocks.len())
                .collect::<Vec<_>>(),
            MODULE_CONFIGS
                .iter()
                .map(|config| config.n_layers)
                .collect::<Vec<_>>()
        );
        assert_eq!(model.rwkv_modules[0].module_index, 0);
        assert_eq!(model.rwkv_modules[4].blocks[3].layer_id, 3);
        assert_eq!(model.rwkv_modules[4].blocks[3].time_mixer.n_heads, N_HEADS);
        assert_eq!(
            model.rwkv_modules[4].blocks[3].time_mixer.head_size,
            HEAD_SIZE
        );
    }

    #[test]
    fn typed_model_reports_missing_expected_tensor() {
        let mut tensors = zero_tensor_map();
        tensors.remove("features2card.0.bias");

        let err = SrsRwkvRnnWeights::from_tensor_map(tensors).unwrap_err();

        assert!(matches!(
            err,
            ModelWeightsError::MissingTensor { ref key } if key == "features2card.0.bias"
        ));
    }

    #[test]
    fn typed_model_reports_unused_tensor() {
        let mut tensors = zero_tensor_map();
        tensors.insert(
            "unexpected.weight".to_string(),
            Tensor::zeros(&[1usize], DType::F32, &Device::Cpu).unwrap(),
        );

        let err = SrsRwkvRnnWeights::from_tensor_map(tensors).unwrap_err();

        assert!(matches!(
            err,
            ModelWeightsError::UnusedTensors { ref keys } if keys == &vec!["unexpected.weight".to_string()]
        ));
    }

    #[test]
    fn committed_pretrained_checkpoints_build_typed_model_weights() {
        let repo = repo_root();
        let paths = [
            repo.join("tests/fixtures/models/RWKV_trained_on_101_4999.safetensors"),
            repo.join("tests/fixtures/models/RWKV_trained_on_5000_10000.safetensors"),
        ];

        for path in paths {
            let model = load_srs_rwkv_rnn_weights(&path).unwrap();
            assert_eq!(model.tensor_count(), expected_weight_specs().len());
            assert_eq!(model.block_count(), 16);
            assert_eq!(
                model.rwkv_modules[4].blocks[3].time_mixer.bonus.dims(),
                &[1, 1, N_HEADS, HEAD_SIZE],
                "{}",
                path.display()
            );
            assert_eq!(
                model.rwkv_modules[0].blocks[0]
                    .channel_mixer
                    .w_k
                    .weight
                    .dims(),
                &[192, 128],
                "{}",
                path.display()
            );
        }
    }

    fn zero_tensor_map() -> BTreeMap<String, Tensor> {
        expected_weight_specs()
            .into_iter()
            .map(|(name, spec)| {
                let tensor = Tensor::zeros(spec.shape.as_slice(), DType::F32, &Device::Cpu)
                    .unwrap_or_else(|err| panic!("failed to create tensor {name}: {err}"));
                (name, tensor)
            })
            .collect()
    }

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives under rust/rwkv-srs-cpu")
            .to_path_buf()
    }
}
