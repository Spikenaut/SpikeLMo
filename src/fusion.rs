use crate::tensor::Tensor;
use crate::tensor::ops::matmul;
use neuromod::{SpikingNetwork, NeuroModulators};
use serde::{Deserialize, Serialize};

/// SNN/LLM Fusion Layer
///
/// Bridges neuromod's SpikingNetwork with the transformer hidden states.
/// The SNN observes the LLM's hidden representations and modulates them
/// through neuromodulator-driven gating before they reach the LM head.
///
/// Architecture:
///   LLM hidden [seq, dim] ──┐
///                            ├─► projection → SNN stimuli → spike outputs
///   Telemetry / reward ──────┘                     │
///                                                  ▼
///                            SNN modulation weights (gate)
///                                                  │
///                            LLM hidden ◉ gate ────► fused output [seq, dim]
#[derive(Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub llm_dim: usize,
    pub snn_channels: usize,
    pub gate_mode: GateMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GateMode {
    /// SNN spike rates multiplicatively gate LLM hidden states
    Multiplicative,
    /// SNN outputs are projected back and added to hidden states
    Additive,
    /// Both: gate = sigmoid(W_g · spike_rates), output = gate ⊙ hidden + (1−gate) ⊙ snn_proj
    Hybrid,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            llm_dim: 2048,
            snn_channels: 16,
            gate_mode: GateMode::Hybrid,
        }
    }
}

/// The fusion layer connecting neuromod SNN ↔ transformer LLM.
#[derive(Serialize, Deserialize)]
pub struct SnnLlmFusion {
    pub config: FusionConfig,
    /// Projects LLM hidden dim down to SNN input channels: [llm_dim, snn_channels]
    pub proj_down: Tensor,
    /// Projects SNN spike outputs back up: [snn_channels, llm_dim]
    pub proj_up: Tensor,
    /// Gate projection: [snn_channels, llm_dim]
    pub gate_proj: Tensor,
    /// The neuromod SpikingNetwork (LIF + Izhikevich banks, neuromodulators, STDP)
    pub snn: SpikingNetwork,
}

impl SnnLlmFusion {
    pub fn new(config: FusionConfig) -> Self {
        let scale_down = 1.0 / (config.llm_dim as f32).sqrt();
        let scale_up = 1.0 / (config.snn_channels as f32).sqrt();

        Self {
            proj_down: Tensor::randn(
                &[config.llm_dim, config.snn_channels],
                0.0,
                scale_down,
            ),
            proj_up: Tensor::randn(
                &[config.snn_channels, config.llm_dim],
                0.0,
                scale_up,
            ),
            gate_proj: Tensor::randn(
                &[config.snn_channels, config.llm_dim],
                0.0,
                scale_up,
            ),
            snn: SpikingNetwork::new(),
            config,
        }
    }

    /// Forward pass: takes LLM hidden states [seq_len, llm_dim] and optional
    /// neuromodulators, returns fused output [seq_len, llm_dim].
    ///
    /// Steps:
    /// 1. Project hidden → SNN channel stimuli
    /// 2. Feed each seq position through the SpikingNetwork as a timestep
    /// 3. Collect spike rates from LIF bank
    /// 4. Use spike rates to gate / modulate the LLM hidden states
    pub fn forward(
        &mut self,
        hidden: &Tensor,
        modulators: Option<NeuroModulators>,
    ) -> Tensor {
        let seq_len = hidden.shape()[0];
        let _llm_dim = self.config.llm_dim;
        let nch = self.config.snn_channels;

        // Apply external neuromodulators if provided
        if let Some(mods) = modulators {
            self.snn.modulators = mods;
        }

        // Project down: [seq, llm_dim] × [llm_dim, nch] → [seq, nch]
        let stimuli = matmul(hidden, &self.proj_down);

        // Run SNN for each seq position, collect spike rates
        let mut spike_rates = vec![0.0f32; seq_len * nch];
        let stim_data = stimuli.data();

        for t in 0..seq_len {
            // Build per-channel stimulus array (clamped to [0, 1])
            // neuromod expects &[f32; 16] (NUM_INPUT_CHANNELS)
            let mut channel_stim = [0.0f32; 16];
            for ch in 0..nch.min(16) {
                channel_stim[ch] = stim_data[t * nch + ch].clamp(0.0, 1.0);
            }

            // Step the SpikingNetwork with this stimulus + current modulators
            self.snn.step(&channel_stim, &self.snn.modulators.clone());

            // Read spike outputs from LIF bank
            for (i, neuron) in self.snn.neurons.iter().enumerate() {
                if i < nch {
                    spike_rates[t * nch + i] = if neuron.last_spike { 1.0 } else { 0.0 };
                }
            }
        }

        let spike_tensor = Tensor::from_vec(spike_rates, &[seq_len, nch]);

        // Apply fusion based on gate mode
        match self.config.gate_mode {
            GateMode::Multiplicative => {
                // spike_rates → project up → sigmoid → element-wise gate
                let gate = matmul(&spike_tensor, &self.gate_proj).fast_sigmoid();
                hidden.mul(&gate)
            }
            GateMode::Additive => {
                // spike_rates → project up → add to hidden
                let snn_contribution = matmul(&spike_tensor, &self.proj_up);
                hidden.add(&snn_contribution)
            }
            GateMode::Hybrid => {
                // gate = fast_sigmoid(spike_rates × gate_proj)
                let gate = matmul(&spike_tensor, &self.gate_proj).fast_sigmoid();
                // Scale gate to [0, 1] range: (fast_sigmoid output is in (-1,1), shift to (0,1))
                let gate = gate.scale(0.5).add_scalar(0.5);
                let snn_proj = matmul(&spike_tensor, &self.proj_up);
                // fused = gate ⊙ hidden + (1 − gate) ⊙ snn_proj
                let one_minus_gate = gate.scale(-1.0).add_scalar(1.0);
                let gated_hidden = hidden.mul(&gate);
                let gated_snn = snn_proj.mul(&one_minus_gate);
                gated_hidden.add(&gated_snn)
            }
        }
    }

    /// Returns the current SNN weight matrix (for FPGA export or inspection).
    pub fn snn_weights_flat(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        for neuron in &self.snn.neurons {
            flat.extend_from_slice(&neuron.weights);
        }
        flat
    }

    pub fn param_count(&self) -> usize {
        self.proj_down.numel() + self.proj_up.numel() + self.gate_proj.numel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_forward_hybrid() {
        let cfg = FusionConfig {
            llm_dim: 64,
            snn_channels: 16,
            gate_mode: GateMode::Hybrid,
        };
        let mut fusion = SnnLlmFusion::new(cfg);
        let hidden = Tensor::randn(&[4, 64], 0.0, 0.1);
        let out = fusion.forward(&hidden, None);
        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_fusion_forward_additive() {
        let cfg = FusionConfig {
            llm_dim: 32,
            snn_channels: 16,
            gate_mode: GateMode::Additive,
        };
        let mut fusion = SnnLlmFusion::new(cfg);
        let hidden = Tensor::randn(&[2, 32], 0.0, 0.1);
        let out = fusion.forward(&hidden, None);
        assert_eq!(out.shape(), &[2, 32]);
    }
}
