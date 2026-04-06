use crate::fusion::SnnLlmFusion;
use spikenaut_reward::{GpuTelemetry, MiningRewardState};
use spikenaut_encoder::SensoryEncoder;
use spikenaut_telemetry::TelemetrySnapshot;
use serde::{Deserialize, Serialize};

/// Training configuration — ported from soma-engine's learning_trainer
/// with extensions for the SNN/LLM fusion pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub trace_decay: f32,
    pub stdp_lr: f32,
    pub eprop_lr: f32,
    pub synaptic_scaling_interval: usize,
    pub weight_clip_min: f32,
    pub weight_clip_max: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            learning_rate: 0.002,
            trace_decay: 0.85,
            stdp_lr: 0.01,
            eprop_lr: 0.002,
            synaptic_scaling_interval: 50,
            weight_clip_min: 0.0,
            weight_clip_max: 2.0,
        }
    }
}

/// Convert a TelemetrySnapshot into the GpuTelemetry struct that
/// spikenaut-reward expects.
fn snapshot_to_gpu_telem(snap: &TelemetrySnapshot) -> GpuTelemetry {
    GpuTelemetry {
        hashrate_mh: snap.dynex_hashrate_mh as f32,
        power_w: snap.gpu_power_w,
        gpu_temp_c: snap.gpu_temp_c,
        gpu_clock_mhz: snap.gpu_clock_mhz,
        vddcr_gfx_v: snap.vddcr_gfx_v,
        mem_clock_mhz: snap.mem_clock_mhz,
        fan_speed_pct: snap.fan_speed_pct,
        ..Default::default()
    }
}

/// Convert a TelemetrySnapshot into the SystemTelemetry struct that
/// spikenaut-encoder expects.
fn snapshot_to_system_telem(snap: &TelemetrySnapshot) -> spikenaut_encoder::types::SystemTelemetry {
    spikenaut_encoder::types::SystemTelemetry {
        cpu_tctl_c: snap.cpu_tctl_c,
        cpu_package_power_w: snap.cpu_package_power_w,
        cpu_ccd1_c: snap.cpu_ccd1_c,
        cpu_ccd2_c: snap.cpu_ccd2_c,
        gpu_temp_c: snap.gpu_temp_c,
        gpu_power_w: snap.gpu_power_w,
        gpu_clock_mhz: snap.gpu_clock_mhz,
        mem_clock_mhz: snap.mem_clock_mhz,
        mem_util_pct: snap.mem_util_pct,
        vddcr_gfx_v: snap.vddcr_gfx_v,
        fan_speed_pct: snap.fan_speed_pct,
        ..Default::default()
    }
}

/// Convert encoder's u16 spike channels into f32 stimuli for neuromod's
/// SpikingNetwork::step(&[f32; 16]).
fn spikes_to_stimuli(spikes: &[u16; 16]) -> [f32; 16] {
    let mut s = [0.0f32; 16];
    for (i, &v) in spikes.iter().enumerate() {
        s[i] = (v as f32).clamp(0.0, 1.0);
    }
    s
}

/// E-prop + OTTT trainer for the SNN/LLM fusion.
///
/// Uses:
///  - `spikenaut-reward::MiningRewardState` for reward computation
///  - `spikenaut-encoder::SensoryEncoder` for telemetry → spike encoding
///  - `spikenaut-telemetry::TelemetrySnapshot` as the input data type
///  - `neuromod::SpikingNetwork` (via fusion layer) for the SNN forward pass
pub struct FusionTrainer {
    pub fusion: SnnLlmFusion,
    pub config: TrainingConfig,
    pub reward_state: MiningRewardState,
    pub encoder: SensoryEncoder,
    /// OTTT presynaptic traces: one per input channel.
    pre_traces: [f32; 16],
    /// E-prop eligibility traces: [num_neurons × num_channels], row-major.
    eligibility: Vec<f32>,
}

impl FusionTrainer {
    pub fn new(fusion: SnnLlmFusion, config: TrainingConfig) -> Self {
        let num_neurons = fusion.snn.neurons.len();
        let num_channels = 16;
        Self {
            fusion,
            config,
            reward_state: MiningRewardState::new(),
            encoder: SensoryEncoder::new(),
            pre_traces: [0.0; 16],
            eligibility: vec![0.0; num_neurons * num_channels],
        }
    }

    /// Run one training tick from a TelemetrySnapshot.
    ///
    /// 1. Encode telemetry → 16-channel spikes via spikenaut-encoder
    /// 2. Compute reward via spikenaut-reward
    /// 3. Step the neuromod SpikingNetwork
    /// 4. E-prop eligibility trace update + weight modification
    pub fn tick(&mut self, snap: &TelemetrySnapshot) {
        // Reward from spikenaut-reward
        let gpu_telem = snapshot_to_gpu_telem(snap);
        let cpu_temp = if snap.cpu_tctl_c > 0.0 { Some(snap.cpu_tctl_c) } else { None };
        let reward = self.reward_state.compute(&gpu_telem, cpu_temp);

        // Encode via spikenaut-encoder → 16-channel spike train
        let sys_telem = snapshot_to_system_telem(snap);
        let spike_channels = self.encoder.encode_system_telemetry(&sys_telem);
        let stimuli = spikes_to_stimuli(&spike_channels);

        // Derive neuromodulators from the snapshot
        let mods = neuromod::NeuroModulators {
            dopamine: (gpu_telem.hashrate_mh * 0.1).clamp(0.0, 1.0),
            cortisol: snap.thermal_stress(),
            acetylcholine: (1.0 - (snap.vddcr_gfx_v - 1.0).abs() * 5.0).clamp(0.0, 1.0),
            tempo: (snap.gpu_clock_mhz / 2640.0).clamp(0.1, 2.0),
            mining_dopamine: reward.max(0.0),
        };

        // Update presynaptic OTTT traces
        for ch in 0..16 {
            self.pre_traces[ch] = self.config.trace_decay * self.pre_traces[ch]
                + stimuli[ch];
        }

        // Step the SNN
        let _spikes = self.fusion.snn.step(&stimuli, &mods);

        // E-prop eligibility trace update + weight modification
        let num_neurons = self.fusion.snn.neurons.len();
        let lr = self.config.eprop_lr;

        for ni in 0..num_neurons {
            let neuron = &self.fusion.snn.neurons[ni];
            // Surrogate gradient: fast_sigmoid(Vm - threshold)
            let surplus = neuron.membrane_potential - neuron.threshold;
            let surrogate = surplus / (1.0 + surplus.abs());

            for ch in 0..16 {
                let idx = ni * 16 + ch;
                // Update eligibility trace
                self.eligibility[idx] = self.config.trace_decay * self.eligibility[idx]
                    + surrogate * self.pre_traces[ch];
                // E-prop weight update: Δw = lr × reward × eligibility
                let dw = lr * reward * self.eligibility[idx];
                self.fusion.snn.neurons[ni].weights[ch] = (
                    self.fusion.snn.neurons[ni].weights[ch] + dw
                ).clamp(self.config.weight_clip_min, self.config.weight_clip_max);
            }
        }
    }

    /// Run multi-epoch training over a dataset of TelemetrySnapshots.
    pub fn train(&mut self, dataset: &[TelemetrySnapshot]) {
        for epoch in 0..self.config.epochs {
            let mut epoch_reward = 0.0f32;
            for snap in dataset {
                let gpu_telem = snapshot_to_gpu_telem(snap);
                let cpu_temp = if snap.cpu_tctl_c > 0.0 { Some(snap.cpu_tctl_c) } else { None };
                epoch_reward += self.reward_state.compute(&gpu_telem, cpu_temp);
                self.tick(snap);
            }
            let avg = epoch_reward / dataset.len().max(1) as f32;
            println!("Epoch {}/{}: avg_reward={avg:.4}", epoch + 1, self.config.epochs);

            // Synaptic scaling every N epochs
            if (epoch + 1) % self.config.synaptic_scaling_interval == 0 {
                self.synaptic_scaling();
            }
        }
    }

    /// L1 synaptic scaling to prevent runaway excitation (from neuromod engine.rs).
    fn synaptic_scaling(&mut self) {
        const WEIGHT_BUDGET: f32 = 2.0;
        for neuron in &mut self.fusion.snn.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > WEIGHT_BUDGET {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w *= scale;
                }
            }
        }
    }

    /// Load training data from a JSONL file of TelemetrySnapshots.
    pub fn load_jsonl(path: &std::path::Path) -> std::io::Result<Vec<TelemetrySnapshot>> {
        let file = std::fs::read_to_string(path)?;
        let mut records = Vec::new();
        for line in file.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            if let Ok(snap) = serde_json::from_str::<TelemetrySnapshot>(line) {
                records.push(snap);
            }
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{FusionConfig, GateMode};

    fn test_snapshot() -> TelemetrySnapshot {
        TelemetrySnapshot {
            gpu_temp_c: 72.0,
            gpu_power_w: 340.0,
            gpu_clock_mhz: 2640.0,
            mem_clock_mhz: 2400.0,
            vddcr_gfx_v: 1.0,
            fan_speed_pct: 60.0,
            cpu_tctl_c: 68.0,
            cpu_package_power_w: 45.0,
            dynex_hashrate_mh: 0.012,
            ..Default::default()
        }
    }

    #[test]
    fn test_single_tick() {
        let cfg = FusionConfig {
            llm_dim: 64,
            snn_channels: 16,
            gate_mode: GateMode::Hybrid,
        };
        let fusion = SnnLlmFusion::new(cfg);
        let mut trainer = FusionTrainer::new(fusion, TrainingConfig::default());
        trainer.tick(&test_snapshot());
    }

    #[test]
    fn test_multi_epoch() {
        let cfg = FusionConfig {
            llm_dim: 64,
            snn_channels: 16,
            gate_mode: GateMode::Hybrid,
        };
        let fusion = SnnLlmFusion::new(cfg);
        let mut trainer = FusionTrainer::new(
            fusion,
            TrainingConfig { epochs: 2, ..Default::default() },
        );
        let dataset = vec![test_snapshot(); 5];
        trainer.train(&dataset);
    }
}
