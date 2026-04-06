// The library crate owns all modules; the binary just uses them.
use spike_lmo::tensor;

fn main() {
    println!("SpikeLMo — SNN/LLM Fusion Framework (candle-free)");
    println!("  neuromod   : SNN neurons + STDP + neuromodulators");
    println!("  spikenaut-encoder   : sensory encoding pipelines");
    println!("  spikenaut-reward    : homeostatic reward computation");
    println!("  spikenaut-telemetry : hardware telemetry snapshots");
    println!();

    // Smoke test: custom tensor matmul (candle replacement)
    let a = tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor::Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = tensor::ops::matmul(&a, &b);
    println!("[tensor] matmul 2×2: {:?}", c.data());

    // Smoke test: neuromod LIF neuron
    let mut neuron = neuromod::LifNeuron::default();
    neuron.weights = vec![1.0; 16];
    neuron.integrate(1.5);
    let vm_before = neuron.membrane_potential;
    let fired = neuron.check_fire();
    println!("[neuromod] LIF Vm_pre={vm_before:.4}, fired={fired:?}");

    // Smoke test: spikenaut-reward
    let mut reward_state = spikenaut_reward::MiningRewardState::new();
    let telem = spikenaut_reward::GpuTelemetry {
        hashrate_mh: 0.012,
        power_w: 340.0,
        gpu_temp_c: 72.0,
        gpu_clock_mhz: 2640.0,
        vddcr_gfx_v: 1.0,
        ..Default::default()
    };
    let reward = reward_state.compute(&telem, Some(68.0));
    println!("[spikenaut-reward] mining dopamine: {reward:.4}");

    // Smoke test: spikenaut-encoder
    let encoder = spikenaut_encoder::SensoryEncoder::new();
    let stats = encoder.get_stats();
    println!("[spikenaut-encoder] encoder ready, gpu_temp_avg={:.1}", stats.gpu_temp_avg);

    println!("\nAll modules loaded. Ready for training pipeline.");
}
