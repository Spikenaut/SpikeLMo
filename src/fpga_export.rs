//! FPGA export — thin wrapper around `spikenaut-fpga`.
//!
//! Re-exports the FpgaParameterExporter and adds convenience functions
//! to export a neuromod SpikingNetwork directly.

use std::path::Path;

// Re-export spikenaut-fpga's public API
pub use spikenaut_fpga::{
    FpgaParameterExporter,
    FpgaParameters,
    FpgaMetadata,
    format_q88_hex,
    q88_to_f32,
};

/// Export a trained neuromod SpikingNetwork to FPGA .mem files via spikenaut-fpga.
///
/// Uses `FpgaParameterExporter::export_to_mem_files` which writes:
///   - `parameters.mem`         (thresholds in Q8.8 hex)
///   - `parameters_weights.mem` (weights in Q8.8 hex)
///   - `parameters_decay.mem`   (decay rates in Q8.8 hex)
///   - `parameters.json`        (metadata)
pub fn export_network(
    dir: &Path,
    network: &neuromod::SpikingNetwork,
) -> Result<FpgaParameters, Box<dyn std::error::Error>> {
    let thresholds: Vec<f32> = network.neurons.iter().map(|n| n.threshold).collect();
    let weights: Vec<Vec<f32>> = network.neurons.iter().map(|n| n.weights.clone()).collect();
    let decays: Vec<f32> = network.neurons.iter().map(|n| n.decay_rate).collect();

    let exporter = FpgaParameterExporter::from_params(thresholds, weights, decays);
    exporter.export_to_mem_files(dir)?;
    Ok(exporter.export())
}

/// Export the SNN from a fusion layer.
pub fn export_fusion_snn(
    dir: &Path,
    fusion: &crate::fusion::SnnLlmFusion,
) -> Result<FpgaParameters, Box<dyn std::error::Error>> {
    export_network(dir, &fusion.snn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q88_roundtrip() {
        let hex = format_q88_hex(1.0);
        let back = q88_to_f32(256); // 1.0 in Q8.8 = 256
        assert!((1.0 - back).abs() < 0.005, "roundtrip: 1.0 -> {hex} -> {back}");
    }

    #[test]
    fn test_exporter_smoke() {
        let mut exporter = FpgaParameterExporter::new();
        exporter.set_thresholds(vec![0.2; 16]);
        exporter.set_weights(vec![vec![0.5; 16]; 16]);
        exporter.set_decay_rates(vec![0.85; 16]);
        let params = exporter.export();
        assert!(!params.thresholds.is_empty());
    }
}
