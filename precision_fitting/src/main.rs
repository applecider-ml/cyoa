use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rayon::prelude::*;

use lightcurve_fitting::{
    build_flux_bands, build_mag_bands, LightcurveFittingResult,
    fit_nonparametric, fit_parametric, fit_thermal, UncertaintyMethod,
};

#[derive(Parser)]
#[command(name = "boom-fit-batch")]
#[command(about = "Batch lightcurve fitting for ZTF transients")]
struct Cli {
    /// Input directory containing photometry data.
    /// Supports two layouts:
    ///   NPZ: {input_dir}/{split}/{obj_id}.npz  (photo_events format)
    ///   CSV: {input_dir}/{obj_id}/photometry.csv (legacy format)
    /// NPZ is tried first, then CSV as fallback.
    #[arg(long)]
    input_dir: PathBuf,

    /// Directory to write per-source JSON fitting results
    #[arg(long)]
    output_dir: PathBuf,

    /// CSV file with obj_id,split columns listing sources to process
    #[arg(long)]
    sources: PathBuf,

    /// Number of rayon threads
    #[arg(long, default_value = "8")]
    threads: usize,

    /// Maximum number of detections to use per source (0 = all).
    /// Keeps the first N rows (earliest by dt) for ablation studies.
    #[arg(long, default_value = "0")]
    max_detections: usize,
}

/// A source entry from splits.csv.
struct SourceEntry {
    obj_id: String,
    split: String,
}

/// Read splits.csv and return list of (obj_id, split) pairs.
fn read_source_list(path: &Path) -> Vec<SourceEntry> {
    let mut rdr = csv::Reader::from_path(path).expect("Failed to open sources CSV");
    let mut entries = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        let obj_id = record.get(0).unwrap_or("").to_string();
        let split = record.get(1).unwrap_or("").to_string();
        if !obj_id.is_empty() {
            entries.push(SourceEntry { obj_id, split });
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// NPZ reader (photo_events format)
// ---------------------------------------------------------------------------

/// Read photometry from an NPZ file (photo_events format).
///
/// NPZ columns: dt(0), dt_prev(1), band_id(2), logflux(3), logflux_err(4), ...
/// band_id: 0=g, 1=r, 2=i
/// logflux: log10(flux) where flux is in µJy (AB ZP = 23.9)
///
/// Returns (times, mags, mag_errs, band_names) with times as relative days
/// and magnitudes converted via mag = -2.5 * logflux + 23.9.
fn read_photometry_npz(path: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let file = fs::File::open(path).ok()?;
    let mut npz = NpzReader::new(file).ok()?;
    let data: Array2<f32> = npz.by_name("data.npy").ok()?;

    if data.ncols() < 5 {
        return None;
    }

    let n_rows = data.nrows();
    let mut times = Vec::with_capacity(n_rows);
    let mut mags = Vec::with_capacity(n_rows);
    let mut mag_errs = Vec::with_capacity(n_rows);
    let mut bands = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        let dt = data[[i, 0]] as f64;
        let band_id = data[[i, 2]] as i32;
        let logflux = data[[i, 3]] as f64;
        let logflux_err = data[[i, 4]] as f64;

        if !logflux.is_finite() || !logflux_err.is_finite() || logflux_err <= 0.0 {
            continue;
        }

        // Convert log10(flux_µJy) to AB magnitude
        let mag = -2.5 * logflux + 23.9;
        let mag_err = 2.5 * logflux_err;

        let band_name = match band_id {
            0 => "g",
            1 => "r",
            2 => "i",
            _ => continue,
        };

        times.push(dt);
        mags.push(mag);
        mag_errs.push(mag_err);
        bands.push(band_name.to_string());
    }

    if times.is_empty() {
        return None;
    }

    Some((times, mags, mag_errs, bands))
}

// ---------------------------------------------------------------------------
// CSV reader (legacy data_ztf format)
// ---------------------------------------------------------------------------

/// Read a single source's photometry.csv and return (times, mags, mag_errs, band_names).
/// Filters to rows with valid magpsf/sigmapsf and maps fid (1=g, 2=r, 3=i).
fn read_photometry_csv(path: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .ok()?;

    let headers = rdr.headers().ok()?.clone();
    let jd_idx = headers.iter().position(|h| h.trim() == "jd")?;
    let mag_idx = headers.iter().position(|h| h.trim() == "magpsf")?;
    let err_idx = headers.iter().position(|h| h.trim() == "sigmapsf")?;
    let fid_idx = headers.iter().position(|h| h.trim() == "fid")?;

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut mag_errs = Vec::new();
    let mut bands = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        let jd: f64 = match record.get(jd_idx).and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let mag: f64 = match record.get(mag_idx).and_then(|s| s.trim().parse::<f64>().ok()) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let err: f64 = match record.get(err_idx).and_then(|s| s.trim().parse::<f64>().ok()) {
            Some(v) if v.is_finite() && v > 0.0 => v,
            _ => continue,
        };
        let fid: i32 = match record.get(fid_idx).and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let band_name = match fid {
            1 => "g",
            2 => "r",
            3 => "i",
            _ => continue,
        };

        times.push(jd);
        mags.push(mag);
        mag_errs.push(err);
        bands.push(band_name.to_string());
    }

    if times.is_empty() {
        return None;
    }

    Some((times, mags, mag_errs, bands))
}

// ---------------------------------------------------------------------------
// Source processing
// ---------------------------------------------------------------------------

/// Process a single source: read photometry, run fitting, write JSON.
///
/// Tries NPZ format first ({input_dir}/{split}/{obj_id}.npz), then falls back
/// to legacy CSV format ({input_dir}/{obj_id}/photometry.csv).
fn process_source(
    obj_id: &str,
    split: &str,
    input_dir: &Path,
    output_dir: &Path,
    max_detections: usize,
) -> Result<(), String> {
    let npz_path = input_dir.join(split).join(format!("{}.npz", obj_id));
    let csv_path = input_dir.join(obj_id).join("photometry.csv");

    let (mut times, mut mags, mut mag_errs, mut bands) = if npz_path.exists() {
        read_photometry_npz(&npz_path)
            .ok_or_else(|| format!("Failed to parse NPZ for {}", obj_id))?
    } else if csv_path.exists() {
        read_photometry_csv(&csv_path)
            .ok_or_else(|| format!("Failed to parse CSV for {}", obj_id))?
    } else {
        return Err(format!("No photometry for {}", obj_id));
    };

    // Truncate to first N detections (NPZ data is already sorted by dt)
    if max_detections > 0 && times.len() > max_detections {
        times.truncate(max_detections);
        mags.truncate(max_detections);
        mag_errs.truncate(max_detections);
        bands.truncate(max_detections);
    }

    if times.len() < 5 {
        return Err(format!("Too few detections ({}) for {}", times.len(), obj_id));
    }

    // Build per-band data
    let mag_bands = build_mag_bands(&times, &mags, &mag_errs, &bands);
    let flux_bands = build_flux_bands(&times, &mags, &mag_errs, &bands);

    // Run nonparametric (on mag bands) + parametric (on flux bands) + thermal (on mag bands)
    let (nonparametric, trained_gps) = fit_nonparametric(&mag_bands);
    let parametric = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);
    let thermal = fit_thermal(&mag_bands, Some(&trained_gps));

    let result = LightcurveFittingResult {
        nonparametric,
        parametric,
        thermal,
    };

    // Write JSON
    let json = serde_json::to_string(&result)
        .map_err(|e| format!("JSON serialize error for {}: {}", obj_id, e))?;
    let out_path = output_dir.join(format!("{}.json", obj_id));
    fs::write(&out_path, json)
        .map_err(|e| format!("Write error for {}: {}", obj_id, e))?;

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    // Set rayon thread count
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .expect("Failed to set rayon thread pool");

    // Create output directory
    fs::create_dir_all(&cli.output_dir).expect("Failed to create output directory");

    // Read source list
    let sources = read_source_list(&cli.sources);
    let total = sources.len();
    eprintln!("Found {} sources to process", total);

    // Filter to sources that haven't been processed yet
    let pending: Vec<&SourceEntry> = sources
        .iter()
        .filter(|s| !cli.output_dir.join(format!("{}.json", s.obj_id)).exists())
        .collect();
    let n_pending = pending.len();
    eprintln!(
        "{} already done, {} remaining",
        total - n_pending,
        n_pending
    );

    // Progress bar
    let pb = ProgressBar::new(n_pending as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let success_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);

    // Process in parallel
    let max_det = cli.max_detections;
    pending.par_iter().for_each(|entry| {
        match process_source(&entry.obj_id, &entry.split, &cli.input_dir, &cli.output_dir, max_det) {
            Ok(()) => {
                success_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                fail_count.fetch_add(1, Ordering::Relaxed);
                eprintln!("WARN: {}", e);
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("done");

    let s = success_count.load(Ordering::Relaxed);
    let f = fail_count.load(Ordering::Relaxed);
    eprintln!("Finished: {} succeeded, {} failed out of {} total", s, f, n_pending);
}
