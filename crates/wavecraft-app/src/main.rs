mod app;
mod ui;

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use wavecraft_core::fixtures::generate_sample_fixtures;
use wavecraft_core::format::inspect_file;

use crate::app::WavecraftApp;

const USAGE: &str = "\
Wavecraft — Audio Waveform Editor & Binary Format Inspector

Usage:
  wavecraft                         Launch GUI workbench
  wavecraft <file>                  Launch GUI with <file> opened
  wavecraft --inspect <file>        Inspect binary structure and metadata in CLI
  wavecraft --generate-samples <dir> Generate demo WAV & PCM fixtures in <dir>
  wavecraft --help | -h             Show this help message
";

/// Parsed CLI command for `wavecraft-app`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliCommand {
    Help,
    GenerateSamples(PathBuf),
    Inspect(PathBuf),
    LaunchGui(Option<PathBuf>),
}

/// Parses command-line arguments (including argv[0]) into a [`CliCommand`].
pub fn parse_cli_args<I, S>(args: I) -> Result<CliCommand, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut iter = args.into_iter();
    let _program = iter.next();
    let rest: Vec<String> = iter.map(|s| s.as_ref().to_string()).collect();

    if rest.is_empty() {
        return Ok(CliCommand::LaunchGui(None));
    }

    match rest[0].as_str() {
        "--help" | "-h" => {
            if rest.len() > 1 {
                return Err(format!("unexpected argument '{}' after --help", rest[1]));
            }
            Ok(CliCommand::Help)
        }
        "--generate-samples" => {
            if rest.len() < 2 {
                return Err("--generate-samples requires a <dir> argument".to_string());
            }
            if rest.len() > 2 {
                return Err(format!("unexpected extra argument '{}'", rest[2]));
            }
            Ok(CliCommand::GenerateSamples(PathBuf::from(&rest[1])))
        }
        "--inspect" => {
            if rest.len() < 2 {
                return Err("--inspect requires a <file> argument".to_string());
            }
            if rest.len() > 2 {
                return Err(format!("unexpected extra argument '{}'", rest[2]));
            }
            Ok(CliCommand::Inspect(PathBuf::from(&rest[1])))
        }
        arg if arg.starts_with('-') => Err(format!("unknown option '{arg}'")),
        file => {
            if rest.len() > 1 {
                return Err(format!("unexpected extra argument '{}'", rest[1]));
            }
            Ok(CliCommand::LaunchGui(Some(PathBuf::from(file))))
        }
    }
}

/// Executes non-GUI CLI commands (`--help`, `--generate-samples`, `--inspect`) and returns
/// formatted stdout output or an error string.
pub fn run_cli_command(cmd: &CliCommand) -> Result<Option<String>, String> {
    match cmd {
        CliCommand::Help => Ok(Some(USAGE.to_string())),
        CliCommand::GenerateSamples(dir) => {
            let paths = generate_sample_fixtures(dir)
                .map_err(|e| format!("failed to generate samples in {}: {e}", dir.display()))?;
            let mut out = String::new();
            for path in paths {
                let _ = writeln!(out, "{}", path.display());
            }
            Ok(Some(out))
        }
        CliCommand::Inspect(file) => {
            let inspected = inspect_file(file, None)
                .map_err(|e| format!("failed to inspect {}: {e}", file.display()))?;
            let meta = &inspected.metadata;
            let mut out = String::new();
            let _ = writeln!(out, "Format:      {}", meta.format.label());
            let _ = writeln!(out, "Sample Rate: {} Hz", meta.sample_rate);
            let _ = writeln!(out, "Channels:    {}", meta.num_channels);
            let _ = writeln!(out, "Frames:      {}", meta.num_frames);
            let _ = writeln!(out, "Duration:    {:.4} s", meta.duration_secs);
            let _ = writeln!(out, "Encoding:    {}", meta.encoding_label);
            let _ = writeln!(out, "File Size:   {} bytes", meta.file_size_bytes);
            let _ = writeln!(out);
            out.push_str(&inspected.tree.format_tree_text());
            Ok(Some(out))
        }
        CliCommand::LaunchGui(_) => Ok(None),
    }
}

fn launch_gui(initial_file: Option<PathBuf>) -> Result<(), String> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Wavecraft")
            .with_inner_size([1440.0, 860.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Wavecraft",
        native_options,
        Box::new(|cc| Ok(Box::new(WavecraftApp::new(cc, initial_file)))),
    )
    .map_err(|e| format!("GUI error: {e}"))
}

fn main() -> ExitCode {
    let cmd = match parse_cli_args(std::env::args()) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("error: {err}\n\n{USAGE}");
            return ExitCode::from(1);
        }
    };

    match run_cli_command(&cmd) {
        Ok(Some(output)) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Ok(None) => {
            let initial_file = match cmd {
                CliCommand::LaunchGui(path) => path,
                _ => None,
            };
            if let Err(err) = launch_gui(initial_file) {
                eprintln!("{err}");
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cli_args_handles_all_modes_and_errors() {
        assert_eq!(
            parse_cli_args(["wavecraft"]).unwrap(),
            CliCommand::LaunchGui(None)
        );
        assert_eq!(
            parse_cli_args(["wavecraft", "tone.wav"]).unwrap(),
            CliCommand::LaunchGui(Some(PathBuf::from("tone.wav")))
        );
        assert_eq!(
            parse_cli_args(["wavecraft", "--help"]).unwrap(),
            CliCommand::Help
        );
        assert_eq!(
            parse_cli_args(["wavecraft", "-h"]).unwrap(),
            CliCommand::Help
        );
        assert_eq!(
            parse_cli_args(["wavecraft", "--inspect", "a.wav"]).unwrap(),
            CliCommand::Inspect(PathBuf::from("a.wav"))
        );
        assert_eq!(
            parse_cli_args(["wavecraft", "--generate-samples", "out_dir"]).unwrap(),
            CliCommand::GenerateSamples(PathBuf::from("out_dir"))
        );

        assert!(parse_cli_args(["wavecraft", "--inspect"]).is_err());
        assert!(parse_cli_args(["wavecraft", "--generate-samples"]).is_err());
        assert!(parse_cli_args(["wavecraft", "--unknown"]).is_err());
        assert!(parse_cli_args(["wavecraft", "a.wav", "b.wav"]).is_err());
    }

    #[test]
    fn run_cli_command_generate_samples_then_inspect_sine_440hz() {
        let tmp =
            std::env::temp_dir().join(format!("wavecraft_app_cli_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);

        let gen_cmd = CliCommand::GenerateSamples(tmp.clone());
        let gen_out = run_cli_command(&gen_cmd).unwrap().unwrap();
        assert!(gen_out.contains("sine_440hz.wav"));
        assert!(tmp.join("sine_440hz.wav").is_file());

        let inspect_cmd = CliCommand::Inspect(tmp.join("sine_440hz.wav"));
        let inspect_out = run_cli_command(&inspect_cmd).unwrap().unwrap();
        assert!(inspect_out.contains("RIFF"));
        assert!(inspect_out.contains("fmt "));
        assert!(inspect_out.contains("44100"));
        assert!(inspect_out.contains("data"));
        assert!(inspect_out.contains("Format:"));
        assert!(inspect_out.contains("Sample Rate:"));
        assert!(inspect_out.contains("Channels:"));
        assert!(inspect_out.contains("Frames:"));
        assert!(inspect_out.contains("Duration:"));
        assert!(inspect_out.contains("Encoding:"));
        assert!(inspect_out.contains("File Size:"));

        let help_out = run_cli_command(&CliCommand::Help).unwrap().unwrap();
        assert!(help_out.contains("--inspect"));
        assert!(help_out.contains("--generate-samples"));

        let missing = run_cli_command(&CliCommand::Inspect(tmp.join("does_not_exist.wav")));
        assert!(missing.is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
