use super::MessageKind;
use crate::world::streaming::ChunkStreamer;

pub const MIN_RENDER_DISTANCE: i32 = 1;
pub const MAX_RENDER_DISTANCE: i32 = 512;

pub struct CommandResponse {
    pub text: String,
    pub kind: MessageKind,
}

fn ok(text: impl Into<String>) -> CommandResponse {
    CommandResponse {
        text: text.into(),
        kind: MessageKind::CommandOk,
    }
}

fn err(text: impl Into<String>) -> CommandResponse {
    CommandResponse {
        text: text.into(),
        kind: MessageKind::CommandError,
    }
}

/// Executes a chat line that starts with `/` (the leading slash is optional
/// here — callers may strip it themselves) against live game state.
pub fn execute(input: &str, streamer: &mut ChunkStreamer) -> CommandResponse {
    let trimmed = input.strip_prefix('/').unwrap_or(input);
    let mut tokens = trimmed.split_whitespace();

    match tokens.next() {
        Some("settings") => {
            let args: Vec<&str> = tokens.collect();
            execute_settings(&args, streamer)
        }
        Some("help") => ok("Commands: /settings rd <chunks>, /help"),
        Some(other) => err(format!(
            "Unknown command: /{other}. Type /help for a list of commands."
        )),
        None => err("Unknown command. Type /help for a list of commands."),
    }
}

fn execute_settings(args: &[&str], streamer: &mut ChunkStreamer) -> CommandResponse {
    match args {
        ["rd", value] => match value.parse::<i32>() {
            Ok(n) if (MIN_RENDER_DISTANCE..=MAX_RENDER_DISTANCE).contains(&n) => {
                streamer.set_load_radius(n);
                ok(format!("Render distance set to {n} chunks."))
            }
            Ok(n) => err(format!(
                "Render distance must be between {MIN_RENDER_DISTANCE} and {MAX_RENDER_DISTANCE} chunks (got {n})."
            )),
            Err(_) => err(format!("'{value}' is not a number.")),
        },
        ["rd"] => err("Usage: /settings rd <chunks>"),
        [sub, ..] => err(format!("Unknown settings subcommand: {sub}. Available: rd")),
        [] => err("Usage: /settings rd <chunks>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streamer() -> ChunkStreamer {
        ChunkStreamer::new(0, 6, 8)
    }

    #[test]
    fn settings_rd_sets_load_radius() {
        let mut streamer = streamer();
        let response = execute("/settings rd 10", &mut streamer);
        assert_eq!(response.text, "Render distance set to 10 chunks.");
        assert!(matches!(response.kind, MessageKind::CommandOk));
        assert_eq!(streamer.load_radius(), 10);
    }

    #[test]
    fn settings_rd_rejects_out_of_range() {
        let mut streamer = streamer();
        let response = execute("/settings rd 999", &mut streamer);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert_eq!(
            streamer.load_radius(),
            6,
            "invalid input must not change state"
        );
    }

    #[test]
    fn settings_rd_rejects_non_numeric() {
        let mut streamer = streamer();
        let response = execute("/settings rd banana", &mut streamer);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert_eq!(streamer.load_radius(), 6);
    }

    #[test]
    fn settings_rd_missing_argument_shows_usage() {
        let mut streamer = streamer();
        let response = execute("/settings rd", &mut streamer);
        assert_eq!(response.text, "Usage: /settings rd <chunks>");
    }

    #[test]
    fn unknown_command_is_reported() {
        let mut streamer = streamer();
        let response = execute("/foo", &mut streamer);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert!(response.text.contains("/foo"));
    }

    #[test]
    fn leading_slash_is_optional() {
        let mut streamer = streamer();
        let response = execute("settings rd 4", &mut streamer);
        assert_eq!(streamer.load_radius(), 4);
        let _ = response;
    }
}
