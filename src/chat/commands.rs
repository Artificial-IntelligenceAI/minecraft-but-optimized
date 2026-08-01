use super::MessageKind;
use crate::fps::FpsCounter;
use crate::world::streaming::ChunkStreamer;

pub const MIN_RENDER_DISTANCE: i32 = 1;
pub const MAX_RENDER_DISTANCE: i32 = 512;

/// Cap applied by `/settings maxfps true` — a plain on/off toggle doesn't say
/// *what* to cap at, so this is the value it means by "capped".
pub const DEFAULT_CAPPED_FPS: u32 = 60;
pub const MIN_FPS_CAP: u32 = 1;
pub const MAX_FPS_CAP: u32 = 1000;

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
pub fn execute(input: &str, streamer: &mut ChunkStreamer, fps: &mut FpsCounter) -> CommandResponse {
    let trimmed = input.strip_prefix('/').unwrap_or(input);
    let mut tokens = trimmed.split_whitespace();

    match tokens.next() {
        Some("settings") => {
            let args: Vec<&str> = tokens.collect();
            execute_settings(&args, streamer, fps)
        }
        Some("help") => {
            ok("Commands: /settings rd <chunks>, /settings maxfps <true|false|number>, /help")
        }
        Some(other) => err(format!(
            "Unknown command: /{other}. Type /help for a list of commands."
        )),
        None => err("Unknown command. Type /help for a list of commands."),
    }
}

fn execute_settings(
    args: &[&str],
    streamer: &mut ChunkStreamer,
    fps: &mut FpsCounter,
) -> CommandResponse {
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
        ["maxfps", value] => execute_maxfps(value, fps),
        ["maxfps"] => err("Usage: /settings maxfps <true|false> or /settings maxfps <number>"),
        [sub, ..] => err(format!(
            "Unknown settings subcommand: {sub}. Available: rd, maxfps"
        )),
        [] => err("Usage: /settings rd <chunks> or /settings maxfps <true|false|number>"),
    }
}

/// `true` caps at [`DEFAULT_CAPPED_FPS`], `false` uncaps, and a plain number
/// caps at that exact value (and implies capping is on).
fn execute_maxfps(value: &str, fps: &mut FpsCounter) -> CommandResponse {
    if let Ok(capped) = value.parse::<bool>() {
        return if capped {
            fps.cap = Some(DEFAULT_CAPPED_FPS);
            ok(format!("FPS capped at {DEFAULT_CAPPED_FPS}."))
        } else {
            fps.cap = None;
            ok("FPS uncapped.")
        };
    }

    match value.parse::<u32>() {
        Ok(n) if (MIN_FPS_CAP..=MAX_FPS_CAP).contains(&n) => {
            fps.cap = Some(n);
            ok(format!("FPS capped at {n}."))
        }
        Ok(n) => err(format!(
            "FPS cap must be between {MIN_FPS_CAP} and {MAX_FPS_CAP} (got {n})."
        )),
        Err(_) => err(format!("'{value}' is not true, false, or a number.")),
    }
}

/// A command's literal token sequence plus an optional trailing argument hint,
/// e.g. `["settings", "rd"]` + `Some("<chunks>")` for `/settings rd <chunks>`.
struct CommandSpec {
    tokens: &'static [&'static str],
    arg_hint: Option<&'static str>,
}

const COMMAND_SPECS: &[CommandSpec] = &[
    CommandSpec {
        tokens: &["help"],
        arg_hint: None,
    },
    CommandSpec {
        tokens: &["settings", "rd"],
        arg_hint: Some("<chunks>"),
    },
    CommandSpec {
        tokens: &["settings", "maxfps"],
        arg_hint: Some("<true|false|number>"),
    },
];

/// A Minecraft-style command suggestion: gray "ghost" text to display right
/// after what the player has typed, and (separately) the literal text a Tab
/// press should insert — which excludes argument placeholders like
/// `<chunks>`, since those aren't real input.
pub struct Suggestion {
    pub ghost_tail: String,
    pub tab_insert: Option<String>,
}

fn suggest_for_spec(spec: &CommandSpec, words: &[&str], partial: &str) -> Option<Suggestion> {
    if words.len() > spec.tokens.len() {
        return None;
    }
    if words.iter().zip(spec.tokens).any(|(w, t)| w != t) {
        return None;
    }

    if words.len() < spec.tokens.len() {
        let next_token = spec.tokens[words.len()];
        if !next_token.starts_with(partial) {
            return None;
        }
        let missing = &next_token[partial.len()..];
        let mut ghost_tail = missing.to_string();
        for t in &spec.tokens[words.len() + 1..] {
            ghost_tail.push(' ');
            ghost_tail.push_str(t);
        }
        if let Some(hint) = spec.arg_hint {
            ghost_tail.push(' ');
            ghost_tail.push_str(hint);
        }
        Some(Suggestion {
            ghost_tail,
            tab_insert: Some(format!("{missing} ")),
        })
    } else if !partial.is_empty() {
        None // mid-argument (all literal tokens typed, still typing the last word): freeform, no suggestion.
    } else {
        spec.arg_hint.map(|hint| Suggestion {
            ghost_tail: hint.to_string(),
            tab_insert: None,
        })
    }
}

/// Suggests a completion for a chat line starting with `/`, matching how
/// Minecraft shows the rest of a matching command in gray after the cursor.
/// Returns `None` for plain (non-command) text, when nothing matches, or
/// when multiple commands still match ambiguously (e.g. bare "/").
pub fn suggest(input: &str) -> Option<Suggestion> {
    let body = input.strip_prefix('/')?;
    let ends_with_space = body.is_empty() || body.ends_with(' ');
    let mut words: Vec<&str> = body.split_whitespace().collect();
    let partial = if ends_with_space {
        ""
    } else {
        words.pop().unwrap_or("")
    };

    let mut matches = COMMAND_SPECS
        .iter()
        .filter_map(|spec| suggest_for_spec(spec, &words, partial));

    let first = matches.next()?;
    if matches.next().is_some() {
        None
    } else {
        Some(first)
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
        let mut fps = FpsCounter::new();
        let response = execute("/settings rd 10", &mut streamer, &mut fps);
        assert_eq!(response.text, "Render distance set to 10 chunks.");
        assert!(matches!(response.kind, MessageKind::CommandOk));
        assert_eq!(streamer.load_radius(), 10);
    }

    #[test]
    fn settings_rd_rejects_out_of_range() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings rd 999", &mut streamer, &mut fps);
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
        let mut fps = FpsCounter::new();
        let response = execute("/settings rd banana", &mut streamer, &mut fps);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert_eq!(streamer.load_radius(), 6);
    }

    #[test]
    fn settings_rd_missing_argument_shows_usage() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings rd", &mut streamer, &mut fps);
        assert_eq!(response.text, "Usage: /settings rd <chunks>");
    }

    #[test]
    fn unknown_command_is_reported() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/foo", &mut streamer, &mut fps);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert!(response.text.contains("/foo"));
    }

    #[test]
    fn leading_slash_is_optional() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("settings rd 4", &mut streamer, &mut fps);
        assert_eq!(streamer.load_radius(), 4);
        let _ = response;
    }

    #[test]
    fn maxfps_true_caps_at_default() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings maxfps true", &mut streamer, &mut fps);
        assert_eq!(response.text, "FPS capped at 60.");
        assert_eq!(fps.cap, Some(DEFAULT_CAPPED_FPS));
    }

    #[test]
    fn maxfps_false_uncaps() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        fps.cap = Some(30);
        let response = execute("/settings maxfps false", &mut streamer, &mut fps);
        assert_eq!(response.text, "FPS uncapped.");
        assert_eq!(fps.cap, None);
    }

    #[test]
    fn maxfps_number_sets_exact_cap() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings maxfps 144", &mut streamer, &mut fps);
        assert_eq!(response.text, "FPS capped at 144.");
        assert_eq!(fps.cap, Some(144));
    }

    #[test]
    fn maxfps_rejects_out_of_range_number() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings maxfps 5000", &mut streamer, &mut fps);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert_eq!(fps.cap, None, "invalid input must not change state");
    }

    #[test]
    fn maxfps_rejects_garbage() {
        let mut streamer = streamer();
        let mut fps = FpsCounter::new();
        let response = execute("/settings maxfps banana", &mut streamer, &mut fps);
        assert!(matches!(response.kind, MessageKind::CommandError));
        assert_eq!(fps.cap, None);
    }

    #[test]
    fn bare_slash_is_ambiguous_between_help_and_settings() {
        assert!(suggest("/").is_none());
    }

    #[test]
    fn settings_prefix_is_ambiguous_between_rd_and_maxfps() {
        assert!(suggest("/settings").is_none());
        assert!(suggest("/settings ").is_none());
    }

    #[test]
    fn settings_r_disambiguates_to_rd() {
        let s = suggest("/settings r").unwrap();
        assert_eq!(s.ghost_tail, "d <chunks>");
    }

    #[test]
    fn settings_m_disambiguates_to_maxfps() {
        let s = suggest("/settings m").unwrap();
        assert_eq!(s.ghost_tail, "axfps <true|false|number>");
        assert_eq!(s.tab_insert.as_deref(), Some("axfps "));
    }

    #[test]
    fn partial_token_suggests_rest_of_that_word() {
        let s = suggest("/h").unwrap();
        assert_eq!(s.ghost_tail, "elp");
        assert_eq!(s.tab_insert.as_deref(), Some("elp "));
    }

    #[test]
    fn trailing_space_after_full_command_suggests_arg_hint_only() {
        let s = suggest("/settings rd ").unwrap();
        assert_eq!(s.ghost_tail, "<chunks>");
        assert_eq!(
            s.tab_insert, None,
            "a placeholder hint isn't real text to insert"
        );
    }

    #[test]
    fn mid_argument_has_no_suggestion() {
        assert!(suggest("/settings rd 8").is_none());
    }

    #[test]
    fn unmatched_prefix_has_no_suggestion() {
        assert!(suggest("/xyz").is_none());
    }

    #[test]
    fn non_command_text_has_no_suggestion() {
        assert!(suggest("hello").is_none());
    }
}
