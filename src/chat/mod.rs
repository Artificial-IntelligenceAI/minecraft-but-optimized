pub mod commands;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A chat line stays fully visible for this long after being sent, then fades
/// out over `FADE_DURATION`. Vanilla Minecraft doesn't publish exact tick
/// values for this anywhere easy to verify, so these are a reasonable
/// approximation of the ~10-second lifetime players are used to, not a
/// byte-for-byte port of the original constants.
const VISIBLE_DURATION: Duration = Duration::from_secs(8);
const FADE_DURATION: Duration = Duration::from_secs(2);

/// Recent messages shown in the passive (chat closed) HUD.
const PASSIVE_LINE_COUNT: usize = 10;
/// Messages shown at once when chat is open and scrolled to the bottom.
const OPEN_LINE_COUNT: usize = 20;
/// Total messages retained, matching vanilla's stored history size.
const MAX_LOG_LEN: usize = 100;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    /// A normal chat line (including the player's own sent messages).
    Chat,
    /// Echo of a command the player ran, shown dimmed.
    CommandEcho,
    /// Successful command feedback.
    CommandOk,
    /// Command usage/validation error.
    CommandError,
}

pub struct ChatMessage {
    pub text: String,
    pub kind: MessageKind,
    created_at: Instant,
}

pub struct Chat {
    log: VecDeque<ChatMessage>,
    pub input: String,
    pub is_open: bool,
    sent_history: Vec<String>,
    /// `None` means "not currently recalling"; `Some(i)` indexes `sent_history`
    /// from the end, where 0 is the most recently sent message.
    history_cursor: Option<usize>,
    /// Lines scrolled back from the bottom of the log while chat is open.
    scroll: usize,
}

impl Chat {
    pub fn new() -> Self {
        Self {
            log: VecDeque::new(),
            input: String::new(),
            is_open: false,
            sent_history: Vec::new(),
            history_cursor: None,
            scroll: 0,
        }
    }

    pub fn push_message(&mut self, text: impl Into<String>, kind: MessageKind) {
        if self.log.len() >= MAX_LOG_LEN {
            self.log.pop_front();
        }
        self.log.push_back(ChatMessage {
            text: text.into(),
            kind,
            created_at: Instant::now(),
        });
        // A brand new message should always be visible, so snap back to the bottom.
        self.scroll = 0;
    }

    pub fn open(&mut self, prefill: &str) {
        self.is_open = true;
        self.input.clear();
        self.input.push_str(prefill);
        self.history_cursor = None;
        self.scroll = 0;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.input.clear();
        self.history_cursor = None;
        self.scroll = 0;
    }

    pub fn push_char(&mut self, c: char) {
        self.input.push(c);
    }

    pub fn backspace(&mut self) {
        self.input.pop();
    }

    /// Appends a Tab-completion's literal text (see [`commands::Suggestion::tab_insert`]).
    pub fn apply_completion(&mut self, text: &str) {
        self.input.push_str(text);
    }

    /// Recalls older sent messages into the input, most-recent-first.
    pub fn history_prev(&mut self) {
        if self.sent_history.is_empty() {
            return;
        }
        let next = match self.history_cursor {
            None => 0,
            Some(i) => (i + 1).min(self.sent_history.len() - 1),
        };
        self.history_cursor = Some(next);
        self.input = self.sent_history[self.sent_history.len() - 1 - next].clone();
    }

    pub fn history_next(&mut self) {
        match self.history_cursor {
            None => {}
            Some(0) => {
                self.history_cursor = None;
                self.input.clear();
            }
            Some(i) => {
                let next = i - 1;
                self.history_cursor = Some(next);
                self.input = self.sent_history[self.sent_history.len() - 1 - next].clone();
            }
        }
    }

    pub fn scroll_up(&mut self) {
        if self.scroll + 1 < self.log.len() {
            self.scroll += 1;
        }
    }

    pub fn scroll_down(&mut self) {
        self.scroll = self.scroll.saturating_sub(1);
    }

    /// Clears the input and returns the submitted line (`None` if it was
    /// blank), recording non-blank submissions into recall history. Always
    /// closes the chat, matching vanilla (even submitting blank input closes it).
    pub fn submit(&mut self) -> Option<String> {
        let line = std::mem::take(&mut self.input);
        self.close();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        self.sent_history.push(trimmed.to_string());
        Some(trimmed.to_string())
    }

    /// Lines to draw this frame, oldest first, with an opacity multiplier in
    /// `0.0..=1.0` already applied for fade-out. When chat is open this shows
    /// a larger scrollable window at full opacity; when closed it shows only
    /// recent, not-yet-faded messages.
    pub fn visible_lines(&self) -> Vec<(&ChatMessage, f32)> {
        if self.is_open {
            let end = self.log.len().saturating_sub(self.scroll);
            let start = end.saturating_sub(OPEN_LINE_COUNT);
            return self.log.range(start..end).map(|m| (m, 1.0)).collect();
        }

        let now = Instant::now();
        let mut lines: Vec<(&ChatMessage, f32)> = self
            .log
            .iter()
            .rev()
            .map_while(|m| fade_alpha(now, m.created_at).map(|alpha| (m, alpha)))
            .take(PASSIVE_LINE_COUNT)
            .collect();
        lines.reverse();
        lines
    }
}

impl Default for Chat {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `None` once a message has fully faded (so callers can stop early
/// via `map_while`, since messages are visited newest-first and fade
/// monotonically with age).
fn fade_alpha(now: Instant, created_at: Instant) -> Option<f32> {
    let age = now.duration_since(created_at);
    if age <= VISIBLE_DURATION {
        Some(1.0)
    } else if age <= VISIBLE_DURATION + FADE_DURATION {
        let into_fade = (age - VISIBLE_DURATION).as_secs_f32();
        Some(1.0 - into_fade / FADE_DURATION.as_secs_f32())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_prefills_input_and_close_clears_it() {
        let mut chat = Chat::new();
        chat.open("/");
        assert!(chat.is_open);
        assert_eq!(chat.input, "/");

        chat.close();
        assert!(!chat.is_open);
        assert!(chat.input.is_empty());
    }

    #[test]
    fn submit_returns_trimmed_line_and_closes() {
        let mut chat = Chat::new();
        chat.open("");
        chat.push_char('h');
        chat.push_char('i');

        let submitted = chat.submit();
        assert_eq!(submitted.as_deref(), Some("hi"));
        assert!(!chat.is_open);
    }

    #[test]
    fn submitting_blank_input_closes_without_a_message() {
        let mut chat = Chat::new();
        chat.open("");
        chat.push_char(' ');

        assert_eq!(chat.submit(), None);
        assert!(!chat.is_open);
    }

    #[test]
    fn backspace_removes_last_character() {
        let mut chat = Chat::new();
        chat.open("ab");
        chat.backspace();
        assert_eq!(chat.input, "a");
    }

    #[test]
    fn history_prev_and_next_cycle_through_sent_messages() {
        let mut chat = Chat::new();
        chat.open("");
        chat.push_char('a');
        chat.submit();
        chat.open("");
        chat.push_char('b');
        chat.submit();

        chat.open("");
        chat.history_prev();
        assert_eq!(chat.input, "b");
        chat.history_prev();
        assert_eq!(chat.input, "a");
        chat.history_prev(); // no more history, stays on oldest
        assert_eq!(chat.input, "a");

        chat.history_next();
        assert_eq!(chat.input, "b");
        chat.history_next();
        assert_eq!(chat.input, "");
    }

    #[test]
    fn fade_alpha_is_full_then_ramps_down_then_gone() {
        let now = Instant::now();
        assert_eq!(fade_alpha(now, now), Some(1.0));
        assert_eq!(fade_alpha(now + VISIBLE_DURATION, now), Some(1.0));

        let mid_fade = fade_alpha(now + VISIBLE_DURATION + FADE_DURATION / 2, now).unwrap();
        assert!((mid_fade - 0.5).abs() < 0.01);

        assert_eq!(
            fade_alpha(
                now + VISIBLE_DURATION + FADE_DURATION + Duration::from_millis(1),
                now
            ),
            None
        );
    }

    #[test]
    fn recent_messages_stay_capped_at_max_log_len() {
        let mut chat = Chat::new();
        for i in 0..(MAX_LOG_LEN + 10) {
            chat.push_message(format!("msg {i}"), MessageKind::Chat);
        }
        assert_eq!(chat.log.len(), MAX_LOG_LEN);
        assert_eq!(chat.log.front().unwrap().text, format!("msg {}", 10));
    }

    #[test]
    fn open_chat_shows_recent_messages_at_full_opacity() {
        let mut chat = Chat::new();
        chat.push_message("hello", MessageKind::Chat);
        chat.open("");

        let lines = chat.visible_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].0.text, "hello");
        assert_eq!(lines[0].1, 1.0);
    }
}
