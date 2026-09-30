//! Vertical scrolling of the conversation with "follow the bottom" behaviour.
//!
//! While `follow` is set the view sticks to the last line, so a streamed reply stays in
//! view. Scrolling up clears it and pins the view to an absolute line; reaching the bottom
//! again sets it back.

/// Scroll position of the conversation view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScrollState {
    /// Index of the first visible line when not following.
    offset: usize,
    /// Stick to the bottom.
    follow: bool,
}

impl Default for ScrollState {
    fn default() -> Self {
        Self {
            offset: 0,
            follow: true,
        }
    }
}

impl ScrollState {
    /// First visible line for `total` lines shown in a view of `height` rows.
    pub fn offset(&self, total: usize, height: usize) -> usize {
        let max = total.saturating_sub(height);
        if self.follow {
            max
        } else {
            self.offset.min(max)
        }
    }

    /// `true` when the view sticks to the bottom.
    pub fn is_following(&self) -> bool {
        self.follow
    }

    /// Scrolls towards the start by `lines`.
    pub fn scroll_up(&mut self, lines: usize, total: usize, height: usize) {
        if total <= height {
            return; // everything is visible
        }
        self.offset = self.offset(total, height).saturating_sub(lines);
        self.follow = false;
    }

    /// Scrolls towards the end by `lines`; following resumes at the bottom.
    pub fn scroll_down(&mut self, lines: usize, total: usize, height: usize) {
        let max = total.saturating_sub(height);
        let offset = self.offset(total, height).saturating_add(lines);
        if offset >= max {
            self.to_bottom();
        } else {
            self.offset = offset;
        }
    }

    /// Scrolls just enough to show `line`, with some context above it.
    pub fn reveal(&mut self, line: usize, total: usize, height: usize) {
        let offset = self.offset(total, height);
        if line >= offset && line < offset + height {
            return;
        }
        let max = total.saturating_sub(height);
        let target = line.saturating_sub(height / 3).min(max);
        if target >= max {
            self.to_bottom();
        } else {
            self.offset = target;
            self.follow = false;
        }
    }

    /// Jumps to the first line.
    pub fn to_top(&mut self, total: usize, height: usize) {
        if total > height {
            self.offset = 0;
            self.follow = false;
        }
    }

    /// Jumps to the bottom and follows new content.
    pub fn to_bottom(&mut self) {
        self.follow = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_growing_content() {
        let scroll = ScrollState::default();
        assert_eq!(scroll.offset(5, 10), 0);
        assert_eq!(scroll.offset(30, 10), 20);
        assert_eq!(scroll.offset(31, 10), 21);
    }

    #[test]
    fn scrolling_up_pins_the_view_while_content_grows() {
        let mut scroll = ScrollState::default();
        scroll.scroll_up(3, 30, 10);
        assert!(!scroll.is_following());
        assert_eq!(scroll.offset(30, 10), 17);
        assert_eq!(scroll.offset(50, 10), 17, "new lines do not move the view");
    }

    #[test]
    fn returning_to_the_bottom_resumes_following() {
        let mut scroll = ScrollState::default();
        scroll.scroll_up(5, 30, 10);
        scroll.scroll_down(2, 30, 10);
        assert!(!scroll.is_following());
        scroll.scroll_down(10, 30, 10);
        assert!(scroll.is_following());
        assert_eq!(scroll.offset(40, 10), 30);
    }

    #[test]
    fn nothing_to_scroll_when_everything_fits() {
        let mut scroll = ScrollState::default();
        scroll.scroll_up(3, 8, 10);
        assert!(scroll.is_following());
        assert_eq!(scroll.offset(8, 10), 0);
    }

    #[test]
    fn scroll_up_saturates_at_the_top() {
        let mut scroll = ScrollState::default();
        scroll.scroll_up(100, 30, 10);
        assert_eq!(scroll.offset(30, 10), 0);
        scroll.to_bottom();
        scroll.to_top(30, 10);
        assert_eq!(scroll.offset(30, 10), 0);
    }

    #[test]
    fn offset_is_clamped_when_content_shrinks() {
        let mut scroll = ScrollState::default();
        scroll.scroll_up(1, 30, 10);
        assert_eq!(scroll.offset(15, 10), 5);
    }
}
