//! Reading files to attach (`/add`) and completing their paths. Runs in the runtime,
//! off the UI loop (these functions block).

use std::{
    fs,
    path::{Path, PathBuf},
};

/// Largest file that can be attached.
pub const MAX_ATTACHMENT_BYTES: u64 = 256 * 1024;

/// Largest image that can be attached (the limit of most vision APIs).
pub const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;

/// A file read for attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    /// The path as the user typed it (shown in the conversation).
    pub source: String,
    /// Text of a text file (empty for an image).
    pub content: String,
    /// An image file.
    pub image: Option<crate::state::Image>,
}

/// Media type of an image file, from its extension.
pub fn image_type(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_string_lossy().to_lowercase();
    Some(match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    })
}

/// Reads an image to attach (at most [`MAX_IMAGE_BYTES`]).
fn read_image(source: &str, path: &Path, media_type: &str) -> Result<Attachment, String> {
    use base64::Engine;
    let metadata = fs::metadata(path).map_err(|e| format!("{source} : {}", io_message(&e)))?;
    if metadata.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "{source} est trop grosse ({} Ko, maximum {} Mo)",
            metadata.len() / 1024,
            MAX_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("{source} : {}", io_message(&e)))?;
    Ok(Attachment {
        source: source.to_owned(),
        content: String::new(),
        image: Some(crate::state::Image {
            media_type: media_type.to_owned(),
            base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        }),
    })
}

/// Whether `c` separates path components here: `/`, and also `\\` on Windows.
pub fn is_separator(c: char) -> bool {
    c == '/' || (cfg!(windows) && c == '\\')
}

/// Expands a leading `~` to the home directory (`~/…`, or `~\\…` on Windows).
pub fn expand_home(path: &str) -> PathBuf {
    let rest = path.strip_prefix('~');
    if let Some(rest) = rest
        && (rest.is_empty() || rest.starts_with(is_separator))
        && let Some(home) = directories::BaseDirs::new().map(|d| d.home_dir().to_path_buf())
    {
        return home.join(rest.trim_start_matches(is_separator));
    }
    PathBuf::from(path)
}

/// Reads a UTF-8 text file of at most [`MAX_ATTACHMENT_BYTES`]. Errors are user-facing.
pub fn read_attachment(source: &str) -> Result<Attachment, String> {
    let source = source.trim();
    let path = expand_home(source);
    if let Some(media_type) = image_type(&path) {
        return read_image(source, &path, media_type);
    }
    let metadata = fs::metadata(&path).map_err(|e| format!("{source} : {}", io_message(&e)))?;
    if metadata.is_dir() {
        return Err(format!("{source} est un dossier (joignez un fichier)"));
    }
    if metadata.len() > MAX_ATTACHMENT_BYTES {
        return Err(format!(
            "{source} est trop gros ({} Ko, maximum {} Ko)",
            metadata.len() / 1024,
            MAX_ATTACHMENT_BYTES / 1024
        ));
    }
    let bytes = fs::read(&path).map_err(|e| format!("{source} : {}", io_message(&e)))?;
    if bytes.contains(&0) {
        return Err(format!("{source} n'est pas un fichier texte"));
    }
    let content =
        String::from_utf8(bytes).map_err(|_| format!("{source} n'est pas du texte UTF-8"))?;
    Ok(Attachment {
        source: source.to_owned(),
        content,
        image: None,
    })
}

fn io_message(error: &std::io::Error) -> String {
    match error.kind() {
        std::io::ErrorKind::NotFound => "fichier introuvable".into(),
        std::io::ErrorKind::PermissionDenied => "accès refusé".into(),
        _ => error.to_string(),
    }
}

/// Completions of a partial path: entries of its directory starting with its last
/// component, sorted, directories suffixed with the last separator typed (`/` by
/// default, `\\` possible on Windows). Hidden files are listed only when the partial name
/// starts with a dot.
pub fn complete_path(partial: &str) -> Vec<String> {
    let (dir_part, name_part) = match partial.rfind(is_separator) {
        Some(slash) => (&partial[..=slash], &partial[slash + 1..]),
        None => ("", partial),
    };
    // Directories end with the separator the partial path last used.
    let separator = dir_part.chars().last().unwrap_or('/');
    let dir = if dir_part.is_empty() {
        PathBuf::from(".")
    } else {
        expand_home(dir_part)
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut candidates: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with(name_part)
                || (name.starts_with('.') && !name_part.starts_with('.'))
            {
                return None;
            }
            let is_dir = entry.file_type().is_ok_and(|t| t.is_dir())
                || fs::metadata(dir.join(&name)).is_ok_and(|m| m.is_dir());
            let mut candidate = format!("{dir_part}{name}");
            if is_dir {
                candidate.push(separator);
            }
            Some(candidate)
        })
        .collect();
    candidates.sort();
    candidates
}

/// Longest common prefix of `candidates` (by characters).
pub fn common_prefix(candidates: &[String]) -> String {
    let Some(first) = candidates.first() else {
        return String::new();
    };
    let mut prefix: Vec<char> = first.chars().collect();
    for candidate in &candidates[1..] {
        let common = prefix
            .iter()
            .zip(candidate.chars())
            .take_while(|(a, b)| **a == *b)
            .count();
        prefix.truncate(common);
    }
    prefix.into_iter().collect()
}

/// Writes `content` to a file that does not exist yet: `path` (`~` allowed), or
/// `suggested` in the current directory; `-2`, `-3`… are added before the extension
/// when the name is taken. Returns the path written (user-facing errors).
pub fn write_new(path: Option<&str>, suggested: &str, content: &str) -> Result<String, String> {
    let wanted = match path {
        Some(path) => expand_home(path),
        None => PathBuf::from(suggested),
    };
    let wanted = if wanted.extension().is_none() {
        wanted.with_extension("md")
    } else {
        wanted
    };
    let stem = wanted
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "conversation".into());
    let extension = wanted
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "md".into());
    for n in 1..1000 {
        let candidate = if n == 1 {
            wanted.clone()
        } else {
            wanted.with_file_name(format!("{stem}-{n}.{extension}"))
        };
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                use std::io::Write;
                file.write_all(content.as_bytes())
                    .map_err(|e| format!("{} : {e}", candidate.display()))?;
                return Ok(candidate.display().to_string());
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("{} : {e}", candidate.display())),
        }
    }
    Err(format!(
        "{} : trop de fichiers du même nom",
        wanted.display()
    ))
}

/// File name of a path, for short messages.
pub fn file_name(source: &str) -> String {
    Path::new(source)
        .file_name()
        .map_or_else(|| source.to_owned(), |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("notes.md"), "# Notes\nbonjour").expect("write");
        fs::write(dir.path().join("notes-old.md"), "ancien").expect("write");
        fs::write(dir.path().join(".cache"), "x").expect("write");
        fs::write(dir.path().join("image.bin"), [0u8, 159, 146, 150]).expect("write");
        fs::create_dir(dir.path().join("sub")).expect("mkdir");
        dir
    }

    fn path(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn reads_text_files() {
        let dir = dir();
        let source = path(&dir, "notes.md");
        let attachment = read_attachment(&source).expect("readable");
        assert_eq!(attachment.content, "# Notes\nbonjour");
        assert_eq!(attachment.source, source);
    }

    #[test]
    fn explains_why_a_file_cannot_be_attached() {
        let dir = dir();
        let missing = path(&dir, "nope.md");
        assert_eq!(
            read_attachment(&missing),
            Err(format!("{missing} : fichier introuvable"))
        );
        let binary = path(&dir, "image.bin");
        assert_eq!(
            read_attachment(&binary),
            Err(format!("{binary} n'est pas un fichier texte"))
        );
        let sub = path(&dir, "sub");
        assert!(read_attachment(&sub).is_err_and(|e| e.contains("dossier")));

        let big = path(&dir, "big.txt");
        fs::write(&big, "x".repeat(300 * 1024)).expect("write");
        assert!(read_attachment(&big).is_err_and(|e| e.contains("trop gros")));
    }

    #[test]
    fn completes_paths() {
        let dir = dir();
        let base = format!("{}/", dir.path().to_string_lossy());
        assert_eq!(
            complete_path(&format!("{base}no")),
            vec![format!("{base}notes-old.md"), format!("{base}notes.md")]
        );
        assert_eq!(
            complete_path(&format!("{base}s")),
            vec![format!("{base}sub/")]
        );
        assert!(!complete_path(&base).iter().any(|c| c.ends_with(".cache")));
        assert_eq!(
            complete_path(&format!("{base}.c")),
            vec![format!("{base}.cache")]
        );
        assert!(complete_path("/definitely/not/here/x").is_empty());
    }

    #[test]
    fn common_prefixes() {
        let candidates = vec!["notes-old.md".to_owned(), "notes.md".to_owned()];
        assert_eq!(common_prefix(&candidates), "notes");
        assert_eq!(common_prefix(&[]), "");
        assert_eq!(file_name("~/docs/plan.md"), "plan.md");
    }

    #[test]
    fn home_is_expanded() {
        assert!(!expand_home("~/x").starts_with("~"));
        assert!(!expand_home("~").starts_with("~"));
        assert_eq!(expand_home("rel/x"), PathBuf::from("rel/x"));
        assert_eq!(expand_home("~x"), PathBuf::from("~x"), "another user");
        if cfg!(windows) {
            assert!(!expand_home("~\\x").starts_with("~"));
        }
    }

    #[cfg(windows)]
    #[test]
    fn completes_windows_paths() {
        let dir = dir();
        let base = format!("{}\\", dir.path().display());
        assert_eq!(
            complete_path(&format!("{base}s")),
            vec![format!("{base}sub\\")]
        );
        assert_eq!(
            complete_path(&format!("{base}notes.")),
            vec![format!("{base}notes.md")]
        );
    }

    #[test]
    fn write_new_never_overwrites() {
        let dir = tempfile::tempdir().expect("temp dir");
        let target = dir.path().join("notes");
        let target = target.to_string_lossy().into_owned();
        let first = write_new(Some(&target), "x.md", "un").expect("write");
        assert!(first.ends_with("notes.md"), "{first}");
        let second = write_new(Some(&target), "x.md", "deux").expect("write");
        assert!(second.ends_with("notes-2.md"), "{second}");
        assert_eq!(std::fs::read_to_string(&first).expect("read"), "un");
        assert!(write_new(Some("/definitely/not/here/x.md"), "x.md", "").is_err());
    }

    #[test]
    fn images_are_read_as_base64() {
        let dir = tempfile::tempdir().expect("temp dir");
        let png = dir.path().join("schéma.PNG");
        std::fs::write(&png, [0x89, b'P', b'N', b'G']).expect("write");
        let attachment = read_attachment(&png.display().to_string()).expect("image");
        let image = attachment.image.expect("an image");
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.base64, "iVBORw==");
        assert!(attachment.content.is_empty());
        assert_eq!(image_type(Path::new("a.jpeg")), Some("image/jpeg"));
        assert_eq!(image_type(Path::new("a.svg")), None, "not for vision APIs");
    }
}
