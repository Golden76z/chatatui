use chatatui::{
    config::{Config, project_dirs},
    runtime::Runtime,
    storage::worker::Location,
    terminal,
};
use color_eyre::Result;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let config = Config::load_or_create()?;
    // Load syntax definitions in the background; the first code block would stall otherwise.
    tokio::task::spawn_blocking(chatatui::markdown::warm_up);

    let database = Location::File(project_dirs()?.data_dir().join("chatatui.db"));

    let tty = terminal::init(config.mouse_capture)?;
    let result = match Runtime::new(config, tty.keyboard_enhanced, database) {
        Ok(runtime) => runtime.run(tty.terminal).await,
        Err(error) => Err(error),
    };
    terminal::restore();
    result
}
