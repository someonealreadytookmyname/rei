use std::sync::Mutex;
use tauri::{Manager, State};

/// Holds the port the Python backend is running on and the child process handle.
struct BackendState {
    port: Mutex<u16>,
    child: Mutex<Option<std::process::Child>>,
}

/// Tauri command: get the backend port so the frontend knows where to connect.
#[tauri::command]
fn get_backend_port(state: State<BackendState>) -> u16 {
    *state.port.lock().unwrap()
}

/// Tauri command: check if Ollama is reachable at localhost:11434.
#[tauri::command]
async fn check_ollama_status() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();

    // Check if Ollama is running
    let running = match client.get("http://127.0.0.1:11434/api/tags").send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(body) => Some(body),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
        Err(_) => None,
    };

    match running {
        Some(body) => {
            let models = body
                .get("models")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            Ok(serde_json::json!({
                "running": true,
                "models": models
            }))
        }
        None => Ok(serde_json::json!({
            "running": false,
            "models": []
        })),
    }
}

/// Tauri command: attempt to start Ollama serve in the background.
#[tauri::command]
async fn start_ollama() -> Result<String, String> {
    use std::process::Command;

    // Check if ollama binary exists
    let check = Command::new("ollama").arg("--version").output();

    match check {
        Ok(output) => {
            if !output.status.success() {
                return Err("Ollama binary found but returned an error.".to_string());
            }
        }
        Err(_) => {
            return Err(
                "Ollama is not installed. Please download it from https://ollama.com/download"
                    .to_string(),
            );
        }
    }

    // Start ollama serve in the background
    let result = Command::new("ollama").arg("serve").spawn();

    match result {
        Ok(_) => {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            Ok("Ollama started successfully.".to_string())
        }
        Err(e) => Err(format!("Failed to start Ollama: {}", e)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = if cfg!(debug_assertions) {
        8000
    } else {
        portpicker::pick_unused_port().unwrap_or(8000)
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .manage(BackendState {
            port: Mutex::new(port),
            child: Mutex::new(None),
        })
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let handle = app.handle().clone();
            let port_num = port;

            // In production release builds, spawn Python backend as a managed child process
            if !cfg!(debug_assertions) {
                let state = app.state::<BackendState>();
                let python_cmd = if cfg!(windows) { "python" } else { "python3" };
                match std::process::Command::new(python_cmd)
                    .args([
                        "-m",
                        "uvicorn",
                        "backend.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        &port_num.to_string(),
                    ])
                    .spawn()
                {
                    Ok(child) => {
                        *state.child.lock().unwrap() = Some(child);
                        log::info!("Spawned Python backend child process on port {}", port_num);
                    }
                    Err(e) => {
                        log::error!("Failed to spawn Python backend: {}", e);
                    }
                }
            }

            // Wait for backend to be ready, then navigate the main window
            tauri::async_runtime::spawn(async move {
                let client = reqwest::Client::new();
                let check_url = format!("http://127.0.0.1:{}/api/info", port_num);

                for i in 0..60 {
                    if let Ok(resp) = client.get(&check_url).send().await {
                        if resp.status().is_success() {
                            log::info!("Backend ready on port {} (attempt {})", port_num, i + 1);
                            if let Some(window) = handle.get_webview_window("main") {
                                let target_url = format!("http://127.0.0.1:{}", port_num);
                                if let Ok(url) = target_url.parse() {
                                    let _ = window.navigate(url);
                                }
                            }
                            return;
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                log::error!("Python backend did not respond within 30 seconds");
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_backend_port,
            check_ollama_status,
            start_ollama,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state = app_handle.state::<BackendState>();
                let mut child_opt = None;
                if let Ok(mut lock) = state.child.lock() {
                    child_opt = lock.take();
                };
                if let Some(mut child) = child_opt {
                    let _ = child.kill();
                }
            }
        });
}
