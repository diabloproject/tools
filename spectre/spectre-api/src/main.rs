use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use replay::Replay;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

type AppState = Arc<Mutex<HashMap<String, (u64, Replay)>>>;

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(Mutex::new(HashMap::new()));

    let app = Router::new()
        .route("/push", post(push_handler))
        .route("/append/{id}", post(append_handler))
        .route("/display/{id}", get(display_handler))
        .route("/history", get(history_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn push_handler(State(state): State<AppState>, body: String) -> Result<String, StatusCode> {
    let id = generate_id();
    let stored_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let replay = Replay::from_yson(&body).unwrap();

    state
        .lock()
        .unwrap()
        .insert(id.clone(), (stored_at, replay));

    Ok(id)
}

async fn append_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: String,
) -> Result<String, StatusCode> {
    let new_replay = Replay::from_yson(&body).map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut state = state.lock().unwrap();
    if let Some((_, replay)) = state.get_mut(&id) {
        replay.rows.extend(new_replay.rows);
        Ok(id)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn display_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<String, StatusCode> {
    let state = state.lock().unwrap();

    match state.get(&id) {
        Some(replay) => {
            let serializer = dsv::RowSerializer::new(b'|');
            let mut rows = String::new();
            for row in &replay.1.rows {
                let result = serializer.serialize(vec![vec![
                    row.timestamp.to_string().as_bytes(),
                    row.log.as_bytes(),
                ]]);
                let row_string = String::from_utf8(result).unwrap_or_default();
                rows.push_str(&row_string);
            }
            Ok(rows)
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn history_handler(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().unwrap();

    let mut output = String::new();
    for (id, replay) in state.iter() {
        let length = replay.1.rows.len();
        let serializer = dsv::RowSerializer::new(b'|');
        output.extend(String::from_utf8(serializer.serialize(vec![vec![
            replay.0.to_string().as_bytes(),
            id.as_bytes(),
            length.to_string().as_bytes(),
        ]])));
    }

    output
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    format!("{:x}", nanos % 0xFFFFFFFF)
}
