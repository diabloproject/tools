use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

const PASTE_API_URL: &str = "https://paste.diabloproject.space/api/paste";

#[derive(Serialize)]
struct PasteRequest {
    text: String,
}

#[derive(Deserialize)]
struct PasteResponse {
    code: String,
}

fn main() -> Result<()> {
    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .context("Failed to read from stdin")?;

    if text.is_empty() {
        anyhow::bail!("No input provided");
    }

    let client = reqwest::blocking::Client::new();
    let request = PasteRequest { text };

    let response = client
        .post(PASTE_API_URL)
        .json(&request)
        .send()
        .context("Failed to send request to paste API")?;

    if !response.status().is_success() {
        anyhow::bail!("API request failed with status: {}", response.status());
    }

    let paste_response: PasteResponse = response
        .json()
        .context("Failed to parse API response")?;

    println!("https://paste.diabloproject.space/{}", paste_response.code);

    Ok(())
}
