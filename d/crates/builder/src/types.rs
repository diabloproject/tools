use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct Spec {
    pub(crate) name: String,
    pub(crate) artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone)]
pub(crate) struct Artifact {
    pub(crate) name: String,
    pub(crate) r#type: ArtifactType,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ArtifactType {
    Executable,
    Library,
    Resource,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactPath {
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactInstance {
    pub(crate) path: PathBuf,
    pub(crate) artifact: Artifact,
}
