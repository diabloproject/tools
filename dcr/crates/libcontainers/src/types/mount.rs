use crate::types::id_map::IDMap;

pub struct MountIDMapping {
    /// `recursive` indicates if the mapping needs to be recursive.
    recursive: bool,
    /// `user_nspath` is a path to a user namespace that indicates the necessary
    /// id-mappings for MOUNT_ATTR_IDMAP. If set to non-"", UIDMappings and
    /// GIDMappings must be set to nil.
    user_nspath: String,
    /// `uid_mapping` is the uid mapping set for this mount, to be used with
    /// MOUNT_ATTR_IDMAP.
    uid_mappings: Vec<IDMap>,
    /// `gid_mapping` is the gid mapping set for this mount, to be used with
    /// MOUNT_ATTR_IDMAP.
    gid_mappings: Vec<IDMap>,
}

pub struct Mount<'a> {
    /// Source path for the mount.
    source: String,
    /// Destination path for the mount inside the container.
    destination: String,
    /// Device the mount is for.
    device: String,
    /// Mount flags.
    flags: usize,
    /// Mount flags that were explicitly cleared in the configuration (meaning
    /// the user explicitly requested that these flags *not* be set).
    cleared_flags: usize,
    /// Propagation flags.
    propagated_flags: Vec<usize>,
    /// Mount data applied to the mount.
    data: String,
    /// Relabel source if set, "z" indicates shared, "Z" indicates unshared.
    relabel: String,
    /// RecAttr represents mount properties to be applied recursively (AT_RECURSIVE), see mount_setattr(2).
    rec_attr: usize, // todo!("Type this")
    /// Extensions are additional flags that are specific to runc.
    extensions: isize,
    /// Mapping is the MOUNT_ATTR_IDMAP configuration for the mount. If non-nil,
    /// the mount is configured to use MOUNT_ATTR_IDMAP-style id mappings.
    id_mapping: &'a MountIDMapping,
}
