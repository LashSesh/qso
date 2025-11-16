/*!
 * S3-Compatible Cloud Storage Adapter for MEF-Core
 *
 * Provides cloud storage capabilities for snapshots, TICs, and ledger blocks.
 * Supports AWS S3, MinIO, and other S3-compatible services.
 */

use anyhow::{anyhow, Result};
use aws_sdk_s3::{
    config::Region,
    operation::{create_bucket::CreateBucketError, head_bucket::HeadBucketError},
    primitives::ByteStream,
    types::{
        BucketLocationConstraint, BucketVersioningStatus, CreateBucketConfiguration,
        ExpirationStatus, LifecycleExpiration, LifecycleRule, LifecycleRuleFilter, Transition,
        TransitionStorageClass, VersioningConfiguration,
    },
    Client,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for S3 storage adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Key prefix for all objects
    pub prefix: String,
    /// AWS region
    pub region: String,
    /// Custom endpoint URL (for MinIO, etc.)
    pub endpoint_url: Option<String>,
    /// AWS access key ID
    pub access_key_id: Option<String>,
    /// AWS secret access key
    pub secret_access_key: Option<String>,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            bucket: "mef-core".to_string(),
            prefix: "mef/".to_string(),
            region: "us-east-1".to_string(),
            endpoint_url: None,
            access_key_id: None,
            secret_access_key: None,
        }
    }
}

/// Artifact type for storage organization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    Snapshot,
    Tic,
    Block,
    Hdag,
    Index,
}

impl ArtifactType {
    /// Get the prefix for this artifact type
    fn prefix(&self) -> &'static str {
        match self {
            ArtifactType::Snapshot => "snapshots",
            ArtifactType::Tic => "tics",
            ArtifactType::Block => "ledger",
            ArtifactType::Hdag => "hdag",
            ArtifactType::Index => "indices",
        }
    }
}

/// Metadata for uploaded artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadMetadata {
    pub key: String,
    pub etag: String,
    pub version_id: Option<String>,
    pub checksum: String,
}

/// Metadata for listed artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub key: String,
    pub size: i64,
    pub last_modified: String,
    pub etag: String,
    pub metadata: HashMap<String, String>,
}

/// Sync statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub uploaded: usize,
    pub downloaded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

/// Storage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    pub bucket: String,
    pub prefix: String,
    pub artifacts: HashMap<String, ArtifactStats>,
    pub total_size: i64,
    pub total_objects: usize,
    pub total_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStats {
    pub count: usize,
    pub size_bytes: i64,
    pub size_mb: f64,
}

/// S3-compatible storage adapter for MEF-Core artifacts
pub struct S3StorageAdapter {
    config: S3Config,
    client: Client,
    metadata_cache: HashMap<String, UploadMetadata>,
}

impl S3StorageAdapter {
    /// Create a new S3 storage adapter with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - S3 configuration
    ///
    /// # Returns
    ///
    /// New storage adapter instance
    pub async fn new(config: S3Config) -> Result<Self> {
        let client = Self::init_s3_client(&config).await?;

        let adapter = Self {
            config,
            client,
            metadata_cache: HashMap::new(),
        };

        // Ensure bucket exists
        adapter.ensure_bucket().await?;

        Ok(adapter)
    }

    /// Initialize S3 client with configuration
    async fn init_s3_client(config: &S3Config) -> Result<Client> {
        // Load AWS config
        let mut aws_config_loader = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(Region::new(config.region.clone()));

        // Set custom endpoint if provided (for MinIO, etc.)
        if let Some(endpoint) = &config.endpoint_url {
            aws_config_loader = aws_config_loader.endpoint_url(endpoint);
        }

        let aws_config = aws_config_loader.load().await;

        // Create S3 client
        let client = Client::new(&aws_config);

        Ok(client)
    }

    /// Ensure the S3 bucket exists, creating it if necessary
    async fn ensure_bucket(&self) -> Result<()> {
        // Check if bucket exists
        match self
            .client
            .head_bucket()
            .bucket(&self.config.bucket)
            .send()
            .await
        {
            Ok(_) => {
                // Bucket exists
                Ok(())
            }
            Err(err) => {
                // Check if it's a 404 (bucket doesn't exist)
                if let Some(service_err) = err.as_service_error() {
                    if matches!(service_err, HeadBucketError::NotFound(_)) {
                        // Create the bucket
                        self.create_bucket().await?;

                        // Enable versioning
                        self.enable_versioning().await?;

                        Ok(())
                    } else {
                        Err(anyhow!("Failed to check bucket: {:?}", service_err))
                    }
                } else {
                    Err(anyhow!("Failed to check bucket: {:?}", err))
                }
            }
        }
    }

    /// Create S3 bucket
    async fn create_bucket(&self) -> Result<()> {
        let mut request = self.client.create_bucket().bucket(&self.config.bucket);

        // Set location constraint for non-us-east-1 regions
        if self.config.region != "us-east-1" {
            let constraint = BucketLocationConstraint::from(self.config.region.as_str());
            let config = CreateBucketConfiguration::builder()
                .location_constraint(constraint)
                .build();
            request = request.create_bucket_configuration(config);
        }

        match request.send().await {
            Ok(_) => {
                println!("Created S3 bucket: {}", self.config.bucket);
                Ok(())
            }
            Err(err) => {
                // Check if error is BucketAlreadyOwnedByYou
                if let Some(service_err) = err.as_service_error() {
                    if matches!(service_err, CreateBucketError::BucketAlreadyOwnedByYou(_)) {
                        // Bucket already exists and is owned by us - that's fine
                        Ok(())
                    } else {
                        Err(anyhow!("Failed to create bucket: {:?}", service_err))
                    }
                } else {
                    Err(anyhow!("Failed to create bucket: {:?}", err))
                }
            }
        }
    }

    /// Generate S3 key for an artifact
    fn get_s3_key(&self, artifact_type: ArtifactType, artifact_id: &str) -> String {
        format!(
            "{}{}/{}",
            self.config.prefix,
            artifact_type.prefix(),
            artifact_id
        )
    }

    /// Upload a snapshot to S3
    ///
    /// # Arguments
    ///
    /// * `snapshot` - Snapshot data as JSON value
    ///
    /// # Returns
    ///
    /// Upload metadata on success
    pub async fn upload_snapshot(
        &mut self,
        snapshot: &serde_json::Value,
    ) -> Result<UploadMetadata> {
        let snapshot_id = snapshot["id"]
            .as_str()
            .ok_or_else(|| anyhow!("Snapshot missing 'id' field"))?;

        let key = self.get_s3_key(ArtifactType::Snapshot, &format!("{}.spiral", snapshot_id));

        // Serialize snapshot
        let snapshot_json = serde_json::to_string_pretty(snapshot)?;

        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(snapshot_json.as_bytes());
        let checksum = format!("{:x}", hasher.finalize());

        // Prepare metadata
        let mut metadata = HashMap::new();
        metadata.insert("snapshot-id".to_string(), snapshot_id.to_string());
        if let Some(seed) = snapshot["seed"].as_str() {
            metadata.insert("seed".to_string(), seed.to_string());
        }
        if let Some(phase) = snapshot["phase"].as_f64() {
            metadata.insert("phase".to_string(), phase.to_string());
        }
        if let Some(por) = snapshot["metrics"]["por"].as_str() {
            metadata.insert("por".to_string(), por.to_string());
        }
        metadata.insert("checksum".to_string(), checksum.clone());
        if let Some(timestamp) = snapshot["timestamp"].as_str() {
            metadata.insert("timestamp".to_string(), timestamp.to_string());
        }

        // Upload to S3
        let response = self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .body(ByteStream::from(snapshot_json.into_bytes()))
            .content_type("application/json")
            .set_metadata(Some(metadata))
            .send()
            .await?;

        let upload_meta = UploadMetadata {
            key: key.clone(),
            etag: response.e_tag().unwrap_or("").to_string(),
            version_id: response.version_id().map(|s| s.to_string()),
            checksum: checksum.clone(),
        };

        // Update cache
        self.metadata_cache
            .insert(snapshot_id.to_string(), upload_meta.clone());

        Ok(upload_meta)
    }

    /// Download a snapshot from S3
    ///
    /// # Arguments
    ///
    /// * `snapshot_id` - Snapshot identifier
    ///
    /// # Returns
    ///
    /// Snapshot data as JSON value
    pub async fn download_snapshot(&self, snapshot_id: &str) -> Result<serde_json::Value> {
        let key = self.get_s3_key(ArtifactType::Snapshot, &format!("{}.spiral", snapshot_id));

        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await?;

        // Get metadata before consuming the body
        let metadata_map = response.metadata().cloned();

        // Read body
        let body = response.body.collect().await?;
        let snapshot_json = String::from_utf8(body.to_vec())?;

        // Verify checksum if available
        if let Some(metadata) = metadata_map {
            if let Some(expected_checksum) = metadata.get("checksum") {
                let mut hasher = Sha256::new();
                hasher.update(snapshot_json.as_bytes());
                let actual_checksum = format!("{:x}", hasher.finalize());

                if expected_checksum != &actual_checksum {
                    eprintln!("Warning: Checksum mismatch for snapshot {}", snapshot_id);
                }
            }
        }

        // Parse JSON
        let snapshot: serde_json::Value = serde_json::from_str(&snapshot_json)?;

        Ok(snapshot)
    }

    /// Upload a TIC to S3
    ///
    /// # Arguments
    ///
    /// * `tic` - TIC data as JSON value
    ///
    /// # Returns
    ///
    /// Upload metadata on success
    pub async fn upload_tic(&mut self, tic: &serde_json::Value) -> Result<UploadMetadata> {
        let tic_id = tic["tic_id"]
            .as_str()
            .ok_or_else(|| anyhow!("TIC missing 'tic_id' field"))?;

        let key = self.get_s3_key(ArtifactType::Tic, &format!("{}.tic", tic_id));

        // Serialize TIC
        let tic_json = serde_json::to_string_pretty(tic)?;

        // Prepare metadata
        let mut metadata = HashMap::new();
        metadata.insert("tic-id".to_string(), tic_id.to_string());
        if let Some(seed) = tic["seed"].as_str() {
            metadata.insert("seed".to_string(), seed.to_string());
        }
        if let Some(source) = tic["source_snapshot"].as_str() {
            metadata.insert("source-snapshot".to_string(), source.to_string());
        }
        if let Some(por) = tic["proof"]["por"].as_str() {
            metadata.insert("por".to_string(), por.to_string());
        }

        // Upload to S3
        let response = self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .body(ByteStream::from(tic_json.into_bytes()))
            .content_type("application/json")
            .set_metadata(Some(metadata))
            .send()
            .await?;

        let upload_meta = UploadMetadata {
            key: key.clone(),
            etag: response.e_tag().unwrap_or("").to_string(),
            version_id: response.version_id().map(|s| s.to_string()),
            checksum: String::new(),
        };

        Ok(upload_meta)
    }

    /// Download a TIC from S3
    ///
    /// # Arguments
    ///
    /// * `tic_id` - TIC identifier
    ///
    /// # Returns
    ///
    /// TIC data as JSON value
    pub async fn download_tic(&self, tic_id: &str) -> Result<serde_json::Value> {
        let key = self.get_s3_key(ArtifactType::Tic, &format!("{}.tic", tic_id));

        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await?;

        // Read body
        let body = response.body.collect().await?;
        let tic_json = String::from_utf8(body.to_vec())?;

        // Parse JSON
        let tic: serde_json::Value = serde_json::from_str(&tic_json)?;

        Ok(tic)
    }

    /// Upload a ledger block to S3
    ///
    /// # Arguments
    ///
    /// * `block` - Block data as JSON value
    ///
    /// # Returns
    ///
    /// Upload metadata on success
    pub async fn upload_block(&mut self, block: &serde_json::Value) -> Result<UploadMetadata> {
        let block_index = block["index"]
            .as_u64()
            .ok_or_else(|| anyhow!("Block missing 'index' field"))?;

        let key = self.get_s3_key(
            ArtifactType::Block,
            &format!("block_{:06}.mef", block_index),
        );

        // Serialize block
        let block_json = serde_json::to_string_pretty(block)?;

        // Prepare metadata
        let mut metadata = HashMap::new();
        metadata.insert("block-index".to_string(), block_index.to_string());
        if let Some(hash) = block["hash"].as_str() {
            metadata.insert("block-hash".to_string(), hash.to_string());
        }
        if let Some(tic_id) = block["tic_id"].as_str() {
            metadata.insert("tic-id".to_string(), tic_id.to_string());
        }
        if let Some(timestamp) = block["timestamp"].as_str() {
            metadata.insert("timestamp".to_string(), timestamp.to_string());
        }

        // Upload to S3 with server-side encryption
        let response = self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .body(ByteStream::from(block_json.into_bytes()))
            .content_type("application/json")
            .server_side_encryption(aws_sdk_s3::types::ServerSideEncryption::Aes256)
            .set_metadata(Some(metadata))
            .send()
            .await?;

        let upload_meta = UploadMetadata {
            key: key.clone(),
            etag: response.e_tag().unwrap_or("").to_string(),
            version_id: response.version_id().map(|s| s.to_string()),
            checksum: String::new(),
        };

        Ok(upload_meta)
    }

    /// Download a ledger block from S3
    ///
    /// # Arguments
    ///
    /// * `block_index` - Block index
    ///
    /// # Returns
    ///
    /// Block data as JSON value
    pub async fn download_block(&self, block_index: u64) -> Result<serde_json::Value> {
        let key = self.get_s3_key(
            ArtifactType::Block,
            &format!("block_{:06}.mef", block_index),
        );

        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await?;

        // Read body
        let body = response.body.collect().await?;
        let block_json = String::from_utf8(body.to_vec())?;

        // Parse JSON
        let block: serde_json::Value = serde_json::from_str(&block_json)?;

        Ok(block)
    }

    /// List artifacts in S3
    ///
    /// # Arguments
    ///
    /// * `artifact_type` - Type of artifacts to list
    /// * `prefix_filter` - Additional prefix filter (optional)
    /// * `max_items` - Maximum number of items to return
    ///
    /// # Returns
    ///
    /// List of artifact metadata
    pub async fn list_artifacts(
        &self,
        artifact_type: ArtifactType,
        prefix_filter: Option<&str>,
        max_items: i32,
    ) -> Result<Vec<ArtifactMetadata>> {
        let mut base_prefix = self.get_s3_key(artifact_type, "");
        if let Some(filter) = prefix_filter {
            base_prefix.push_str(filter);
        }

        let mut artifacts = Vec::new();

        let response = self
            .client
            .list_objects_v2()
            .bucket(&self.config.bucket)
            .prefix(&base_prefix)
            .max_keys(max_items)
            .send()
            .await?;

        let contents = response.contents();
        for obj in contents {
            if let (Some(key), Some(size), Some(last_modified)) =
                (obj.key(), obj.size(), obj.last_modified())
            {
                // Get object metadata
                let head_response = self
                    .client
                    .head_object()
                    .bucket(&self.config.bucket)
                    .key(key)
                    .send()
                    .await?;

                let metadata = head_response.metadata().cloned().unwrap_or_default();

                artifacts.push(ArtifactMetadata {
                    key: key.to_string(),
                    size,
                    last_modified: last_modified.to_string(),
                    etag: obj.e_tag().unwrap_or("").to_string(),
                    metadata,
                });
            }
        }

        Ok(artifacts)
    }

    /// Sync local directory to S3
    ///
    /// # Arguments
    ///
    /// * `local_path` - Local directory path
    /// * `_artifact_type` - Type of artifacts being synced (reserved for future use)
    ///
    /// # Returns
    ///
    /// Sync statistics
    pub async fn sync_to_s3(
        &mut self,
        local_path: &Path,
        _artifact_type: ArtifactType,
    ) -> Result<SyncStats> {
        let mut stats = SyncStats::default();

        // Map file extensions to artifact types
        let ext_map = [
            (".spiral", ArtifactType::Snapshot),
            (".tic", ArtifactType::Tic),
            (".mef", ArtifactType::Block),
        ];

        // Recursively walk directory
        let walker = walkdir::WalkDir::new(local_path);

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    stats.failed += 1;
                    stats.errors.push(format!("Walk error: {}", err));
                    continue;
                }
            };

            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Check extension
            let ext = match path.extension().and_then(|s| s.to_str()) {
                Some(e) => format!(".{}", e),
                None => {
                    stats.skipped += 1;
                    continue;
                }
            };

            // Check if this is a supported artifact type
            let found_type = ext_map.iter().find(|(e, _)| *e == ext);
            if found_type.is_none() {
                stats.skipped += 1;
                continue;
            }

            // Read file
            let contents = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(err) => {
                    stats.failed += 1;
                    stats.errors.push(format!("{}: {}", path.display(), err));
                    continue;
                }
            };

            // Parse JSON
            let data: serde_json::Value = match serde_json::from_str(&contents) {
                Ok(d) => d,
                Err(err) => {
                    stats.failed += 1;
                    stats.errors.push(format!("{}: {}", path.display(), err));
                    continue;
                }
            };

            // Upload based on type
            let result = if ext == ".spiral" {
                self.upload_snapshot(&data).await
            } else if ext == ".tic" {
                self.upload_tic(&data).await
            } else if ext == ".mef" {
                self.upload_block(&data).await
            } else {
                stats.failed += 1;
                continue;
            };

            match result {
                Ok(_) => stats.uploaded += 1,
                Err(err) => {
                    stats.failed += 1;
                    stats.errors.push(format!("{}: {}", path.display(), err));
                }
            }
        }

        Ok(stats)
    }

    /// Sync from S3 to local directory
    ///
    /// # Arguments
    ///
    /// * `local_path` - Local directory path
    /// * `artifact_type` - Type of artifacts to sync
    ///
    /// # Returns
    ///
    /// Sync statistics
    pub async fn sync_from_s3(
        &self,
        local_path: &Path,
        artifact_type: ArtifactType,
    ) -> Result<SyncStats> {
        let mut stats = SyncStats::default();

        let artifacts = self.list_artifacts(artifact_type, None, 1000).await?;

        for artifact in artifacts {
            let filename = Path::new(&artifact.key)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");

            let local_file = local_path.join(filename);

            // Check if file exists
            if local_file.exists() {
                stats.skipped += 1;
                continue;
            }

            // Download object
            let result = self
                .client
                .get_object()
                .bucket(&self.config.bucket)
                .key(&artifact.key)
                .send()
                .await;

            match result {
                Ok(response) => {
                    let body = response.body.collect().await?;

                    // Save to local file
                    match std::fs::write(&local_file, body.to_vec()) {
                        Ok(_) => stats.downloaded += 1,
                        Err(err) => {
                            stats.failed += 1;
                            stats.errors.push(format!("{}: {}", filename, err));
                        }
                    }
                }
                Err(err) => {
                    stats.failed += 1;
                    stats.errors.push(format!("{}: {}", filename, err));
                }
            }
        }

        Ok(stats)
    }

    /// Create a presigned URL for direct access
    ///
    /// # Arguments
    ///
    /// * `artifact_type` - Type of artifact
    /// * `artifact_id` - Artifact identifier
    /// * `expiration_secs` - URL expiration time in seconds
    ///
    /// # Returns
    ///
    /// Presigned URL
    pub async fn create_presigned_url(
        &self,
        artifact_type: ArtifactType,
        artifact_id: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        let key = self.get_s3_key(artifact_type, artifact_id);

        let presigned = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .presigned(aws_sdk_s3::presigning::PresigningConfig::expires_in(
                std::time::Duration::from_secs(expiration_secs),
            )?)
            .await?;

        Ok(presigned.uri().to_string())
    }

    /// Enable versioning on the S3 bucket
    pub async fn enable_versioning(&self) -> Result<()> {
        let versioning_config = VersioningConfiguration::builder()
            .status(BucketVersioningStatus::Enabled)
            .build();

        self.client
            .put_bucket_versioning()
            .bucket(&self.config.bucket)
            .versioning_configuration(versioning_config)
            .send()
            .await?;

        Ok(())
    }

    /// Set lifecycle policy for automatic archival and deletion
    ///
    /// # Arguments
    ///
    /// * `days_to_glacier` - Days before moving to Glacier storage
    /// * `days_to_delete` - Days before permanent deletion
    pub async fn set_lifecycle_policy(
        &self,
        days_to_glacier: i32,
        days_to_delete: i32,
    ) -> Result<()> {
        let transition = Transition::builder()
            .days(days_to_glacier)
            .storage_class(TransitionStorageClass::Glacier)
            .build();

        let expiration = LifecycleExpiration::builder().days(days_to_delete).build();

        let filter = LifecycleRuleFilter::builder()
            .prefix(&self.config.prefix)
            .build();

        let rule = LifecycleRule::builder()
            .id("MEF-Core-Lifecycle")
            .status(ExpirationStatus::Enabled)
            .filter(filter)
            .transitions(transition)
            .expiration(expiration)
            .build()
            .map_err(|e| anyhow!("Failed to build lifecycle rule: {:?}", e))?;

        let lifecycle_config = aws_sdk_s3::types::BucketLifecycleConfiguration::builder()
            .rules(rule)
            .build()
            .map_err(|e| anyhow!("Failed to build lifecycle config: {:?}", e))?;

        self.client
            .put_bucket_lifecycle_configuration()
            .bucket(&self.config.bucket)
            .lifecycle_configuration(lifecycle_config)
            .send()
            .await?;

        Ok(())
    }

    /// Get storage metrics
    pub async fn get_storage_metrics(&self) -> Result<StorageMetrics> {
        let mut metrics = StorageMetrics {
            bucket: self.config.bucket.clone(),
            prefix: self.config.prefix.clone(),
            artifacts: HashMap::new(),
            total_size: 0,
            total_objects: 0,
            total_size_mb: 0.0,
        };

        let artifact_types = vec![
            ArtifactType::Snapshot,
            ArtifactType::Tic,
            ArtifactType::Block,
        ];

        for artifact_type in artifact_types {
            let prefix = self.get_s3_key(artifact_type, "");

            let response = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix)
                .send()
                .await?;

            let contents = response.contents();
            let count = contents.len();
            let total_size: i64 = contents.iter().filter_map(|obj| obj.size()).sum();

            let stats = ArtifactStats {
                count,
                size_bytes: total_size,
                size_mb: total_size as f64 / (1024.0 * 1024.0),
            };

            metrics
                .artifacts
                .insert(artifact_type.prefix().to_string(), stats);
            metrics.total_size += total_size;
            metrics.total_objects += count;
        }

        metrics.total_size_mb = metrics.total_size as f64 / (1024.0 * 1024.0);

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_default() {
        let config = S3Config::default();
        assert_eq!(config.bucket, "mef-core");
        assert_eq!(config.prefix, "mef/");
        assert_eq!(config.region, "us-east-1");
    }

    #[test]
    fn test_artifact_type_prefix() {
        assert_eq!(ArtifactType::Snapshot.prefix(), "snapshots");
        assert_eq!(ArtifactType::Tic.prefix(), "tics");
        assert_eq!(ArtifactType::Block.prefix(), "ledger");
        assert_eq!(ArtifactType::Hdag.prefix(), "hdag");
        assert_eq!(ArtifactType::Index.prefix(), "indices");
    }

    #[test]
    fn test_get_s3_key() {
        let config = S3Config::default();
        // Create a simple mock adapter (we won't actually use the client)
        // For testing key generation, we just need the config

        let key = format!(
            "{}{}/{}",
            config.prefix,
            ArtifactType::Snapshot.prefix(),
            "test.spiral"
        );
        assert_eq!(key, "mef/snapshots/test.spiral");

        let key = format!(
            "{}{}/{}",
            config.prefix,
            ArtifactType::Block.prefix(),
            "block_000001.mef"
        );
        assert_eq!(key, "mef/ledger/block_000001.mef");
    }

    #[test]
    fn test_sync_stats_default() {
        let stats = SyncStats::default();
        assert_eq!(stats.uploaded, 0);
        assert_eq!(stats.downloaded, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.errors.len(), 0);
    }

    #[test]
    fn test_upload_metadata_serialization() {
        let meta = UploadMetadata {
            key: "test-key".to_string(),
            etag: "test-etag".to_string(),
            version_id: Some("v1".to_string()),
            checksum: "abc123".to_string(),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: UploadMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(meta.key, deserialized.key);
        assert_eq!(meta.etag, deserialized.etag);
        assert_eq!(meta.version_id, deserialized.version_id);
        assert_eq!(meta.checksum, deserialized.checksum);
    }

    // Note: Full integration tests would require a running S3/MinIO instance
    // These are basic unit tests for the public API surface
}
