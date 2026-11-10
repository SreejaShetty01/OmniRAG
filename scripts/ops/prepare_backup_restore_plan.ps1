param(
    [Parameter(Mandatory = $true)][string]$SourceProject,
    [Parameter(Mandatory = $true)][string]$RestoreProject,
    [Parameter(Mandatory = $true)][string]$SourceCollection,
    [Parameter(Mandatory = $true)][string]$RestoreCollection,
    [Parameter(Mandatory = $true)][string]$SourceApiVolume,
    [Parameter(Mandatory = $true)][string]$RestoreApiVolume,
    [Parameter(Mandatory = $true)][string]$SourceQdrantVolume,
    [Parameter(Mandatory = $true)][string]$RestoreQdrantVolume
)

$ErrorActionPreference = 'Stop'

if ($RestoreProject -eq $SourceProject -or
    $RestoreCollection -eq $SourceCollection -or
    $RestoreApiVolume -eq $SourceApiVolume -or
    $RestoreQdrantVolume -eq $SourceQdrantVolume) {
    throw 'Restore project, collection, API volume, and Qdrant volume must all differ from source identities.'
}
if (-not $RestoreCollection.StartsWith('omnirag_eval_restore_', [StringComparison]::Ordinal)) {
    throw 'Restore collection must start with omnirag_eval_restore_.'
}

# This is intentionally a plan printer. It performs no Docker, Qdrant, SQLite,
# filesystem, or network operation. The emitted placeholders must be reviewed
# and filled from approved runtime labels before a separately approved drill.
$plan = [ordered]@{
    status = 'PREPARED_NOT_RUN'
    approval_required = $true
    source = [ordered]@{
        compose_project = $SourceProject
        collection = $SourceCollection
        api_volume = $SourceApiVolume
        qdrant_volume = $SourceQdrantVolume
    }
    restore = [ordered]@{
        compose_project = $RestoreProject
        collection = $RestoreCollection
        api_volume = $RestoreApiVolume
        qdrant_volume = $RestoreQdrantVolume
    }
    steps = @(
        'Record source container IDs, collection schema/count, volume IDs, dataset/snapshot checksums, and SQLite integrity result.',
        'Create a Qdrant snapshot of only the source evaluation collection; hash the snapshot file.',
        'Create a transaction-consistent SQLite online backup and hash the database plus associated uploaded source files.',
        'Render an isolated restore Compose project using only the four distinct restore identities; inspect it before startup.',
        'Restore Qdrant snapshot and SQLite backup into new restore volumes and a new restore collection.',
        'Validate SQLite integrity, Qdrant schema/count/indexes, source hashes, and readiness.',
        'Run five approved reference queries with signed identities; compare ordered top-K IDs and allow/deny ACL outcomes.',
        'Measure RPO from capture timestamps and RTO through all checks; preserve the resulting audit record.'
    )
    unresolved_runtime_values = @(
        'source API and Qdrant container names/IDs', 'Qdrant API URL and snapshot storage path',
        'SQLite file path and upload directory', 'restore Compose files and ports',
        'five approved reference query IDs and expected ACL outcomes'
    )
}
$plan | ConvertTo-Json -Depth 6
