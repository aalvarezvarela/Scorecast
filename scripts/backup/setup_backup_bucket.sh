#!/usr/bin/env bash
#
# Create and configure the dedicated backup bucket.
#
# Backups used to share the model-registry bucket, which meant one bad
# lifecycle rule or an over-broad delete could take out the models *and* the
# only copy of the data. This gives them separate blast radii.
#
# Retention is tiered rather than deduplicated. Weekly in-season runs pile up,
# but the data changes every run (measured: 0 of 11 line-history files were
# byte-identical between two backups three weeks apart), so dedup would save
# nothing while making each date folder non-self-contained. Instead the backup
# code tags the first run of each month `retention=monthly` and the rest
# `retention=weekly`, and the lifecycle rules below act on those tags.
#
# Needs admin credentials -- the CI role cannot configure buckets.
#
# Usage:
#   scripts/backup/setup_backup_bucket.sh --dry-run     # print the plan
#   scripts/backup/setup_backup_bucket.sh               # apply it
#   scripts/backup/setup_backup_bucket.sh --migrate     # also copy existing history

set -euo pipefail

BUCKET="${S3_BACKUP_BUCKET:-adrian-nba-backups-eu-west-1}"
OLD_BUCKET="${S3_OLD_BUCKET:-adrian-nba-model-registry-eu-west-1}"
REGION="${AWS_REGION:-eu-west-1}"
PROFILE="${S3_AWS_PROFILE:-adrian-personal}"

# How long each tier lives. Weekly runs give dense recent coverage; the
# monthly ones are the deep history.
WEEKLY_EXPIRE_DAYS=90
MONTHLY_GLACIER_DAYS=90
MONTHLY_EXPIRE_DAYS=1095   # 3 years
NONCURRENT_EXPIRE_DAYS=30

DRY_RUN=0
MIGRATE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --migrate) MIGRATE=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

AWS=(aws --region "$REGION")
[ -n "$PROFILE" ] && AWS+=(--profile "$PROFILE")

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '  [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

echo "bucket : s3://$BUCKET  (region $REGION)"
echo "profile: ${PROFILE:-<ambient>}"
echo

# ---------------------------------------------------------------------------
# 1. The bucket itself
# ---------------------------------------------------------------------------
if "${AWS[@]}" s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "1. bucket already exists, leaving it alone"
else
    echo "1. creating bucket"
    run "${AWS[@]}" s3api create-bucket \
        --bucket "$BUCKET" \
        --create-bucket-configuration "LocationConstraint=$REGION"
fi

# ---------------------------------------------------------------------------
# 2. Versioning: an overwrite or delete stays recoverable
# ---------------------------------------------------------------------------
echo "2. enabling versioning"
run "${AWS[@]}" s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled

# ---------------------------------------------------------------------------
# 3. Encryption and public access
# ---------------------------------------------------------------------------
echo "3. enabling default encryption, blocking public access"
run "${AWS[@]}" s3api put-bucket-encryption \
    --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

run "${AWS[@]}" s3api put-public-access-block \
    --bucket "$BUCKET" \
    --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

# ---------------------------------------------------------------------------
# 4. Tiered retention, driven by the tags the backup code writes
# ---------------------------------------------------------------------------
echo "4. applying lifecycle rules"
LIFECYCLE=$(cat <<JSON
{
  "Rules": [
    {
      "ID": "expire-weekly-backups",
      "Status": "Enabled",
      "Filter": {"Tag": {"Key": "retention", "Value": "weekly"}},
      "Expiration": {"Days": $WEEKLY_EXPIRE_DAYS}
    },
    {
      "ID": "archive-then-expire-monthly-backups",
      "Status": "Enabled",
      "Filter": {"Tag": {"Key": "retention", "Value": "monthly"}},
      "Transitions": [
        {"Days": $MONTHLY_GLACIER_DAYS, "StorageClass": "GLACIER_IR"}
      ],
      "Expiration": {"Days": $MONTHLY_EXPIRE_DAYS}
    },
    {
      "ID": "expire-old-noncurrent-versions",
      "Status": "Enabled",
      "Filter": {},
      "NoncurrentVersionExpiration": {"NoncurrentDays": $NONCURRENT_EXPIRE_DAYS}
    },
    {
      "ID": "abort-incomplete-multipart-uploads",
      "Status": "Enabled",
      "Filter": {},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    }
  ]
}
JSON
)
if [ "$DRY_RUN" = "1" ]; then
    printf '  [dry-run] put-bucket-lifecycle-configuration:\n%s\n' "$LIFECYCLE"
else
    "${AWS[@]}" s3api put-bucket-lifecycle-configuration \
        --bucket "$BUCKET" \
        --lifecycle-configuration "$LIFECYCLE"
fi

# ---------------------------------------------------------------------------
# 5. Existing history
# ---------------------------------------------------------------------------
if [ "$MIGRATE" = "1" ]; then
    echo "5. copying existing backups across (server-side, nothing downloads)"
    # --copy-props metadata-directive stops the CLI reading each source
    # object's tags. Multipart copies (anything over 8 MB) otherwise call
    # GetObjectTagging on the source, which silently failed for every large
    # table and left the migration 261 MB short. We tag the copies ourselves
    # below, so the source tags are not wanted anyway.
    run "${AWS[@]}" s3 sync \
        "s3://$OLD_BUCKET/backups/" "s3://$BUCKET/backups/" \
        --copy-props metadata-directive

    # A sync that copies most of the data and fails on the rest still looks
    # like progress. Nothing should be deleted from the old bucket until the
    # two inventories actually agree, so compare them here.
    if [ "$DRY_RUN" != "1" ]; then
        echo
        echo "   verifying the copy is complete …"
        src_count=$("${AWS[@]}" s3 ls "s3://$OLD_BUCKET/backups/" --recursive | wc -l)
        dst_count=$("${AWS[@]}" s3 ls "s3://$BUCKET/backups/" --recursive | wc -l)
        echo "   source: $src_count objects / destination: $dst_count objects"
        if [ "$src_count" != "$dst_count" ]; then
            echo
            echo "   MIGRATION INCOMPLETE -- do not delete anything from the" >&2
            echo "   old bucket. Re-run with --migrate to retry." >&2
            exit 1
        fi
        echo "   inventories match."
    fi

    echo
    echo "   Migrated objects carry no retention tag, so no lifecycle rule"
    echo "   matches them and they are kept indefinitely. Tag them with:"
    echo "     python scripts/backup/tag_existing_backups.py --bucket $BUCKET"
else
    echo "5. skipping migration (pass --migrate to copy existing backups)"
fi

echo
echo "Done."
echo
echo "Still to do by hand, because it needs the role ARNs:"
echo "  * grant the CI OIDC role s3:PutObject/PutObjectTagging/GetObject/"
echo "    ListBucket on s3://$BUCKET"
echo "  * deny s3:DeleteObject to every role except an admin one, so the"
echo "    prediction pipeline cannot reach the backups at all"
echo "  * once verified, remove backups/ from s3://$OLD_BUCKET"
