#!/usr/bin/env bash
# Launch AWS spot instances, build the RNG project, run benchmarks, collect results.
#
# Prerequisites:
#   - AWS CLI v2 installed and configured (aws configure)
#   - Sufficient EC2 permissions (RunInstances, TerminateInstances, CreateKeyPair,
#     CreateSecurityGroup, AuthorizeSecurityGroupIngress, DescribeInstances, etc.)
#
# Usage:
#   ./bench.sh                                          # defaults: c6i, c7i, c7a
#   ./bench.sh --instances c6i.xlarge,c7i.xlarge        # custom instance types
#   ./bench.sh --keep                                   # don't terminate on completion
#   ./bench.sh --region us-west-2                       # override region

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_ID"

KEY_NAME="rng-bench"
KEY_FILE="$SCRIPT_DIR/.${KEY_NAME}.pem"
SG_NAME="rng-bench-ssh"
SSH_USER="ubuntu"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"

# Default instance types — all x86-64 with AVX-512
INSTANCE_TYPES=("c6i.xlarge" "c7i.xlarge" "c7a.xlarge")
KEEP_INSTANCES=false
REGION_FLAG=()

# Instance tracking directory (for cleanup on failure/interrupt)
TRACK_DIR=$(mktemp -d)

# ── Helpers ───────────────────────────────────────────────────────────────────

die()  { echo "FATAL: $*" >&2; exit 1; }
log()  { echo "[$(date +%H:%M:%S)] $*" >&2; }

usage() {
    cat <<'EOF'
Usage: bench.sh [OPTIONS]

Options:
  --instances TYPE[,TYPE...]  Comma-separated instance types (default: c6i.xlarge,c7i.xlarge,c7a.xlarge)
  --region REGION             AWS region override
  --keep                      Don't terminate instances after benchmarking
  --help                      Show this help

Instance type suggestions (x86-64 with AVX-512):
  c6i.xlarge    Intel Ice Lake           (AVX-512F/VL/BW/DQ/VNNI)
  c7i.xlarge    Intel Sapphire Rapids    (AVX-512F/VL/BW/DQ/VNNI/FP16/IFMA/VBMI2)
  c7a.xlarge    AMD EPYC Zen 4           (AVX-512F/VL/BW/DQ/VNNI/VBMI2)
  c5.xlarge     Intel Skylake/Cascade    (AVX-512F/VL/BW/DQ)

For AVX2-only comparison:
  c6a.xlarge    AMD EPYC Zen 3           (AVX2 only, no AVX-512)

Examples:
  ./bench.sh
  ./bench.sh --instances c6i.xlarge,c7i.xlarge,c7a.xlarge,c5.xlarge
  ./bench.sh --instances c7i.xlarge --region us-east-1 --keep
EOF
}

cleanup() {
    shopt -s nullglob
    if [[ "$KEEP_INSTANCES" == "true" ]]; then
        local remaining=()
        for f in "$TRACK_DIR"/*.id; do
            [[ -f "$f" ]] && remaining+=("$(basename "$f" .id): $(cat "$f")")
        done
        if [[ ${#remaining[@]} -gt 0 ]]; then
            log "Keeping instances alive (--keep):"
            for r in "${remaining[@]}"; do log "  $r"; done
        fi
    else
        local ids=()
        for f in "$TRACK_DIR"/*.id; do
            [[ -f "$f" ]] && ids+=("$(cat "$f")")
        done
        if [[ ${#ids[@]} -gt 0 ]]; then
            log "Terminating ${#ids[@]} remaining instance(s): ${ids[*]}"
            aws ec2 terminate-instances "${REGION_FLAG[@]}" \
                --instance-ids "${ids[@]}" >/dev/null 2>&1 || true
        fi
    fi
    rm -rf "$TRACK_DIR"
}
trap cleanup EXIT INT TERM

# ── Arg parsing ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --instances) shift; IFS=',' read -ra INSTANCE_TYPES <<< "$1" ;;
        --region)    shift; export AWS_DEFAULT_REGION="$1"; REGION_FLAG=(--region "$1") ;;
        --keep)      KEEP_INSTANCES=true ;;
        --help)      usage; exit 0 ;;
        *)           die "Unknown option: $1 (try --help)" ;;
    esac
    shift
done

# ── Prerequisites ─────────────────────────────────────────────────────────────

command -v aws   >/dev/null || die "AWS CLI not found. Install: https://aws.amazon.com/cli/"
command -v rsync >/dev/null || die "rsync not found. Install: sudo apt install rsync"
aws sts get-caller-identity "${REGION_FLAG[@]}" >/dev/null 2>&1 \
    || die "AWS credentials not configured. Run: aws configure"

# ── Resolve latest Amazon Linux 2023 AMI ──────────────────────────────────────

log "Resolving latest Ubuntu 24.04 AMI..."
AMI_ID=$(aws ec2 describe-images "${REGION_FLAG[@]}" \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd*/ubuntu-noble-24.04-amd64-server-*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text 2>/dev/null) \
    || die "Failed to resolve Ubuntu 24.04 AMI."
[[ -z "$AMI_ID" || "$AMI_ID" == "None" ]] && die "No Ubuntu 24.04 AMI found in this region."
log "AMI: $AMI_ID"

# ── Key pair ──────────────────────────────────────────────────────────────────

if [[ ! -f "$KEY_FILE" ]]; then
    log "Creating EC2 key pair '$KEY_NAME'..."
    aws ec2 delete-key-pair "${REGION_FLAG[@]}" --key-name "$KEY_NAME" 2>/dev/null || true
    aws ec2 create-key-pair "${REGION_FLAG[@]}" \
        --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
else
    log "Using existing key: $KEY_FILE"
    if ! aws ec2 describe-key-pairs "${REGION_FLAG[@]}" \
            --key-names "$KEY_NAME" >/dev/null 2>&1; then
        log "Key pair '$KEY_NAME' missing from AWS — recreating..."
        rm -f "$KEY_FILE"
        aws ec2 create-key-pair "${REGION_FLAG[@]}" \
            --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
    fi
fi

# ── Security group ────────────────────────────────────────────────────────────

VPC_ID=$(aws ec2 describe-vpcs "${REGION_FLAG[@]}" \
    --filters Name=is-default,Values=true \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
[[ -z "$VPC_ID" || "$VPC_ID" == "None" ]] && die "No default VPC found in this region"

SG_ID=$(aws ec2 describe-security-groups "${REGION_FLAG[@]}" \
    --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
    log "Creating security group '$SG_NAME'..."
    SG_ID=$(aws ec2 create-security-group "${REGION_FLAG[@]}" \
        --group-name "$SG_NAME" \
        --description "SSH access for RNG benchmarks" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text)
fi

MY_IP=$(curl -sf https://checkip.amazonaws.com) || die "Cannot determine public IP"
log "Authorizing SSH from $MY_IP..."
aws ec2 authorize-security-group-ingress "${REGION_FLAG[@]}" \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MY_IP}/32" 2>/dev/null || true

# ── Results directory ─────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"

# ── Benchmark one instance type ───────────────────────────────────────────────

bench_one() {
    local itype=$1
    local result_file="$RESULTS_DIR/${itype}.txt"
    local log_file="$RESULTS_DIR/${itype}.log"

    # ── Launch spot instance ──
    log "[$itype] Requesting spot instance..."
    local instance_id
    instance_id=$(aws ec2 run-instances "${REGION_FLAG[@]}" \
        --image-id "$AMI_ID" \
        --instance-type "$itype" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rng-bench-${itype}-${RUN_ID}}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1) || {
        echo "FAILED to launch: $instance_id" > "$result_file"
        log "[$itype] Launch failed: $instance_id"
        return 1
    }

    # Track for cleanup
    echo "$instance_id" > "$TRACK_DIR/${itype}.id"
    log "[$itype] Launched $instance_id"

    # ── Wait for running ──
    log "[$itype] Waiting for instance..."
    aws ec2 wait instance-running "${REGION_FLAG[@]}" --instance-ids "$instance_id" || {
        echo "Instance failed to reach running state" > "$result_file"
        return 1
    }

    local public_ip
    public_ip=$(aws ec2 describe-instances "${REGION_FLAG[@]}" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    log "[$itype] IP: $public_ip"

    # ── Wait for SSH ──
    log "[$itype] Waiting for SSH..."
    local attempt
    for attempt in $(seq 1 30); do
        if ssh $SSH_OPTS -i "$KEY_FILE" "$SSH_USER@$public_ip" true 2>/dev/null; then
            break
        fi
        if [[ $attempt -eq 30 ]]; then
            echo "SSH connection timeout after 150s" > "$result_file"
            return 1
        fi
        sleep 5
    done

    # ── Upload source ──
    log "[$itype] Uploading source..."
    rsync -az --delete \
        --exclude='build*' --exclude='.git' --exclude='aws/results' --exclude='aws/.*.pem' \
        --exclude='.cache' --exclude='compile_commands.json' \
        -e "ssh $SSH_OPTS -i $KEY_FILE" \
        "$PROJECT_DIR/" "$SSH_USER@$public_ip:~/rng/" >> "$log_file" 2>&1

    # ── Build & benchmark ──
    log "[$itype] Building & benchmarking (this takes a few minutes)..."
    ssh $SSH_OPTS -i "$KEY_FILE" "$SSH_USER@$public_ip" \
        'bash ~/rng/aws/remote-setup.sh' > "$result_file" 2>> "$log_file"

    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "[$itype] Remote script failed (rc=$rc). See $log_file"
        return 1
    fi

    log "[$itype] Done -> $result_file"

    # ── Terminate ──
    if [[ "$KEEP_INSTANCES" == "false" ]]; then
        aws ec2 terminate-instances "${REGION_FLAG[@]}" \
            --instance-ids "$instance_id" >/dev/null 2>&1 || true
        rm -f "$TRACK_DIR/${itype}.id"
        log "[$itype] Terminated $instance_id"
    fi
}

# ── Launch all in parallel ────────────────────────────────────────────────────

log "Benchmarking ${#INSTANCE_TYPES[@]} instance type(s): ${INSTANCE_TYPES[*]}"
log "Results will be saved to: $RESULTS_DIR/"
echo ""

PIDS=()
for itype in "${INSTANCE_TYPES[@]}"; do
    bench_one "$itype" &
    PIDS+=($!)
done

# Wait for all and track failures
FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        ((FAILURES++)) || true
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " RNG Benchmark Results — $RUN_ID"
echo "============================================================"

for itype in "${INSTANCE_TYPES[@]}"; do
    result_file="$RESULTS_DIR/${itype}.txt"
    if [[ -f "$result_file" ]]; then
        echo ""
        echo "──── $itype ────"
        cat "$result_file"
    else
        echo ""
        echo "──── $itype ──── (no results)"
    fi
done

echo ""
echo "Results saved to: $RESULTS_DIR/"
if [[ $FAILURES -gt 0 ]]; then
    echo "WARNING: $FAILURES instance(s) failed. Check .log files for details."
fi

exit "$FAILURES"
